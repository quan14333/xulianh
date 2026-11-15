"""Guarded Development trainer. PROVISIONAL runs are capped at two microbatches."""
from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader

from .checkpoint import (atomic_json, environment, resume_training, save_training_checkpoint,
                         sha256, source_snapshot, state_digest, strict_load)
from .config import authorize, load_config, new_output_directory
from .dataset import PairDataset, audit_inputs, load_protocol
from .losses import FusionLoss
from .model import fused_output, initial_pair, make_student, training_mode


def configure_runtime(cfg, device=None):
    t = cfg["training"]
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(t["seed"])
    np.random.seed(t["seed"])
    torch.manual_seed(t["seed"])
    torch.set_num_threads(t["cpu_threads"])
    torch.use_deterministic_algorithms(t["deterministic"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = t["deterministic"]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    selected = torch.device(device or t["device"])
    if selected.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
        torch.cuda.manual_seed_all(t["seed"])
    elif selected.type != "cpu":
        raise ValueError("Only CPU/CUDA are supported")
    return selected


def build_optimizer(model, cfg):
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable parameters")
    opt = cfg["optimizer"]
    return getattr(torch.optim, opt["name"])(
        params, lr=opt["lr"], betas=tuple(opt["betas"]),
        eps=opt["eps"], weight_decay=opt["weight_decay"])


def build_scheduler(optimizer, cfg, n_images):
    t = cfg["training"]
    batches = math.ceil(n_images / t["microbatch"])
    updates = math.ceil(batches / t["accumulation"]) * t["epochs"]
    if cfg["scheduler"]["name"] == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=updates, eta_min=cfg["scheduler"]["eta_min"])


def epoch_order(n_images, seed, epoch):
    generator = torch.Generator().manual_seed(seed + epoch)
    return torch.randperm(n_images, generator=generator).tolist()


def gradient_norm(model):
    total = 0.0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            if p.grad is not None:
                raise RuntimeError(f"Frozen parameter has a gradient: {name}")
            continue
        if p.grad is None:
            raise RuntimeError(f"Missing gradient: {name}")
        if not torch.isfinite(p.grad).all().item():
            raise FloatingPointError(f"Nonfinite gradient: {name}")
        total += float(p.grad.detach().double().square().sum().item())
    norm = math.sqrt(total)
    if not math.isfinite(norm) or norm <= 0:
        raise FloatingPointError("Gradient is zero/nonfinite for the whole model")
    return norm


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def memory_record(device, cfg):
    if device.type != "cuda":
        return {"peak_allocated_bytes": None, "peak_reserved_bytes": None, "free_bytes": None}
    free, _ = torch.cuda.mem_get_info(device)
    peak = torch.cuda.max_memory_allocated(device)
    result = {"peak_allocated_bytes": peak,
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(device), "free_bytes": free}
    if peak > cfg["training"]["peak_limit_gib"] * 2**30:
        raise MemoryError(f"Peak GPU allocation exceeded configured cap: {result}")
    if free < cfg["training"]["min_free_gib"] * 2**30:
        raise MemoryError(f"Free GPU memory below configured guard: {result}")
    return result


def validate(model, reference, records, objective, device):
    model.eval()
    totals = {k: 0.0 for k in ("total", "intensity", "gradient", "anchor")}
    with torch.no_grad():
        for sample in PairDataset(records, training=False):
            vis, ir = sample["vis"][None].to(device), sample["ir"][None].to(device)
            ref = fused_output(reference, vis, ir)
            output = fused_output(model, vis, ir)
            parts = objective(output, vis, ir, ref)
            for k in totals:
                totals[k] += parts[k].item()
    return {k: v / len(records) for k, v in totals.items()}


def write_history(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(cfg, output, *, mode="smoke", smoke_batches=2, resume=None, device=None):
    authorize(cfg, mode, smoke_batches)  # Must happen BEFORE data/model/output work.
    if mode == "train" and device and device != cfg["training"]["device"]:
        raise PermissionError("Full-training device override is not in the approved config")
    run_dir = new_output_directory(output, cfg)
    started = time.perf_counter()
    try:
        selected = configure_runtime(cfg, device)
        atomic_json(run_dir / "config.json", cfg.snapshot())
        atomic_json(run_dir / "environment.json", environment(selected))
        train_records, dev_records = load_protocol(cfg)
        inventory_hash = audit_inputs(train_records + dev_records, run_dir / "input_audit.csv")
        provenance = {
            "source_hashes": source_snapshot(cfg),
            "input_inventory_sha256": inventory_hash,
            "pretrained_sha256": cfg["model"]["pretrained_sha256"],
            "train_manifest_sha256": cfg["data"]["train_sha256"],
            "dev_manifest_sha256": cfg["data"]["dev_sha256"],
            "device": str(selected),
        }
        student, reference = initial_pair(cfg)
        init_digest = state_digest(student.state_dict())
        student.to(selected)
        reference.to(selected)
        if selected.type == "cuda":
            torch.cuda.reset_peak_memory_stats(selected)
        t = cfg["training"]
        dataset = PairDataset(train_records, training=True, crop_hw=t["crop_hw"],
                              hflip_p=t["hflip_p"], seed=t["seed"])
        probe = dataset[0]
        student.eval()
        with torch.no_grad():
            v, i = probe["vis"][None].to(selected), probe["ir"][None].to(selected)
            s0, r0 = fused_output(student, v, i), fused_output(reference, v, i)
            if not torch.equal(s0, r0):
                raise RuntimeError("Zero-step Student/reference output mismatch")
        atomic_json(run_dir / "initialization.json", {
            "status": "PASS", "strict": True, "each_tensor_equal": True,
            "state_sha256": init_digest, "reference_sha256": init_digest,
            "zero_step_equal": True, "zero_step_range": [s0.min().item(), s0.max().item()],
            "parameters": sum(p.numel() for p in student.parameters()),
            "state_entries": len(student.state_dict()),
            "trainable_names": [n for n, p in student.named_parameters() if p.requires_grad],
        })
        del s0, r0, v, i
        optimizer = build_optimizer(student, cfg)
        scheduler = build_scheduler(optimizer, cfg, len(dataset))
        objective = FusionLoss(**cfg["loss"]).to(selected)
        progress = {"epoch": 0, "sample_cursor": 0, "global_step": 0}
        best_loss, history = float("inf"), []
        if resume:
            old = resume_training(resume, student, optimizer, scheduler,
                                  config_hash=cfg.fingerprint(), provenance=provenance, mode=mode)
            progress, best_loss, history = old["progress"], old["best_loss"], old["history"]
            if not (0 <= progress["epoch"] < t["epochs"] and 0 <= progress["sample_cursor"] < len(dataset)):
                raise ValueError("Resume progress is invalid or training is already complete")
            # Carry the auxiliary best artifact; never overwrite the source run.
            from .checkpoint import read_checkpoint
            previous_best = Path(resume).resolve().parent / "best.pt"
            _, best_meta = read_checkpoint(previous_best)
            if (best_meta.get("config_hash") != cfg.fingerprint()
                    or best_meta.get("provenance") != provenance
                    or best_meta.get("best_loss") != best_loss):
                raise ValueError("Resume requires the matching companion best.pt")
            import shutil
            shutil.copyfile(previous_best, run_dir / "best.pt")
        reference_digest = state_digest(reference.state_dict())
        fixed_names = {n for n, p in student.named_parameters() if not p.requires_grad}
        if cfg["model"]["bn_running_stats"] == "frozen_pretrained":
            fixed_names |= {n for n, _ in student.named_buffers()}
        fixed_before = {n: student.state_dict()[n].detach().cpu().clone() for n in fixed_names}
        batch_logs = []
        compute_seconds = 0.0
        microbatches_processed = 0
        stop = False
        for epoch in range(progress["epoch"], t["epochs"]):
            dataset.set_epoch(epoch)
            cursor = progress["sample_cursor"] if epoch == progress["epoch"] else 0
            order = epoch_order(len(dataset), t["seed"], epoch)[cursor:]
            if mode == "smoke":
                order = order[:smoke_batches * t["microbatch"]]
            loader = DataLoader(
                dataset, batch_size=t["microbatch"], sampler=order, num_workers=0,
                drop_last=False, generator=torch.Generator().manual_seed(t["seed"] + 10000 + epoch))
            iterator = iter(loader)
            training_mode(student, cfg)
            sums = {k: 0.0 for k in ("total", "intensity", "gradient", "anchor")}
            seen = 0
            while True:
                window = list(itertools.islice(iterator, t["accumulation"]))
                if not window:
                    break
                count = sum(len(b["image_id"]) for b in window)
                optimizer.zero_grad(set_to_none=True)
                synchronize(selected)
                compute_start = time.perf_counter()
                window_logs = []
                for batch in window:
                    if any(s != "train" for s in batch["split"]):
                        raise PermissionError("Non-Train sample reached optimizer")
                    vis, ir = batch["vis"].to(selected), batch["ir"].to(selected)
                    with torch.no_grad():
                        ref = fused_output(reference, vis, ir)
                    fused = fused_output(student, vis, ir)
                    parts = objective(fused, vis, ir, ref)
                    n = len(batch["image_id"])
                    (parts["total"] * (n / count)).backward()
                    for k in sums:
                        sums[k] += parts[k].detach().item() * n
                    window_logs.append({
                        "epoch": epoch + 1, "image_ids": "|".join(batch["image_id"]),
                        "samples": n, **{k: v.detach().item() for k, v in parts.items()},
                        "output_min": fused.detach().min().item(),
                        "output_max": fused.detach().max().item(),
                    })
                    seen += n
                    microbatches_processed += 1
                    del fused, ref, parts, vis, ir
                norm = gradient_norm(student)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in student.parameters() if p.requires_grad],
                    t["gradient_clip_norm"], error_if_nonfinite=True)
                clipped = gradient_norm(student)
                if clipped > t["gradient_clip_norm"] * 1.00001:
                    raise FloatingPointError("Gradient clipping bound was not respected")
                optimizer.step()
                scheduler.step()
                if any(p.grad is not None for p in reference.parameters()):
                    raise RuntimeError("Reference received a gradient")
                if state_digest(reference.state_dict()) != reference_digest:
                    raise RuntimeError("Reference weights or buffers changed")
                current = student.state_dict()
                for name, before in fixed_before.items():
                    if not torch.equal(current[name].detach().cpu(), before):
                        raise RuntimeError(f"Frozen tensor changed: {name}")
                if any(not torch.isfinite(p).all().item() for p in student.parameters()):
                    raise FloatingPointError("Nonfinite parameters after optimizer step")
                synchronize(selected)
                compute_seconds += time.perf_counter() - compute_start
                cursor += count
                progress = {"epoch": epoch, "sample_cursor": cursor,
                            "global_step": progress["global_step"] + 1}
                for row in window_logs:
                    row.update(global_step=progress["global_step"], grad_norm=norm,
                               clipped_grad_norm=clipped, lr=optimizer.param_groups[0]["lr"])
                batch_logs.extend(window_logs)
                memory_record(selected, cfg)
            if seen == 0:
                raise RuntimeError("No samples processed; resume cursor may be invalid/completed")
            val_records = dev_records[:2] if mode == "smoke" else dev_records
            validation = validate(student, reference, val_records, objective, selected)
            row = {
                "epoch": epoch + 1, "epoch_complete": cursor == len(dataset),
                "train_samples": seen, "dev_samples": len(val_records),
                **{f"train_{k}": sums[k] / seen for k in sums},
                **{f"dev_{k}": v for k, v in validation.items()},
                "global_step": progress["global_step"], "lr": optimizer.param_groups[0]["lr"],
            }
            history.append(row)
            memory = memory_record(selected, cfg)
            if cursor == len(dataset):
                progress.update(epoch=epoch + 1, sample_cursor=0)
            improved = validation["total"] < best_loss
            best_loss = min(best_loss, validation["total"])
            kwargs = dict(config_hash=cfg.fingerprint(), provenance=provenance,
                          progress=progress.copy(), best_loss=best_loss, mode=mode, history=history)
            save_training_checkpoint(run_dir / "last.pt", student, optimizer, scheduler, **kwargs)
            if improved:
                save_training_checkpoint(run_dir / "best.pt", student, optimizer, scheduler, **kwargs)
            write_history(run_dir / "history.csv", history)
            write_history(run_dir / "batch_losses.csv", batch_logs)
            print(f"{mode} epoch={epoch+1} samples={seen} steps={progress['global_step']} "
                  f"loss={row['train_total']:.8f} dev_loss={validation['total']:.8f}", flush=True)
            if mode == "smoke":
                stop = True
            if stop:
                break
        if sha256(cfg["model"]["pretrained"]) != cfg["model"]["pretrained_sha256"]:
            raise RuntimeError("Original pretrained checkpoint changed")
        if source_snapshot(cfg) != provenance["source_hashes"]:
            raise RuntimeError("Implementation/source changed during run")
        reloaded = make_student(cfg)
        strict_load(reloaded, run_dir / "last.pt")
        if state_digest(reloaded.state_dict()) != state_digest(student.state_dict()):
            raise RuntimeError("Saved/reloaded checkpoint differs")
        changed = state_digest(student.state_dict()) != init_digest
        if not changed:
            raise RuntimeError("No parameters changed after optimizer steps")
        result = {
            "status": "PASS", "mode": mode, "objective_status": cfg["status"],
            "dry_run_only": mode == "smoke", "full_training_run": mode == "train",
            "microbatches": microbatches_processed, "optimizer_steps_this_run": len(
                {r["global_step"] for r in batch_logs}),
            "train_samples_this_run": sum(r["samples"] for r in batch_logs),
            "validation_samples": history[-1]["dev_samples"],
            "validation_is_smoke_subset": mode == "smoke",
            "compute_seconds": compute_seconds, "wall_seconds": time.perf_counter() - started,
            "memory": memory, "strict_reload": "PASS", "pretrained_unchanged": True,
            "reference_unchanged": True, "frozen_tensors_unchanged": True,
            "parameters_changed": changed, "initial_state_sha256": init_digest,
            "last_state_sha256": state_digest(student.state_dict()),
            "last_sha256": sha256(run_dir / "last.pt"),
            "best_sha256": sha256(run_dir / "best.pt"),
            "canonical_checkpoint": "last.pt",
            "best_is_auxiliary": "minimum Dev reconstruction objective; NOT mIoU selection",
            "config_sha256": cfg.fingerprint(), "provenance": provenance,
            "fmb_test_used": False, "labels_used": False,
        }
        atomic_json(run_dir / "run_result.json", result)
        return result
    except Exception:
        atomic_json(run_dir / "failure.json", {"status": "FAIL", "traceback": traceback.format_exc()})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True, help="New, nonexisting artifact directory")
    parser.add_argument("--mode", choices=["smoke", "train"], default="smoke")
    parser.add_argument("--smoke-batches", type=int, default=2)
    parser.add_argument("--device", choices=["cpu", "cuda", "cuda:0"])
    parser.add_argument("--resume", help="Resume into a NEW run directory, same config/mode/source/input")
    args = parser.parse_args(argv)
    result = run(load_config(args.config), args.output, mode=args.mode,
                 smoke_batches=args.smoke_batches, device=args.device, resume=args.resume)
    print(f"PASS: {args.output}; last SHA256={result['last_sha256']}", flush=True)


if __name__ == "__main__":
    main()
