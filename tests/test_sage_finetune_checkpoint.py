"""Strict loading, freezing, safety guards, and optimizer/scheduler/RNG replay."""
import copy
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from src.methods.sage_finetune.checkpoint import (
    capture_rng, cpu_state, restore_rng, resume_training, save_training_checkpoint,
    sha256, state_digest, strict_load,
)
from src.methods.sage_finetune.config import (
    REPO_ROOT, authorize, load_config, new_output_directory, parse_config,
)
from src.methods.sage_finetune.model import (
    fused_output, initial_pair, load_pretrained, make_student, training_mode,
)
from src.methods.sage_finetune.train import configure_runtime, epoch_order, gradient_norm, run
from tests.sage_finetune_helpers import fixture


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.cfg=fixture(self.root)
        configure_runtime(self.cfg,"cpu")

    def tearDown(self):
        self.tmp.cleanup()

    def test_pretrained_pair_each_tensor_and_frozen_reference(self):
        student,reference=initial_pair(self.cfg)
        self.assertEqual(state_digest(student.state_dict()),state_digest(reference.state_dict()))
        self.assertTrue(all(not p.requires_grad for p in reference.parameters()))
        self.assertEqual(len(student.state_dict()),len(reference.state_dict()))
        student.eval()
        x=torch.rand(1,1,9,13)
        with torch.no_grad():
            self.assertTrue(torch.equal(fused_output(student,x,x),fused_output(reference,x,x)))

    def test_strict_rejects_missing_extra_shape_and_nan(self):
        original=cpu_state(load_pretrained(self.cfg))
        variants=[]
        missing=copy.deepcopy(original);missing.pop(next(iter(missing)));variants.append(missing)
        extra=copy.deepcopy(original);extra["unexpected"]=torch.zeros(1);variants.append(extra)
        wrong=copy.deepcopy(original);wrong["conv.weight"]=torch.zeros(9,9);variants.append(wrong)
        invalid=copy.deepcopy(original);invalid["conv.weight"][0,0,0,0]=float("nan");variants.append(invalid)
        for index,state in enumerate(variants):
            path=self.root/f"bad_{index}.pt";torch.save(state,path)
            with self.assertRaises((RuntimeError,ValueError)):
                strict_load(make_student(self.cfg),path)

    def test_hash_guard_prevents_random_or_wrong_initialization(self):
        self.cfg.values["model"]["pretrained_sha256"]="0"*64
        with self.assertRaises(ValueError):
            load_pretrained(self.cfg)
        self.cfg.values["model"]["pretrained_sha256"]=sha256(self.cfg["model"]["pretrained"])
        self.cfg.values["model"]["source_sha256"]="0"*64
        with self.assertRaises(ValueError):
            load_pretrained(self.cfg)

    def test_freeze_patterns_bn_buffers_and_gradients(self):
        self.cfg.values["model"]["freeze_patterns"]=["conv.*"]
        student,reference=initial_pair(self.cfg)
        training_mode(student,self.cfg)
        self.assertFalse(student.bn.training)
        self.assertTrue(student.bn.weight.requires_grad)
        before=cpu_state(student)
        x=torch.rand(2,1,8,12)
        optimizer=torch.optim.Adam([p for p in student.parameters() if p.requires_grad],lr=.001)
        fused_output(student,x,x).square().mean().backward();optimizer.step()
        for name in ["conv.weight","conv.bias","bn.running_mean","bn.running_var","bn.num_batches_tracked"]:
            self.assertTrue(torch.equal(student.state_dict()[name],before[name]),name)
        self.assertIsNone(reference.bn.weight.grad)
        self.cfg.values["model"]["freeze_patterns"]=["typo.*"]
        with self.assertRaises(ValueError):
            initial_pair(self.cfg)

    def test_rng_restore_python_numpy_torch(self):
        state=capture_rng()
        a=(random.random(),np.random.rand(),torch.rand(3))
        restore_rng(state)
        b=(random.random(),np.random.rand(),torch.rand(3))
        self.assertEqual(a[:2],b[:2])
        self.assertTrue(torch.equal(a[2],b[2]))

    def test_resume_matches_uninterrupted_next_update(self):
        torch.manual_seed(9)
        model=torch.nn.Linear(3,1)
        opt=torch.optim.AdamW(model.parameters(),lr=.01)
        sch=torch.optim.lr_scheduler.StepLR(opt,1,.8)
        def step(m,o,s):
            x=torch.rand(2,3)+np.random.rand()+random.random()
            o.zero_grad();m(x).square().mean().backward();o.step();s.step()
        step(model,opt,sch)
        path=self.root/"resume.pt"
        save_training_checkpoint(path,model,opt,sch,config_hash="fixture",
                                 provenance={"source":"synthetic"},progress={"epoch":0},
                                 best_loss=1.0,mode="smoke",history=[])
        step(model,opt,sch)
        expected=cpu_state(model)
        new=torch.nn.Linear(3,1)
        opt2=torch.optim.AdamW(new.parameters(),lr=.01)
        sch2=torch.optim.lr_scheduler.StepLR(opt2,1,.8)
        resume_training(path,new,opt2,sch2,config_hash="fixture",
                        provenance={"source":"synthetic"},mode="smoke")
        step(new,opt2,sch2)
        self.assertEqual(state_digest(expected),state_digest(new.state_dict()))
        self.assertEqual(sch.state_dict(),sch2.state_dict())
        for p,q in zip(opt.state.values(),opt2.state.values()):
            for key in p:
                if isinstance(p[key],torch.Tensor):
                    self.assertTrue(torch.equal(p[key],q[key]))
        with self.assertRaises(ValueError):
            resume_training(path,new,opt2,sch2,config_hash="wrong",
                            provenance={"source":"synthetic"},mode="smoke")
        with self.assertRaises(PermissionError):
            resume_training(path,new,opt2,sch2,config_hash="fixture",
                            provenance={"source":"synthetic"},mode="train")

    def test_provisional_full_training_and_large_smoke_blocked_before_output(self):
        target=self.root/"MUST_NOT_EXIST"
        with self.assertRaises(PermissionError):
            run(self.cfg,target,mode="train")
        with self.assertRaises(PermissionError):
            authorize(self.cfg,"smoke",3)
        self.assertFalse(target.exists())

    def test_output_protection_and_existing_directory(self):
        with self.assertRaises(ValueError):
            new_output_directory(Path(self.cfg["data"]["root"])/"new",self.cfg)
        with self.assertRaises(ValueError):
            new_output_directory(Path(self.cfg["model"]["source_root"])/"new",self.cfg)
        with self.assertRaises(FileExistsError):
            new_output_directory(self.root,self.cfg)

    def test_pipeline_partial_resume_matches_manual_two_updates(self):
        from src.methods.sage_finetune.dataset import PairDataset,load_protocol
        from src.methods.sage_finetune.losses import FusionLoss
        from src.methods.sage_finetune.train import build_optimizer,build_scheduler
        cfg=fixture(self.root/"resume_fixture",n_train=9,n_dev=2)
        run(cfg,self.root/"prefix",mode="smoke",device="cpu")
        result=run(cfg,self.root/"resumed",mode="smoke",device="cpu",
                   resume=self.root/"prefix/last.pt")
        self.assertEqual(result["optimizer_steps_this_run"],1)
        from src.methods.sage_finetune.checkpoint import read_checkpoint
        resumed,packet=read_checkpoint(self.root/"resumed/last.pt")
        self.assertEqual(packet["progress"]["global_step"],2)
        self.assertEqual(packet["progress"]["sample_cursor"],8)
        configure_runtime(cfg,"cpu")
        student,ref=initial_pair(cfg);training_mode(student,cfg)
        train,_=load_protocol(cfg)
        ds=PairDataset(train,training=True,crop_hw=cfg["training"]["crop_hw"],
                       hflip_p=cfg["training"]["hflip_p"],seed=cfg["training"]["seed"])
        opt=build_optimizer(student,cfg);sch=build_scheduler(opt,cfg,len(ds));loss=FusionLoss(**cfg["loss"])
        order=epoch_order(len(ds),cfg["training"]["seed"],0)[:8]
        for offset in (0,4):
            opt.zero_grad(set_to_none=True)
            for j in (offset,offset+2):
                batch=[ds[k] for k in order[j:j+2]]
                v=torch.stack([x["vis"] for x in batch]);i=torch.stack([x["ir"] for x in batch])
                with torch.no_grad():r=fused_output(ref,v,i)
                (loss(fused_output(student,v,i),v,i,r)["total"]*.5).backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(),cfg["training"]["gradient_clip_norm"])
            opt.step();sch.step()
        self.assertEqual(state_digest(resumed),state_digest(student.state_dict()))

    def test_missing_zero_and_nonfinite_gradient_guard(self):
        model=torch.nn.Linear(2,1)
        with self.assertRaises(RuntimeError):
            gradient_norm(model)
        for p in model.parameters():
            p.grad=torch.zeros_like(p)
        with self.assertRaises(FloatingPointError):
            gradient_norm(model)
        next(model.parameters()).grad.fill_(float("nan"))
        with self.assertRaises(FloatingPointError):
            gradient_norm(model)

    def test_large_provisional_microbatch_is_blocked(self):
        self.cfg.values["training"]["microbatch"]=1045
        with self.assertRaises(PermissionError):
            authorize(self.cfg,"smoke",2)

    def test_production_config_counts_and_formula(self):
        cfg=load_config(REPO_ROOT/"configs/methods/sage_student_finetune.yaml")
        self.assertEqual(cfg["status"],"PROVISIONAL")
        self.assertEqual(cfg["loss"],dict(intensity=1,gradient=.2,anchor=1))
        self.assertEqual((cfg["data"]["train_count"],cfg["data"]["dev_count"]),(1045,175))
        bad=cfg.snapshot();bad["data"]["train_count"]=4
        with self.assertRaises(ValueError):
            parse_config(bad)
        bad=cfg.snapshot();bad["loss"]["anchor"]=float("nan")
        with self.assertRaises(ValueError):
            parse_config(bad)
        self.assertEqual(epoch_order(7,2,0),epoch_order(7,2,0))
        self.assertNotEqual(epoch_order(7,2,0),epoch_order(7,2,1))


if __name__=="__main__":
    unittest.main()
