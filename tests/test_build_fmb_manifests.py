"""CPU-only tests for the FMB manifest builder, using tiny synthetic data."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data import build_fmb_manifests as bm

LEAK = {f"{i:05d}" for i in range(1171, 1178)}
DEV_IDS = set(range(1011, 1161))


def audit_row(image_id: int, split: str) -> dict[str, str]:
    return {
        "name": f"{image_id:05d}",
        "split": f"{split}/{split}",
        "vis_w": "800",
        "vis_h": "600",
        "vis_md5": "",
    }


def fake_audit() -> list[dict[str, str]]:
    """1.220 train + 280 test, with test ids separating dev from train on both sides."""
    train_ids = [*range(1, 1001), *DEV_IDS, *range(1171 + 280, 1171 + 280 + 70)]
    test_ids = [*range(1001, 1011), *range(1161, 1171), *range(1171, 1431)]
    return [audit_row(i, "train") for i in sorted(train_ids)] + [
        audit_row(i, "test") for i in test_ids
    ]


def write_sample(root: Path, prefix: str, split: str, name: str, pixels: np.ndarray) -> str:
    """Write one VIS/IR/label triple and return the VIS md5."""
    base = root / prefix
    for folder in bm.THU_MUC.values():
        (base / folder).mkdir(parents=True, exist_ok=True)
    vis = base / "Visible" / f"{name}.png"
    Image.fromarray(pixels).resize(bm.SIZE, Image.NEAREST).save(vis)
    Image.fromarray(pixels).convert("L").resize(bm.SIZE).save(base / "Infrared" / f"{name}.bmp")
    label = np.full((bm.SIZE[1], bm.SIZE[0]), 3, dtype=np.uint8)
    Image.fromarray(label).save(base / "Label" / f"{name}.png")
    return hashlib.md5(vis.read_bytes()).hexdigest()


def fake_tap(root: Path, prefix_of: dict[str, str]) -> dict[str, list]:
    """One image per split, with dev being an exact copy of train."""
    rng = np.random.default_rng(0)
    shared = rng.integers(0, 255, (6, 8, 3), dtype=np.uint8)
    other = rng.integers(0, 255, (6, 8, 3), dtype=np.uint8)
    samples = [("train", "00001", shared), ("dev", "00002", shared), ("test", "00003", other)]
    tap: dict[str, list] = {}
    for split, name, pixels in samples:
        md5 = write_sample(root, prefix_of[split], split, name, pixels)
        tap[split] = [{"image_id": name, "vis_md5": md5, "ir_md5": "", "label_md5": ""}]
        for kind in bm.THU_MUC:
            tap[split][0][f"{kind}_path"] = ""
    return tap


def test_tach_khoi_groups_consecutive_ids() -> None:
    assert bm.tach_khoi([1, 2, 3, 7, 9, 10]) == [(1, 3), (7, 7), (9, 10)]


def test_chon_khoi_dev_skips_forbidden_blocks_and_is_deterministic() -> None:
    blocks = [(start, start + 49) for start in range(1, 401, 60)]
    brightness = {i: 50.0 + (i % 100) for a, b in blocks for i in range(a, b + 1)}
    forbidden = {blocks[0][0] + 5}

    chosen, _ = bm.chon_khoi_dev(brightness, forbidden)

    assert chosen == bm.chon_khoi_dev(brightness, forbidden)[0]
    assert blocks[0] not in chosen
    assert bm.DEV_MIN <= sum(b - a + 1 for a, b in chosen) <= bm.DEV_MAX


def test_chon_khoi_dev_fails_when_no_combination_fits() -> None:
    with pytest.raises(SystemExit):
        bm.chon_khoi_dev({i: 100.0 for i in range(1, 11)}, set())


def test_valid_split_passes_all_list_checks() -> None:
    tap = bm.tao_dong(fake_audit(), DEV_IDS, LEAK)

    assert bm.kiem_tra_danh_sach(tap, LEAK, DEV_IDS) == []
    assert [len(tap[k]) for k in ("train", "dev", "test")] == [1070, 150, 280]
    assert {d["image_id"] for d in tap["test"] if d["leak"]} == LEAK


def test_overlap_between_splits_is_reported() -> None:
    tap = bm.tao_dong(fake_audit(), DEV_IDS, LEAK)
    tap["dev"].append(dict(tap["train"][0]))

    assert "train ∩ dev rỗng" in bm.kiem_tra_danh_sach(tap, LEAK, DEV_IDS)


def test_duplicate_id_in_one_file_is_reported() -> None:
    tap = bm.tao_dong(fake_audit(), DEV_IDS, LEAK)
    tap["test"].append(dict(tap["test"][0]))

    assert "không trùng id trong test.csv" in bm.kiem_tra_danh_sach(tap, LEAK, DEV_IDS)


def test_missing_leak_flag_is_reported() -> None:
    tap = bm.tao_dong(fake_audit(), DEV_IDS, LEAK)
    next(d for d in tap["test"] if d["leak"])["leak"] = False

    assert "đúng 7 ảnh test gắn cờ leak" in bm.kiem_tra_danh_sach(tap, LEAK, DEV_IDS)


def test_dev_adjacent_to_train_is_reported() -> None:
    audit = [audit_row(i, "train") for i in range(1, 1221)]
    audit += [audit_row(i, "test") for i in range(1221, 1501)]
    dev_ids = set(range(1001, 1151))
    tap = bm.tao_dong(audit, dev_ids, LEAK)

    assert "dev không liền số với train" in bm.kiem_tra_danh_sach(tap, LEAK, dev_ids)


@pytest.mark.parametrize(
    ("prefix_of", "expected_path"),
    [
        ({"train": "train", "dev": "train", "test": "test"}, "train/Infrared/00002.bmp"),
        (
            {"train": "train/train", "dev": "train/train", "test": "test/test"},
            "train/Infrared/00002.bmp",
        ),
    ],
    ids=["flat", "colab-nested"],
)
def test_file_check_handles_both_layouts_and_fills_checksums(
    tmp_path: Path, prefix_of: dict[str, str], expected_path: str
) -> None:
    root, check_dir = tmp_path / "FMB", tmp_path / "audit"
    check_dir.mkdir()
    tap = fake_tap(root, prefix_of)

    assert bm.kiem_tra_file(tap, root, check_dir) == []

    assert tap["dev"][0]["ir_path"] == expected_path
    assert len(tap["dev"][0]["ir_md5"]) == 32
    assert len(tap["dev"][0]["label_md5"]) == 32
    with (check_dir / "kiem_tra_dev_gan_train.csv").open(encoding="utf-8") as f:
        near = list(csv.DictReader(f))
    assert near == [{"anh_dev": "00002", "anh_train": "00001", "hamming_vis": "0"}]
    assert (check_dir / "kiem_tra_dev_gan_train.png").is_file()
    with (check_dir / "kiem_tra_lop_theo_tap.csv").open(encoding="utf-8") as f:
        classes = {row["class_id"]: row for row in csv.DictReader(f)}
    assert classes["3"]["so_anh_dev"] == "1"
    assert classes["13"]["so_anh_train"] == "0"


def test_file_check_reports_faults(tmp_path: Path) -> None:
    root = tmp_path / "FMB"
    tap = fake_tap(root, {"train": "train", "dev": "train", "test": "test"})
    (root / "test" / "Infrared" / "00003.bmp").unlink()
    Image.new("L", (640, 512)).save(root / "train" / "Label" / "00001.png")
    tap["dev"][0]["vis_md5"] = "0" * 32

    errors = bm.kiem_tra_file(tap, root, tmp_path)

    assert "thiếu ir của test/00003" in errors
    assert "md5 VIS khác audit: 00002" in errors
    assert any(e.startswith("sai kích thước 00001.png") for e in errors)


def test_label_above_class_range_is_reported(tmp_path: Path) -> None:
    root = tmp_path / "FMB"
    tap = fake_tap(root, {"train": "train", "dev": "train", "test": "test"})
    Image.new("L", bm.SIZE, color=255).save(root / "test" / "Label" / "00003.png")

    assert "nhãn bất thường: 00003" in bm.kiem_tra_file(tap, root, tmp_path)
