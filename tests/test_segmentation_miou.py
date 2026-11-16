"""CPU-only tests cho cách tính mIoU của baseline B3 (src/segmentation/chung.py), dùng ma trận nhầm lẫn nhỏ."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.segmentation import chung

ROOT = Path(__file__).resolve().parents[1]


def test_iou_lop_dung_dinh_nghia() -> None:
    c = np.zeros((14, 14), np.int64)
    c[0, 0], c[0, 1], c[1, 1] = 6, 2, 4  # lớp 0: tp 6, gt 8, pred 6 -> 6/8; lớp 1: tp 4, gt 4, pred 6 -> 4/6
    iou = chung.iou_lop(c)
    assert np.isclose(iou[0], 75.0)
    assert np.isclose(iou[1], 100 * 4 / 6)
    assert np.isnan(iou[2])  # lớp không có trong GT


def test_cac_miou_15_lop_bo_pixel_unlabeled() -> None:
    c = np.zeros((15, 15), np.int64)
    c[0, 0] = 1000  # pixel unlabeled đoán đúng: không ảnh hưởng mIoU_co_mat
    c[1, 1] = 9  # road đúng 9
    c[1, 0] = 1  # road đoán thành unlabeled: vẫn tính là sai
    c[8, 8] = 5  # person đúng 5
    kq, iou, iou15 = chung.cac_miou(c)
    assert np.isclose(iou[0], 90.0)  # road 9 / (10 + 9 - 9)
    assert np.isclose(iou[7], 100.0)  # person
    assert kq["so_lop_co_mat"] == 2
    assert np.isclose(kq["mIoU_co_mat"], 95.0)
    assert np.isclose(kq["mIoU_14"], 190 / 14)
    assert iou15 is not None and np.isclose(kq["mIoU_15_nan0"], np.nan_to_num(iou15).mean())


def test_nham_bo_qua_ignore() -> None:
    gt = np.array([[0, 1], [chung.IGNORE, 1]])
    pred = np.array([[0, 0], [1, 1]])
    c = chung.nham(gt, pred, 2)
    assert c.tolist() == [[1, 0], [1, 1]]


def test_thong_tin_anh_test_khop_manifest() -> None:
    def doc(p: Path) -> list[dict[str, str]]:
        with p.open(encoding="utf-8") as f:
            return list(csv.DictReader(f))

    info = doc(ROOT / "src/segmentation/thong_tin_anh_test.csv")
    test = doc(ROOT / "data/manifests/test.csv")
    assert [r["image_id"] for r in info] == [r["image_id"] for r in test]
    assert [r["leak"] for r in info] == [r["leak"] for r in test]
    assert sum(r["leak"] == "True" for r in info) == 7
