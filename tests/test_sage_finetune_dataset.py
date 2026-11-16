"""Synthetic unit tests; no real FMB labels or Test images are opened."""
import csv
from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from src.methods.sage_finetune.checkpoint import sha256
from src.methods.sage_finetune.dataset import (PairDataset, audit_inputs, load_protocol,
                                              read_manifest, read_pair, resolve_fmb_path)
from tests.sage_finetune_helpers import fixture


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = fixture(self.root)
        self.train, self.dev = load_protocol(self.cfg)

    def tearDown(self):
        self.tmp.cleanup()

    def test_nested_layout_and_leading_zero_id_and_no_labels(self):
        self.assertEqual(self.train[0].image_id, "00001")
        self.assertIn("train/train", self.train[0].vis.as_posix())
        self.assertFalse((Path(self.cfg["data"]["root"])/"train/Label").exists())
        rows = PairDataset(self.train, training=False)
        self.assertEqual(rows[0]["vis"].shape, (1,17,23))
        self.assertEqual(rows[0]["vis"].dtype, torch.float32)
        self.assertTrue(0 <= rows[0]["vis"].min() <= rows[0]["vis"].max() <= 1)

    def test_direct_pair_cannot_disguise_test_or_labels_as_train(self):
        r=self.train[0]
        for path in [self.root/"test/Visible/00001.png", self.root/"train/Label/00001.png"]:
            with self.assertRaises(PermissionError):
                read_pair(replace(r,vis=path))

    def test_flat_layout(self):
        other = self.root/"flat"
        cfg = fixture(other, nested=False)
        train, _ = load_protocol(cfg)
        self.assertNotIn("train/train", train[0].vis.as_posix())

    def test_paired_augmentation_and_epoch_replay(self):
        ds = PairDataset(self.train, training=True, crop_hw=[8,12], hflip_p=1, seed=2)
        ds.set_epoch(3)
        a, b = ds[0], ds[0]
        self.assertTrue(torch.equal(a["vis"], b["vis"]))
        ycc, ir, _ = read_pair(self.train[0])
        top, left = a["top"], a["left"]
        expected_v = ycc[top:top+8,left:left+12,0][:,::-1].copy()
        expected_i = ir[top:top+8,left:left+12][:,::-1].copy()
        self.assertTrue(torch.equal(a["vis"], torch.from_numpy(expected_v)[None].float()/255))
        self.assertTrue(torch.equal(a["ir"], torch.from_numpy(expected_i)[None].float()/255))

    def test_official_color_preprocessing(self):
        r = self.train[0]
        ycc, ir, _ = read_pair(r)
        self.assertTrue(np.array_equal(ycc, cv2.cvtColor(cv2.imread(str(r.vis)), cv2.COLOR_BGR2YCrCb)))
        self.assertTrue(np.array_equal(ir, cv2.cvtColor(cv2.imread(str(r.ir)), cv2.COLOR_BGR2GRAY)))

    def test_dev_cannot_train_and_test_fails_before_image_read(self):
        with self.assertRaises(PermissionError):
            PairDataset(self.dev, training=True, crop_hw=[8,12])
        with patch("cv2.imread", side_effect=AssertionError("Must not read Test")):
            with self.assertRaises(PermissionError):
                resolve_fmb_path(self.cfg["data"]["root"], "test/Visible/00001.png")
            with self.assertRaises(PermissionError):
                PairDataset([replace(self.train[0], split="test")], training=False)

    def test_pure_resolver_supports_nested_test_fixture_only(self):
        root = self.root/"fake_test_layout"
        path = root/"test/test/Visible/00001.png"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"fixture-name-only")
        self.assertEqual(resolve_fmb_path(root, "test/Visible/00001.png", allow_test=True), path.resolve())

    def test_hash_count_and_duplicate_guards(self):
        d = self.cfg["data"]
        with self.assertRaises(ValueError):
            read_manifest(d["train_manifest"], d["root"], "train", expected_count=3, expected_sha256="0"*64)
        with self.assertRaises(ValueError):
            read_manifest(d["train_manifest"], d["root"], "train", expected_count=4, expected_sha256=d["train_sha256"])
        p = Path(d["train_manifest"])
        with p.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        rows[1] = rows[0].copy()
        with p.open("w", encoding="utf-8", newline="") as stream:
            w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        with self.assertRaises(ValueError):
            read_manifest(p,d["root"],"train",expected_count=3,expected_sha256=sha256(p))

    def test_missing_path_shape_and_input_hash_fail(self):
        r = self.train[0]
        with self.assertRaises(ValueError):
            read_pair(replace(r, width=22))
        r.vis.write_bytes(b"changed")
        with self.assertRaises(ValueError):
            read_pair(r)
        with self.assertRaises(FileNotFoundError):
            resolve_fmb_path(self.cfg["data"]["root"], "train/Visible/99999.png")

    def test_16_bit_and_path_traversal_fail(self):
        r = self.train[0]
        cv2.imwrite(str(r.vis), np.zeros((17,23,3),np.uint16))
        expected = hashlib.md5(r.vis.read_bytes()).hexdigest()
        with self.assertRaises(ValueError):
            read_pair(replace(r, vis_md5=expected))
        with self.assertRaises(ValueError):
            resolve_fmb_path(self.cfg["data"]["root"], "../Visible/00001.png")

    def test_preflight_inventory_and_crop_guard(self):
        digest = audit_inputs(self.train+self.dev,self.root/"audit.csv")
        self.assertEqual(len(digest),64)
        with (self.root/"audit.csv").open(encoding="utf-8", newline="") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))),5)
        with self.assertRaises(ValueError):
            PairDataset(self.train,training=True,crop_hw=[100,100])[0]


if __name__ == "__main__":
    unittest.main()
