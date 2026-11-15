"""Common-method/B3 integration tests using tiny synthetic pairs and actual FMBSeg."""
import csv
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from src.methods.b3_adapter import build_method, export_variant, method_statuses
from src.methods.b3_loader_adapter import inspect_inputs, resolve_dev_label
from src.methods.fusion_interface import FusionPair, FusionPrediction, WaitingForMethodSpec
from src.methods.quan_method import QuanMethodPending, QuanMethodSpec
from src.methods.sage_finetune.checkpoint import sha256
from src.methods.sage_finetune.config import REPO_ROOT
from src.methods.sage_finetune.model import load_pretrained
from src.methods.sage_finetune.train import run
from tests.sage_finetune_helpers import fixture


class B3AdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = fixture(self.root)
        torch.set_num_threads(2)

    def tearDown(self):
        self.tmp.cleanup()

    def pair(self):
        base = Path(self.cfg["data"]["root"])/"train/train"
        return FusionPair("00001", cv2.imread(str(base/"Visible/00001.png")),
                          cv2.imread(str(base/"Infrared/00001.png")))

    def pilot(self):
        run(self.cfg, self.root/"pilot", mode="smoke", device="cpu")
        return self.root/"pilot/last.pt"

    def labels_for_loader(self):
        path = Path(self.cfg["data"]["dev_manifest"])
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        folder = Path(self.cfg["data"]["root"])/"train/train/Label"
        folder.mkdir()
        for row in rows:
            sid = row["image_id"]
            label = np.arange(17*23,dtype=np.uint16).reshape(17,23)%15
            p = folder/(sid+".png")
            self.assertTrue(cv2.imwrite(str(p),label.astype(np.uint8)))
            row["label_path"] = f"train/Label/{sid}.png"
            row["label_md5"] = hashlib.md5(p.read_bytes()).hexdigest()
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        self.cfg.values["data"]["dev_sha256"] = sha256(path)

    def test_shared_contract_rejects_wrong_id_dtype_shape_and_test(self):
        pair = self.pair()
        pair.validate()
        for bad in [
            FusionPair("1",pair.vis_bgr,pair.ir_bgr),
            FusionPair("00001",pair.vis_bgr.astype(np.float32),pair.ir_bgr),
            FusionPair("00001",pair.vis_bgr,pair.ir_bgr[:4]),
            FusionPair("00001",None,pair.ir_bgr),
        ]:
            with self.assertRaises(ValueError):
                bad.validate()
        with self.assertRaises(PermissionError):
            FusionPair("00001",pair.vis_bgr,pair.ir_bgr,"test").validate()
        with self.assertRaises(ValueError):
            FusionPrediction(np.zeros((3,3,3),np.uint8),{}).validate(pair)

    def test_sage_pretrained_pixels_match_independent_b3_formula(self):
        pair = self.pair()
        before = sha256(self.cfg["model"]["pretrained"])
        method = build_method("sage",self.cfg,device="cpu",expected_sha256=before)
        actual = method.fuse(pair)
        net = load_pretrained(self.cfg).eval()
        ycc = cv2.cvtColor(pair.vis_bgr,cv2.COLOR_BGR2YCrCb)
        ir = cv2.cvtColor(pair.ir_bgr,cv2.COLOR_BGR2GRAY)
        t = lambda z:torch.from_numpy(z.copy()).float().div(255)[None,None]
        with torch.no_grad():
            y8=(net(t(ycc[:,:,0]),t(ir))[0].clamp(0,1)*255).round().byte().numpy()[0,0]
        expected=cv2.cvtColor(np.dstack([y8,ycc[:,:,1:]]),cv2.COLOR_YCrCb2RGB)
        np.testing.assert_array_equal(actual.rgb,expected)
        self.assertEqual(actual.metadata["checkpoint_role"],"FROZEN_PRETRAINED")
        self.assertEqual(sha256(self.cfg["model"]["pretrained"]),before)

    def test_checkpoint_hash_and_baseline_replacement_guards(self):
        with self.assertRaises(ValueError):
            build_method("sage",self.cfg,expected_sha256="0"*64)
        checkpoint=self.pilot()
        with self.assertRaises(ValueError):
            build_method("sage",self.cfg,checkpoint=checkpoint)
        with self.assertRaises(ValueError):
            build_method("sage_finetuned",self.cfg)  # Original weights cannot masquerade as trained.

    def test_pilot_requires_explicit_flag_and_is_never_full_baseline(self):
        checkpoint=self.pilot()
        with self.assertRaises(PermissionError):
            build_method("sage_finetuned",self.cfg,checkpoint=checkpoint)
        method=build_method("sage_finetuned",self.cfg,checkpoint=checkpoint,
                            expected_sha256=sha256(checkpoint),allow_pilot_checkpoint=True)
        self.assertEqual(method.identity["checkpoint_role"],"PILOT_ONLY")
        self.assertFalse(method.identity["full_evaluation_eligible"])
        method.fuse(self.pair()).validate(self.pair())

    def test_quan_waits_without_fallback_or_output(self):
        with patch("src.methods.b3_adapter.load_pretrained",side_effect=AssertionError("No fallback")):
            with self.assertRaises(WaitingForMethodSpec):
                build_method("quan_improved",self.cfg)
        destination=self.root/"NO_OUTPUT"
        with self.assertRaises(WaitingForMethodSpec):
            export_variant(self.cfg,"quan_improved",destination)
        self.assertFalse(destination.exists())
        self.assertEqual(method_statuses()["quan_improved"],"WAITING_FOR_METHOD_SPEC")

    def test_quan_spec_fields_do_not_create_an_algorithm(self):
        spec=QuanMethodSpec("","","PLACEHOLDER","","")
        self.assertEqual(len(spec.missing_fields()),5)
        with self.assertRaises(WaitingForMethodSpec):
            spec.validate()
        complete=QuanMethodSpec("hypothesis","formula","config","prototype","ablations")
        complete.validate()
        with self.assertRaises(WaitingForMethodSpec):
            QuanMethodPending(complete)

    def test_uniform_export_layout_and_provenance(self):
        result=export_variant(self.cfg,"sage",self.root/"export",limit=2,device="cpu")
        self.assertEqual(result["image_ids"],["00004","00005"])
        with (self.root/"export/output_manifest.csv").open(encoding="utf-8") as stream:
            rows=list(csv.DictReader(stream))
        for row in rows:
            self.assertEqual(row["path"],f"sage/train/{row['image_id']}.png")
            self.assertEqual(row["manifest_split"],"dev")
            self.assertEqual(sha256(self.root/"export"/row["path"]),row["png_sha256"])
        with self.assertRaises(FileExistsError):
            export_variant(self.cfg,"sage",self.root/"export",limit=2,device="cpu")

    def test_no_test_or_unbounded_provisional_export(self):
        for kwargs in [dict(split="test"),dict(limit=None),dict(limit=175)]:
            with self.assertRaises(PermissionError):
                export_variant(self.cfg,"sage",self.root/"forbidden",device="cpu",**kwargs)
        self.assertFalse((self.root/"forbidden").exists())

    def test_actual_man_loader_reads_15class_dev_without_fmb_test(self):
        self.labels_for_loader()
        export_variant(self.cfg,"sage",self.root/"export",limit=2,device="cpu")
        source=REPO_ROOT/"src/segmentation/train_seg.py"
        before=sha256(source)
        previous=sys.modules.get("chung")
        result=inspect_inputs(self.cfg,self.root/"export","sage",REPO_ROOT,self.root/"loader.json",
                              expected_checkpoint_sha256=self.cfg["model"]["pretrained_sha256"])
        self.assertEqual(result["count"],2)
        self.assertEqual(result["status"],"PASS")
        self.assertEqual(result["images"][0]["label_ids"],list(range(15)))
        self.assertFalse(result["fmb_test_used"])
        self.assertEqual(result["source_sha256"]["train_seg.py"],before)
        self.assertIs(sys.modules.get("chung"),previous)
        self.assertFalse((Path(self.cfg["data"]["root"])/"test").exists())

    def test_loader_rejects_test_paths_and_png_tampering(self):
        with self.assertRaises(PermissionError):
            resolve_dev_label(self.cfg["data"]["root"],"test/Label/00004.png",{"00004"})
        self.labels_for_loader()
        export_variant(self.cfg,"sage",self.root/"export",limit=2,device="cpu")
        p=self.root/"export/sage/train/00004.png"
        p.write_bytes(b"tampered")
        with self.assertRaises(ValueError):
            inspect_inputs(self.cfg,self.root/"export","sage",REPO_ROOT,self.root/"bad.json")

    def test_loader_rejects_wrong_source_pin_and_protected_output(self):
        self.labels_for_loader()
        export_variant(self.cfg,"sage",self.root/"export",limit=2,device="cpu")
        with self.assertRaises(ValueError):
            inspect_inputs(self.cfg,self.root/"export","sage",REPO_ROOT,self.root/"badpin.json",
                           {"chung.py":"0"*64,"train_seg.py":"0"*64})
        with self.assertRaises(ValueError):
            inspect_inputs(self.cfg,self.root/"export","sage",REPO_ROOT,
                           Path(self.cfg["data"]["root"])/"BAD.json")


if __name__=="__main__":
    unittest.main()
