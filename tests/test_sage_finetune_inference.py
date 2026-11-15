"""Independent B3 pixel contract, PNG checks, and a complete tiny CPU smoke."""
import csv
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import torch
from PIL import Image

from src.methods.sage_finetune.checkpoint import read_checkpoint, sha256
from src.methods.sage_finetune.infer import export, reconstruct_bgr, save_png
from src.methods.sage_finetune.model import fused_output
from src.methods.sage_finetune.train import run
from tests.sage_finetune_helpers import fixture


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.cfg=fixture(self.root)
        torch.set_num_threads(2)

    def tearDown(self):
        self.tmp.cleanup()

    def test_pixel_contract_chroma_clamp_and_tie_rounding(self):
        ycc=np.array([[[4,255,128],[7,128,128],[9,90,180],[6,40,220]]],dtype=np.uint8)
        raw=torch.tensor([[[[-.1,.5/255,1.,.45]]]],dtype=torch.float32)
        expected_y=np.array([[0,0,255,115]],dtype=np.uint8)
        expected=cv2.cvtColor(np.dstack((expected_y,ycc[:,:,1:])),cv2.COLOR_YCrCb2BGR)
        actual=reconstruct_bgr(raw,ycc)
        np.testing.assert_array_equal(actual,expected)
        self.assertEqual(actual[0,0,2],178)  # B3 path, not official float-RGB result153.

    def test_png_reopens_rgb_and_bgr_without_channel_swap(self):
        bgr=np.array([[[0,0,255],[255,0,0],[0,255,0]]],np.uint8)
        path=self.root/"image.png"
        digest=save_png(path,bgr)
        self.assertEqual(digest,sha256(path))
        np.testing.assert_array_equal(cv2.imread(str(path)),bgr)
        with Image.open(path) as image:
            self.assertEqual(image.mode,"RGB")
            np.testing.assert_array_equal(np.array(image),bgr[:,:,::-1])
        with self.assertRaises(FileExistsError):
            save_png(path,bgr)

    def test_bad_output_and_shape_fail(self):
        ycc=np.zeros((3,4,3),np.uint8)
        for raw,exception in [(torch.full((1,1,3,4),float("inf")),FloatingPointError),
                              (torch.full((1,1,3,4),1.5),ValueError),
                              (torch.zeros(1,1,2,4),ValueError)]:
            with self.assertRaises(exception):
                reconstruct_bgr(raw,ycc)
        class Broken(torch.nn.Module):
            def forward(self,v,i):
                return v*float("nan"),()
        with self.assertRaises(FloatingPointError):
            fused_output(Broken(),torch.zeros(1,1,4,4),torch.zeros(1,1,4,4))

    def test_export_test_and_unbounded_provisional_fail(self):
        with self.assertRaises(PermissionError):
            export(self.cfg,self.root/"no_test",split="test")
        with self.assertRaises(PermissionError):
            export(self.cfg,self.root/"no_all",limit=None)
        self.assertFalse((self.root/"no_test").exists())
        self.assertFalse((self.root/"no_all").exists())

    def test_cpu_two_batches_checkpoint_and_export_pipeline(self):
        result=run(self.cfg,self.root/"smoke",mode="smoke",smoke_batches=2,device="cpu")
        self.assertEqual(result["microbatches"],2)
        self.assertEqual(result["optimizer_steps_this_run"],1)
        self.assertEqual(result["train_samples_this_run"],3)  # Uneven2+1, actual-window normalization.
        self.assertTrue(result["parameters_changed"] and result["reference_unchanged"])
        self.assertTrue(result["frozen_tensors_unchanged"])
        state,packet=read_checkpoint(self.root/"smoke/last.pt")
        self.assertTrue(packet["dry_run_only"])
        self.assertEqual(packet["progress"]["global_step"],1)
        self.assertTrue((self.root/"smoke/best.pt").is_file())
        exported=export(self.cfg,self.root/"export",checkpoint=self.root/"smoke/last.pt",
                        split="dev",limit=2,device="cpu")
        self.assertEqual(exported["count"],2)
        self.assertTrue(exported["checkpoint_dry_run_only"])
        with (self.root/"export/output_manifest.csv").open(encoding="utf-8") as stream:
            rows=list(csv.DictReader(stream))
        self.assertEqual([r["image_id"] for r in rows],["00004","00005"])
        for row in rows:
            p=self.root/"export"/row["path"]
            self.assertEqual(sha256(p),row["png_sha256"])
            self.assertEqual(row["source_split"],"train")
            self.assertEqual(cv2.imread(str(p)).shape,(17,23,3))


if __name__=="__main__":
    unittest.main()
