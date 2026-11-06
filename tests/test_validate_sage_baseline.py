"""CPU-only tests of the archive validator with tiny synthetic ZIPs."""

import csv
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def fixture_zip(path: Path, *, bad_image=False, missing_qa=False, duplicate_id=False):
    ids = ["00001", "00001" if duplicate_id else "00002"]
    qa = {"split": "test", "input_pairs": 2, "selected_pairs": 2,
          "fused_pngs": 2, "qa_pass": 2, "qa_fail": 0}
    config = {"run_id": "sage_fmb_test_student_20261106T160350Z", "split": "test",
              "expected_count": 2, "selected_pairs": 2,
              "repo_commit": "377fdbef6b8b6ccbd1139358ad66bb41ea084dc1",
              "checkpoint_sha256": "ecfde477ca4cd65401f207d5dc91ce6c475a49891d768dbdb7a66366255ee836"}

    def csv_bytes(fields, rows):
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        return out.getvalue()

    image = io.BytesIO()
    Image.new("RGB", (800, 600), (1, 2, 3)).save(image, format="PNG")
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in (("qa_summary.json", qa), ("run_config.json", config),
                           ("environment.json", {"python": "3.10", "torch": "2", "gpu": "T4"}),
                           ("qa_failures.json", [])):
            if name != "qa_summary.json" or not missing_qa:
                zf.writestr("run/" + name, json.dumps(data))
        zf.writestr("run/inference.csv", csv_bytes(
            ["stem", "split", "fused_path"],
            [{"stem": stem, "split": "test", "fused_path": f"run/fused/{stem}.png"} for stem in ids]))
        zf.writestr("run/manifest.csv", csv_bytes(
            ["stem", "split", "ir_path", "vis_path"],
            [{"stem": stem, "split": "test", "ir_path": f"IR/{stem}.png",
              "vis_path": f"Vis/{stem}.png"} for stem in ids]))
        zf.writestr("run/qa_per_image.csv", csv_bytes(
            ["stem", "status", "mode", "format", "width", "height"],
            [{"stem": stem, "status": "PASS", "mode": "RGB", "format": "PNG",
              "width": 800, "height": 600} for stem in ids]))
        for stem in ("00001", "00002"):
            zf.writestr(f"run/fused/{stem}.png", b"broken" if bad_image and stem == "00002"
                        else image.getvalue())


def run_validator(path: Path, report: Path):
    result = subprocess.run(
        [sys.executable, "scripts/validate_sage_baseline.py", "--input", str(path),
         "--report", str(report), "--expected-count", "2"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    return result, json.loads(report.read_text(encoding="utf-8"))


def test_valid_archive(tmp_path):
    archive, report = tmp_path / "valid.zip", tmp_path / "report.json"
    fixture_zip(archive)
    result, data = run_validator(archive, report)
    assert result.returncode == 0, result.stdout + result.stderr
    assert data["status"] == "PASS"
    assert data["checks"]["checkpoint_binary_sha256"]["status"] == "NOT VERIFIED"


def test_decode_failure(tmp_path):
    archive, report = tmp_path / "broken.png.zip", tmp_path / "report.json"
    fixture_zip(archive, bad_image=True)
    result, data = run_validator(archive, report)
    assert result.returncode != 0
    assert data["checks"]["fused_decode_rgb_800x600"]["status"] == "FAIL"


def test_missing_metadata(tmp_path):
    archive, report = tmp_path / "missing.zip", tmp_path / "report.json"
    fixture_zip(archive, missing_qa=True)
    result, data = run_validator(archive, report)
    assert result.returncode != 0
    assert data["checks"]["file_qa_summary.json"]["status"] == "FAIL"


def test_duplicate_record_id(tmp_path):
    archive, report = tmp_path / "duplicate.zip", tmp_path / "report.json"
    fixture_zip(archive, duplicate_id=True)
    result, data = run_validator(archive, report)
    assert result.returncode != 0
    assert data["checks"]["inference_ids"]["status"] == "FAIL"


def test_corrupt_zip(tmp_path):
    archive, report = tmp_path / "corrupt.zip", tmp_path / "report.json"
    archive.write_bytes(b"not a zip")
    result, data = run_validator(archive, report)
    assert result.returncode != 0
    assert data["checks"]["archive_open"]["status"] == "FAIL"
