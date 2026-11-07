"""Validate a SAGE FMB Student result archive without extracting its images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

from PIL import Image


EXPECTED_RUN = "sage_fmb_test_student_20261106T160350Z"
EXPECTED_COMMIT = "377fdbef6b8b6ccbd1139358ad66bb41ea084dc1"
EXPECTED_CHECKPOINT = "ecfde477ca4cd65401f207d5dc91ce6c475a49891d768dbdb7a66366255ee836"
REQUIRED = (
    "qa_summary.json", "run_config.json", "environment.json", "inference.csv",
    "manifest.csv", "qa_per_image.csv", "qa_failures.json",
)


def validate(archive: Path, expected_count: int = 280) -> dict:
    checks: dict[str, dict] = {}

    def check(name: str, ok: bool | None, detail: str) -> None:
        checks[name] = {"status": "NOT VERIFIED" if ok is None else "PASS" if ok else "FAIL",
                        "detail": detail}

    report = {"archive": str(archive), "expected_count": expected_count, "checks": checks}
    try:
        with zipfile.ZipFile(archive) as zf:
            infos = [i for i in zf.infolist() if not i.is_dir()]
            paths = [i.filename for i in infos]
            duplicates = [p for p, n in Counter(paths).items() if n > 1]
            check("archive_entries_unique", not duplicates, f"duplicate paths: {duplicates[:5]}")
            check("archive_integrity", zf.testzip() is None, "ZIP CRC check")

            def locate(base: str) -> str | None:
                matches = [p for p in paths if PurePosixPath(p).name == base
                           and "/fused/" not in "/" + p]
                if len(matches) != 1:
                    check(f"file_{base}", False, f"expected one entry, found {len(matches)}")
                    return None
                check(f"file_{base}", True, matches[0])
                return matches[0]

            found = {base: locate(base) for base in REQUIRED}

            def load_json(base: str):
                try:
                    return json.loads(zf.read(found[base])) if found[base] else None
                except (ValueError, UnicodeError, KeyError) as exc:
                    check(f"parse_{base}", False, str(exc))
                    return None

            def load_csv(base: str):
                try:
                    if not found[base]:
                        return None
                    return list(csv.DictReader(io.StringIO(zf.read(found[base]).decode("utf-8-sig"))))
                except (ValueError, UnicodeError, csv.Error, KeyError) as exc:
                    check(f"parse_{base}", False, str(exc))
                    return None

            qa, config, env, failures = (load_json(x) for x in
                                         ("qa_summary.json", "run_config.json", "environment.json", "qa_failures.json"))
            inference, manifest, per_image = (load_csv(x) for x in
                                              ("inference.csv", "manifest.csv", "qa_per_image.csv"))
            fused = [p for p in paths if PurePosixPath(p).parent.name == "fused"
                     and p.lower().endswith(".png")]
            check("fused_count", len(fused) == expected_count, f"{len(fused)}/{expected_count}")
            stems = [PurePosixPath(p).stem for p in fused]
            check("fused_names_unique", len(stems) == len(set(stems)),
                  f"{len(stems)} names, {len(set(stems))} unique")
            image_errors = []
            for path in fused:
                try:
                    with Image.open(io.BytesIO(zf.read(path))) as image:
                        image.load()
                        if image.format != "PNG" or image.mode != "RGB" or image.size != (800, 600):
                            image_errors.append(f"{path}: {image.format}, {image.mode}, {image.size}")
                except (OSError, ValueError, KeyError) as exc:
                    image_errors.append(f"{path}: {exc}")
            check("fused_decode_rgb_800x600", not image_errors,
                  f"decoded {len(fused)}; errors: {image_errors[:5]}")

            def table_check(name: str, rows: list[dict] | None) -> None:
                if rows is None:
                    return
                ids = [r.get("stem", "") for r in rows]
                check(f"{name}_count", len(rows) == expected_count, f"{len(rows)}/{expected_count}")
                check(f"{name}_ids", len(ids) == len(set(ids)) and set(ids) == set(stems),
                      f"{len(set(ids))} unique IDs; matching fused names: {set(ids) == set(stems)}")

            for name, rows in (("inference", inference), ("manifest", manifest),
                               ("qa_per_image", per_image)):
                table_check(name, rows)
            for name, rows in (("inference", inference), ("manifest", manifest)):
                if rows is not None:
                    check(f"{name}_split", all(r.get("split") == "test" for r in rows),
                          "all rows require test split")
            if inference is not None:
                check("inference_paths", all(PurePosixPath(r.get("fused_path", "")).stem == r.get("stem")
                                             for r in inference), "fused_path stem matches record ID")
            if manifest is not None:
                check("manifest_paths", all(PurePosixPath(r.get("ir_path", "")).stem == r.get("stem")
                                            and PurePosixPath(r.get("vis_path", "")).stem == r.get("stem")
                                            for r in manifest), "IR and visible paths match record ID")
            if per_image is not None:
                check("per_image_qa", all(r.get("status") == "PASS" and r.get("mode") == "RGB"
                                          and r.get("format") == "PNG" and r.get("width") == "800"
                                          and r.get("height") == "600" for r in per_image),
                      "all per-image QA rows PASS, RGB PNG 800x600")
            if isinstance(qa, dict):
                valid = (qa.get("split") == "test" and qa.get("input_pairs") == expected_count
                         and qa.get("selected_pairs") == expected_count
                         and qa.get("fused_pngs") == len(fused)
                         and qa.get("qa_pass") == expected_count and qa.get("qa_fail") == 0)
                check("qa_summary", valid, f"split={qa.get('split')}, pass={qa.get('qa_pass')}, fail={qa.get('qa_fail')}")
            elif qa is not None:
                check("qa_summary", False, "expected JSON object")
            if failures is not None:
                check("qa_failures", failures == [], f"failures: {str(failures)[:150]}")
            if isinstance(config, dict):
                valid = (config.get("run_id") == EXPECTED_RUN and config.get("split") == "test"
                         and config.get("expected_count") == expected_count
                         and config.get("selected_pairs") == expected_count
                         and config.get("repo_commit") == EXPECTED_COMMIT
                         and config.get("checkpoint_sha256") == EXPECTED_CHECKPOINT)
                check("run_config", valid, "run ID, split, counts, source commit and checkpoint hash")
            elif config is not None:
                check("run_config", False, "expected JSON object")
            if isinstance(env, dict):
                check("environment", bool(env.get("python") and env.get("torch") and env.get("gpu")),
                      "Python, PyTorch and GPU recorded")
            elif env is not None:
                check("environment", False, "expected JSON object")
            checkpoint = [p for p in paths if p.endswith("/checkpoints/checkpoints.pt")]
            if len(checkpoint) == 1:
                digest = hashlib.sha256(zf.read(checkpoint[0])).hexdigest()
                check("checkpoint_binary_sha256", digest == EXPECTED_CHECKPOINT, digest)
            else:
                check("checkpoint_binary_sha256", None, f"found {len(checkpoint)} checkpoint binaries")
            refs = [p for p in paths if p.endswith("/.git/refs/heads/main")]
            if len(refs) == 1:
                commit = zf.read(refs[0]).decode("ascii").strip()
                check("source_commit_ref", commit == EXPECTED_COMMIT, commit)
            else:
                check("source_commit_ref", None, f"found {len(refs)} main refs")
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        check("archive_open", False, str(exc))
    report["status"] = "FAIL" if any(c["status"] == "FAIL" for c in checks.values()) else "PASS"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--report", type=Path, help="write JSON report here")
    parser.add_argument("--expected-count", type=int, default=280, help="for small test fixtures")
    args = parser.parse_args()
    report = validate(args.input, args.expected_count)
    for name, result in report["checks"].items():
        print(f"{result['status']}: {name} — {result['detail']}")
    print(f"OVERALL: {report['status']}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
