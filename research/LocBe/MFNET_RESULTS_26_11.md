# CS406 / LocBe — MFNet cross-dataset seed 0, 26/11/2026

## Scope and provenance

- **Dataset:** MFNet, RGB+thermal four-channel PNG input, 784 original Train + 392 Val + 393 Test; raw `train.txt` included 784 extra `_flip` copies, excluded from effective Train; original split manifests untouched.
- **Data leakage audit:** all `_flip` pairs complete, no base-ID overlap Train/Val/Test (by `report/MFNET_SPLIT_DIAGNOSTIC.json`). Dataset not FMB and not SemanticRT.
- **Fusion:** frozen SAGE Student and DCTR-C3-T (C3-T SHA256 `6eda7aabead6207d54107478edc52792538383f86913dc4dd838c8879504d4b6`; C2 SHA256 `7641e319a714fd44b246e3921404ece1110bc6766dd4d5bdc1c9c35c4165d621`). Frozen threshold 0.30, unchanged after seeing Test.
- **Downstream segmentation:** separately trained SegFormer-B3, 9 classes per input method, **100 epochs, seed 0 only**. This evaluates transfer of *fusion*, not zero-shot transfer of FMB's 15-class segmenter.

## Results (percent)

| Split | SAGE mIoU9 | C3-T mIoU9 | Delta (pp) | SAGE mIoU8 ignore GT0 | C3-T mIoU8 ignore GT0 | Delta (pp) |
|---|---:|---:|---:|---:|---:|---:|
| Validation (392) | 57.5254 | 57.2138 | -0.3116 | 57.7474 | 57.1090 | -0.6384 |
| Test (393) | 53.4829 | 54.0895 | +0.6066 | 53.2687 | 54.2545 | +0.9858 |

**Metric note:** mIoU9 includes `unlabeled` class 0. For mIoU8 ignore GT0, confusion is recomputed with all ground-truth class 0 pixels excluded, before calculating IoU of classes 1–8. These values are protocol-specific and not directly comparable to FMB mIoU13 or published external benchmarks without matching convention.

### Per-class IoU on Test (393 images; percentage)

| Class | SAGE | C3-T | C3-T − SAGE (pp) |
|---|---:|---:|---:|
| `unlabeled` | 98.0139 | 98.0404 | +0.0266 |
| `car` | 88.4373 | 88.6591 | +0.2218 |
| `person` | 70.3455 | 70.5129 | +0.1674 |
| `bike` | 62.4368 | 62.9703 | +0.5335 |
| `curb` | 37.0782 | 37.3195 | +0.2413 |
| `car_stop` | 27.0962 | 25.8138 | -1.2824 |
| `guardrail` | 0.3605 | 0.6767 | +0.3163 |
| `color_cone` | 51.5816 | 52.2162 | +0.6346 |
| `bump` | 45.9958 | 50.5965 | +4.6007 |

**Observed:** C3-T is +0.6066 pp on Test mIoU9 but −0.3116 pp on Val mIoU9. The single-seed evidence is mixed; do not claim robust improvement, statistical significance, or superiority over SAGE/TB. On Test, `guardrail` IoU remains below 1% for both methods; C3-T reduces `car_stop` IoU by ~1.28 pp.

## Independent file audit (26/11/2026)

- Original `CS406_LOCBE_MFNET_REPORT.zip` 58 entries, ZIP integrity PASS. `CS406_LOCBE_MFNET_FUSION_STAGE.zip` 19 entries, integrity PASS, its content is byte-identical to corresponding report entries.
- Recalculated all four confusion matrices (`SAGE/C3-T × Val/Test`) from NPZ, matching every per-class IoU, mIoU9 and mIoU8 in JSON; expected pixel counts 392×480×640 Val and 393×480×640 Test.
- `MFNET_FUSION_IMAGE_MANIFEST.csv` contains **4707 unique split/ID/method rows** (1569 images × 3 methods), all contain 64-character SHA256 values. **All 4707 fusion PNG binaries are NOT in the ZIP**, so per-image SHA256 cannot be independently rechecked here.
- Six qualitative fusion+GT panels are included, and 6 Test + 6 Val prediction masks are saved per segmentation method (samples, not full output). Panels do NOT include baseline segmentation predictions side by side; compare with masks inside `runs/`.
- Recorded checkpoint SHA256: SAGE `b51087c73a2ad43a098708b827abfa68161f7280c0eb5a1656e66c147e50e356`, C3-T `5554340fa5d87e144c07e24533dcf39378490399feca7da00f006083de3d1ad9`. **Model weights are not in this ZIP**; save `last.pt` for both MFNet seed-0 runs separately to Google Drive and verify SHA256 before claiming independently reproducible checkpoints.
- SAGE student checkpoint MD5 is provenance metadata, not independently verified from file bytes in this archive.

## Pending

1. Save MFNet **B3 `last.pt`** for SAGE seed0 and C3-T seed0 to Drive, checksum match with CSV.
2. Optional robustness: rerun both methods under identical settings with seeds 1 and 2; report mean±SD; full comparator TB B3 if group needs it.
3. Optionally save all 4707 fusion images or a deterministic reproducibility recipe with SAGE student checkpoint/input hashes. Do not infer image integrity only from manifest hashes.
4. SemanticRT remains separate and untested by this MFNet report.

Do not alter the frozen C3-T fusion parameters based on this Test result.
