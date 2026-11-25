# Checklist theo yêu cầu nhóm — Lộc Bé — deadline 27/11/2026

| Mục | Trạng thái | Bằng chứng / việc còn thiếu |
|---|---|---|
| Method card & references (2–3) | DONE | `METHOD_SUMMARY.md` |
| Frozen formula/config/source and baseline code pin | DONE | `source/`, `configs/`, `FROZEN_MANIFEST.json` |
| Split, input manifest/hash & reproducibility | PARTIAL | 1,500 image IDs+hash archived; actual VIS/IR PNG and original checkpoint binaries external |
| Deterministic CPU self test & confusion parity | DONE | `source/self_test.py`, `source/evaluate.py`, `results/evaluation.json` |
| Per-seed mean±SD, pp vs SAGE and TB | DONE | `results/results_overall.csv` |
| Per-class IoU and class delta | DONE | `results/results_per_class.csv`, `RESULTS_DISCUSSION.md` |
| Raw per-image confusion (280×15×15 per seed) | DONE | `results/confusion_per_image.npz`, `evidence/seed*_test_nham_last.npz` |
| 4–6 qualitative VIS/IR/GT/pred panels | MISSING | No C3-T/B3 predicted PNG in provided report; see `QUALITATIVE_MISSING.md` |
| Pure per-image latency / peak VRAM / FPS | PARTIAL | CPU export+IO 1,500-image elapsed and B3 hours only; pure latency and peak VRAM missing |
| Fusion metrics EN/SD/SCD/MS-SSIM | MISSING | Implemented explicit EN/SD/SCD tool, **not measured**; MS-SSIM reference convention not frozen |
| Baseline+TTA (if applicable) | MISSING | No matched TTA experiment |
| Paired C2→C3-T ablation | PARTIAL | Historical Dev exploratory evidence; no equal official Test comparison |
| Test data protocol provenance | DONE with caveat | FMB Test is **post-exposure / exploratory**, not independent |
| MFNet / independent dataset | MISSING / OPTIONAL | No evidence; requires MFNet-specific 9-class B3 recipe |
| SegFormer B3 checkpoints (3×last.pt) | MISSING BYTES | Only frozen SHA256 recorded; retrieve from Kaggle Output/Drive |
| Git branch `LocBe` remote push | BLOCKED | Connected GitHub user lacks push permission; use attached scripts with account that has write access |
| Final ZIP + SHA256 | DONE locally | `METHOD_DCTR_C3T_FINAL.zip` and `SHA256.txt` |

**Do not present missing rows as completed, do not silently retune after viewing Test.**
