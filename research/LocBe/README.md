# LocBe — DCTR-C3-T final research handoff

**Repository target:** `https://github.com/quan14333/xulianh`, branch `LocBe`, directory `research/LocBe`.

**Research evidence GitHub milestone (verified 26/11/2026):** commit `bb53a374d3c7cf2f94918c715b50ebfdf1065a24` contains the FMB plus MFNet seed0 evidence. This handoff document is an additive reporting update; use Git HEAD to identify its eventual commit SHA.

**Important caveat:** FMB Test result is post-exposure / exploratory, not an untouched independent evaluation. Do not tune C3-T after seeing the Test scores. No demonstrated uplift over SAGE/TB.

## Verify locally (CPU; no Kaggle GPU)

```bash
python -m pip install numpy opencv-python-headless pandas
python source/self_test.py evidence
python source/evaluate.py --evidence-dir evidence --output results/evaluation_recomputed.json
```

Run from inside `research/LocBe/`. Frozen C2 and C3-T source are in `source/`. To produce new fusion PNG without GT, first generate matched SAGE RGB PNG from pinned group SAGE Student, then:

```bash
python source/inference.py --vis-dir path/to/Visible --ir-dir path/to/Infrared --sage-dir path/to/sage --output-dir path/to/c3t
```

The original reproducibility notebook is `notebooks/CS406_C3T_FMB_GROUP_TEST_ALL_IN_ONE_KAGGLE.ipynb` and the artifact ZIP is saved in `evidence/`. It used the pinned group scripts for downstream segmentation. **Do not rerun as if it were an independent held-out Test**. Reproducing only confusion scoring from existing NPZ requires no GPU.

## GitHub handoff status

Branch `LocBe` already exists; this storage update starts from verified HEAD `cb035dbac414d1335061e1e89b85a208c23a0189`. The research-evidence milestone `bb53a37` remains unchanged. Apply only documentation/checksum updates; do not rerun branch creation or experiments. This update does not merge `main` or create a PR. Review/PR decisions remain with the owner. See `FINAL_HANDOFF_27_11.md` and `CHECKLIST_27_11.md`.

## Contents

- `METHOD_SUMMARY.md`: method card, pipeline, equations, references
- `configs/`: frozen params and per-seed exact B3 run configs
- `source/`: C2 origin, C3-T guard, inference, replay validation, self-test and optional fusion metric script
- `results/`: full table mIoU per seed, class IoU and differences, 3-seed confusion, runtime evidence
- `figures/quant_*.png`: numerical plots (do not substitute for required qualitative cases)
- `evidence/`: original Kaggle report and immutable source artifacts
- `CHECKLIST_27_11.md`: completed/partial/missing evidence
- `SLIDE_OUTLINE.md`: 6-slide outline aligned with requirements
- `source/render_qualitative_panels.py`: create real qualitative panels when source images and predicted masks are retrieved

## New inference-only evidence — 26/11/2026

- `evidence/inference_only/report/`: source provenance, checkpoint verification, frozen PNG hash parity, 280 prediction SHA256 manifest, confusion parity, measured metrics and timings.
- `evidence/inference_only/predictions/c3t_seed0/`: all 280 predicted **class-ID** masks for B3 seed 0. These are post-exposure FMB Test replays, NOT independently selected.
- `figures/qualitative/`: six **real** fixed-ID panels (VIS, IR, SAGE fusion RGB, C3-T fusion RGB, GT, C3-T prediction and error). SAGE B3 prediction not present; no per-image win/loss against SAGE can be concluded from these panels.
- `results/fusion_metrics_summary.csv` and `results/fusion_metrics_per_image.csv`: EN/SD/SCD for 280 and 273 images, three fusion variants; grayscale [0,1] conventions documented in frozen source.
- `results/runtime_{b3_seed0,c3t_cpu_pure}.json`: measured isolated component latency/FPS; NOT end-to-end FPS including SAGE Student, data preprocessing, I/O and B3 together.
- `UPDATE_26_11.md`: results and remaining gaps.

**Frozen method, config and checkpoint selection were not modified.** Validation is still post-exposure/exploratory on FMB Test. B3 predictions for SAGE baseline, MS-SSIM convention, and an end-to-end timing remain unverified. The 3 B3 checkpoints were SHA256-matched in a separate recovery step and are outside GitHub (Kaggle Output/Drive).

## Frozen B3 checkpoints — Google Drive storage

Three FMB seed-specific checkpoints are stored in [LocBe_C3T_Checkpoints](https://drive.google.com/drive/folders/1rLJSzac-jkw1M5zmk5TyVllEJGZNmBaI?usp=sharing). See [`CHECKPOINT_STORAGE.md`](CHECKPOINT_STORAGE.md) for IDs, original SHA256 and verification scope. Folder metadata on 27/11 still reports **Anyone: Writer**, with a risk of unintended edits. Sharing is retained as chosen by the owner; it has not been changed or declared safe. Original-file SHA256 was verified; Drive copies were checked only for names and sizes, not redownloaded for SHA256. Access through Quân/Mân accounts remains unverified. FMB model identities/weights remain unchanged.

## MFNet cross-dataset seed0 — 26/11/2026

- Full detailed results: [`MFNET_RESULTS_26_11.md`](MFNET_RESULTS_26_11.md), evidence at `evidence/mfnet_seed0/`.
- MFNet 9-class B3 was trained separately for SAGE and C3-T, 100 epochs, seed0, same 784 original train images. Test mIoU9: SAGE 53.4829%, C3-T 54.0895% (+0.6066 pp); Validation mIoU9: SAGE 57.5254%, C3-T 57.2138% (−0.3116 pp). One seed **does not prove** robust improvement.
- Two MFNet seed0 checkpoints are **stored on Google Drive**; original-file SHA256 was verified. Drive metadata confirms only names and sizes; no redownload/hash of the uploaded copies has been performed. The binaries remain outside Git/result ZIP; FMB checkpoints are separate 15-class models.
- This is **MFNet, not SemanticRT**. Read [`FINAL_HANDOFF_27_11.md`](FINAL_HANDOFF_27_11.md) for evidence, missing items, and access warnings.

## MFNet seed0 checkpoint storage — 27/11/2026

Folder: https://drive.google.com/drive/u/0/folders/13CNqHSPiLUgLP8pxq6ykR2jnxr-T0I1Y

| Phương pháp / seed | Tên file trên Drive | Drive file ID | Bytes | SHA256 bản gốc đã kiểm tra | File |
|---|---|---|---:|---|---|
| SAGE / seed 0 | `sage_b3_seed0_mfnet_last.pt` | `1uJvT7Oh4shsfYQPjHmlmQ5XPpenDBcbA` | 189143705 | `b51087c73a2ad43a098708b827abfa68161f7280c0eb5a1656e66c147e50e356` | [Drive](https://drive.google.com/file/d/1uJvT7Oh4shsfYQPjHmlmQ5XPpenDBcbA/view) |
| C3-T / seed 0 | `c3t_b3_seed0_mfnet_last.pt` | `1HjKgMYMTgrHLjUecBzU5ZkAACtVnICQN` | 189143705 | `5554340fa5d87e144c07e24533dcf39378490399feca7da00f006083de3d1ad9` | [Drive](https://drive.google.com/file/d/1HjKgMYMTgrHLjUecBzU5ZkAACtVnICQN/view) |

**Verification scope:** the listed SHA256 values were checked on the **original files**, as recorded in the supplied handoff; only filename and size have been confirmed on Drive. The uploads have **not been redownloaded for SHA256**; remote SHA256 verification is **NOT PERFORMED**.

MFNet folder and both files currently report **Anyone: Writer**. Sharing is retained according to the owner's decision; unintended edits remain possible. Permissions were not changed or declared safe. Access from Quân/Mân accounts remains unverified.

These are MFNet **9-class** SegFormer-B3 checkpoints, separate from the three FMB **15-class** checkpoints. Storage metadata updates do not change C2/C3-T, the frozen threshold, configs, metrics or experiment results. See [`CHECKPOINT_STORAGE.md`](CHECKPOINT_STORAGE.md) for download-check commands and remaining checks.
