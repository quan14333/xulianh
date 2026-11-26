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

Branch `LocBe` already exists and the evidence has been pushed through `bb53a37` (26/11/2026). Do **not** rerun the initial-branch creation instructions. Apply only the final-document patch after checking the current branch HEAD. Open a **Draft PR** to `main` for Quân to review; do not auto-merge. See `FINAL_HANDOFF_27_11.md` and `CHECKLIST_27_11.md`.

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

Three seed-specific checkpoints are stored in [LocBe_C3T_Checkpoints](https://drive.google.com/drive/folders/1rLJSzac-jkw1M5zmk5TyVllEJGZNmBaI?usp=sharing). See [`CHECKPOINT_STORAGE.md`](CHECKPOINT_STORAGE.md) for file IDs, exact SHA256 and verified scope. **Sharing must be changed from anyone-writer to restricted/group-viewer (or at least link-viewer) before the link is circulated in a public PR.** The local originals were SHA256-verified; the Drive uploads have been checked for names and size but not yet redownloaded for SHA256. No frozen code or model weights were altered.

## MFNet cross-dataset seed0 — 26/11/2026

- Full detailed results: [`MFNET_RESULTS_26_11.md`](MFNET_RESULTS_26_11.md), evidence at `evidence/mfnet_seed0/`.
- MFNet 9-class B3 was trained separately for SAGE and C3-T, 100 epochs, seed0, same 784 original train images. Test mIoU9: SAGE 53.4829%, C3-T 54.0895% (+0.6066 pp); Validation mIoU9: SAGE 57.5254%, C3-T 57.2138% (−0.3116 pp). One seed **does not prove** robust improvement.
- `last.pt` for two MFNet SegFormer runs **not in result ZIP or Drive as last verified**; FMB checkpoints on Drive are a different 15-class model.
- This is **MFNet, not SemanticRT**. Read [`FINAL_HANDOFF_27_11.md`](FINAL_HANDOFF_27_11.md) for evidence, missing items, and access warnings.
