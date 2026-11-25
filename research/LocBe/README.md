# LocBe — DCTR-C3-T final research handoff

**Repository target:** `https://github.com/quan14333/xulianh`, branch `LocBe`, directory `research/LocBe`.

**Current status:** Branch-ready files; no GitHub push performed in this environment (connected user only has read permission).

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

## Upload

Unzip `METHOD_DCTR_C3T_FINAL.zip`, then follow `scripts/PUSH_LOCBE.ps1` or `scripts/PUSH_LOCBE.sh` to create the **real** branch `LocBe` based on the group repository `main`. Only the account with write permission can push; not automatically merged to main.

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
