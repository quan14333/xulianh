# METHOD SUMMARY — Lộc Bé / DCTR-C3-T

**Status:** Code and evaluated configuration frozen; FMB Test **post-exposure / exploratory**, not untouched held-out confirmation. Deadline: 27/11/2026.

**Task/output:** RGB fusion image from aligned VIS and IR for image fusion, then downstream FMB semantic segmentation through the group's SegFormer-B3. C3-T generates an RGB fusion image, **NOT** a segmentation mask and does not prove improvement in fusion visual quality from mIoU alone.

## Hypothesis and distinction from SAGE

SAGE supplies the baseline fused image. C2 tries to recover high-frequency VIS details missing from SAGE, with structure-tensor coherence/reliability masks. C3-T additionally protects regions where normalized IR gradient exceeds VIS gradient by >=0.30 and reverts them *pixelwise* to SAGE. Novelty is the combined deterministic post-processing rule as a **project hypothesis**, not a claim of research-first novelty; no extra trainable fusion weights.

## Exact pipeline

`VIS RGB + IR RGB -> SAGE Student frozen RGB -> C2 residual on SAGE Y -> normalized thermal map from VIS/IR -> C3-T revert to SAGE in thermal>=0.30 -> fused RGB -> SegFormer-B3 (separate training) -> mIoU`.

- C2: `m_def=clip((|grad Yvis|-|grad Ysage|)/(|grad Yvis|+0.02),0,1)`.
- `m_rel = structure_coherence(Yvis; sigma=1.0) * |grad Yvis|/(|grad Yvis|+0.06)`.
- `Yc2 = clip(Ysage + 0.2 * m_def * m_rel * (Yvis-Gaussian(Yvis,sigma=1.2)),0,1)`; preserve SAGE chroma and byte identity where unchanged.
- Thermal: Gaussian blur VIS/IR grayscale sigma=1; Sobel magnitudes each normalized by within-image 90th percentile; `thermal=max(normGradIR-normGradVIS,0)`.
- `C3T(x)=SAGE(x)` if `thermal(x)>=0.30`, else `C2(x)`.

The `lambda_thermal=0.75` and `lambda_conflict=0.50` values were defined in the original notebook's parameter dict but **unused by the C3-T output decision**. They are retained as audit provenance, not presented as active hyperparameters.

## Frozen B3 downstream protocol

FMB 1045 train + 175 dev combined as 1220 train; 280 Test / 273 clean Test excluding 7 flagged leak IDs; 15-label SegFormer-B3; augmentation scale 0.5–2, random crop 512, flip; batch 8, LR=6e-5, AdamW, 100 epochs, 15,200 steps, FP16; seeds 0/1/2; evaluate last.pt (no Test-based selection). GT class 0 pixels ignored in scoring before pooled confusion mIoU; metric reported as 13 present semantic classes.

## Evidence and conclusion (273 clean Test)

- C3-T: **63.1994 ± 0.4631 pp** (mean ± seed sample SD, NOT CI).
- SAGE: **63.3700%**, delta C3T−SAGE **-0.1706 pp**.
- TB: **64.4233%**, delta C3T−TB **-1.2239 pp**.

Seed 0/1 C3-T beat SAGE, seed 2 decreased ~1.59pp vs SAGE; TB beat C3-T across all 3 seeds. Hence **no robust accuracy gain demonstrated**. Per-class details are in `results/results_per_class.csv`.

## Strengths/weaknesses

Strengths: fully deterministic postprocessing, no segmentation GT during fusion, single fixed guard, separate reproducible downstream segmentation recipe. Weaknesses: threshold/guard may suppress useful VIS edges, model-specific performance instability, Test already exposed, no test image panels or checkpoint binary in current archive, no fusion EN/SD/SCD/MS-SSIM evidence, no matched TTA, no cross-dataset result.

## Related work

1. Wu et al., *Every SAM Drop Counts: Embracing Semantic Priors for Multi-Modality Image Fusion and Beyond*, CVPR 2025: https://openaccess.thecvf.com/content/CVPR2025/html/Wu_Every_SAM_Drop_Counts_Embracing_Semantic_Priors_for_Multi-Modality_Image_CVPR_2025_paper.html
2. Liu et al., *Multi-interactive Feature Learning and a Full-time Multi-modality Benchmark for Image Fusion and Segmentation*, ICCV 2023: https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_Multi-interactive_Feature_Learning_and_a_Full-time_Multi-modality_Benchmark_for_Image_ICCV_2023_paper.pdf
3. Xie et al., *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers*, NeurIPS 2021: https://proceedings.neurips.cc/paper/2021/hash/64f1f27bf1b4ec22924fd0acb550c235-Abstract.html
