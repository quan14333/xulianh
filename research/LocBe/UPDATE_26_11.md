# LocBe DCTR-C3-T — evidence addendum (26/11/2026)

**Scope:** Validation/reporting addendum, no change to frozen C3-T code, parameters, three-seed B3 results or checkpoint selection. FMB Test is **post-exposure exploratory**.

## Validation and independent file audit

- ZIP contents checked: 304 files, CRC OK.
- 280 prediction PNGs (seed 0) verified independently against 280 manifest SHA256 values, 0 mismatch. All masks are 600×800 with class IDs within 0..14; image IDs equal archived seed0 confusion image IDs.
- Kaggle report: `confusion_parity.json` **PASS**, zero mismatching images, 273-clean mIoU13 seed0 = 63.054613342650406%.
- Preflight reports 280/280 C3-T fusion PNG hashes match original frozen manifest, and all three last.pt checkpoint SHA256 verified.
- Six fixed real-data panels IDs: `00043`, `00364`, `00719`, `01053`, `01378`, `01500`. These show errors of C3-T only, **not** improvements vs SAGE segmentation baseline.

## Fusion metrics (273 clean FMB Test, per-image arithmetic means)

| Method | EN (bits) | SD grayscale [0,1] | SCD explicit |
|---|---:|---:|---:|
| SAGE | 6.835371 | 0.147749 | 1.712055 |
| TB | 6.189293 | 0.099877 | 1.385138 |
| C3-T | 6.837764 | 0.147877 | 1.713356 |

C3-T minus SAGE: EN +0.002394, SD +0.000128, SCD +0.001302. Improvements are **very small** and do not imply better segmentation or universally superior fusion. Metric definitions in `source/fusion_metrics.py`; MS-SSIM was not measured (no frozen dual-reference convention).

## Performance measurements (isolated, not total)

- C3-T algorithm on Kaggle **CPU**: 20 distinct images × 2 runs = 40 samples; mean 93.671 ms/image = 10.676 inverse-mean FPS. Excludes VIS/IR/SAGE file I/O and SAGE neural inference.
- SegFormer-B3 seed0 on Tesla T4 GPU: 280 predictions, mean forward plus CPU transfer 72.960 ms/image = 13.706 inverse-mean FPS (first five warmups excluded in latency). Preprocess mean 23.780 ms/image; PNG write mean 2.073 ms/image. Peak GPU allocated 1139.595 MiB; reserved 1224 MiB. NOT a complete end-to-end pipeline benchmark.

## Open items for 27/11 handoff

1. Retrieve SAGE B3 baseline mask predictions, using frozen SAGE seed0 checkpoint, to generate matched baseline comparison panels without tuning from Test.
2. Upload 3 verified B3 `last.pt` checkpoint binaries to Drive and record share links and SHA256.
3. Decide with Quân whether 280 PNG files + original raw NPZ/log/ZIP belong on GitHub (otherwise attach to Drive); ignored files must be force-added intentionally.
4. State missing MS-SSIM, full end-to-end latency, matched C2 ablation, TTA, MFNet as unverified / optional as appropriate.
5. Produce final ZIP and SHA256 after commit and Drive upload.
