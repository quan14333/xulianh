# LocBe — MFNet cross-dataset report

Dataset MFNet: 784 Train / 392 Val / 393 Test, 9 classes.
Source train.txt had 1568 entries: 784 originals + 784 precomputed _flip; V4 verified exact pairs, used originals only; did not change Val/Test.
Fusion C3-T frozen (FMB origin), SAGE Student frozen; SegFormer B3 9-class trained freshly per method.
Seeds: (0,); epochs: 100
FMB score and MFNet score are not directly comparable.
mIoU9_present includes unlabeled; mIoU8_ignore_GT0 excludes GT=0.
Results CSV: MFNET_B3_RESULTS.csv; fusion metrics are separate from segmentation.
This notebook never changes C3-T hyperparameters in response to Test results.
Partial 1-seed pilot must not be called 3-seed validation.