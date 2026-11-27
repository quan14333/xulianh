# CS406 LocBe C3-T — inference-only handoff

Replayed frozen C3-T B3 seed(s) on all 280 FMB Test images; no training.
Inference PNG SHA256 manifest is saved before GT scoring. `confusion_parity.json` compares original 280 confusion.
Qualitative panels: fixed six IDs evenly spaced over 273 clean images.
SAGE segmentation predictions NOT included; cannot claim per-image SAGE vs C3-T wins/losses.
Fusion EN/SD/SCD use explicit grayscale [0,1] definitions in source_frozen/fusion_metrics.py.
B3 GPU latency excludes disk read/write; C3-T CPU benchmark excludes all image IO and SAGE network.
This is POST-EXPOSURE EXPLORATORY FMB Test, not independent held-out confirmation.
Checkpoint weights (189 MB per seed) remain in prior Kaggle Output/Drive, not packaged here.
