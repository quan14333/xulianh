# Qualitative examples — pending, not fabricated

**Required group evidence still missing**: 4–6 aligned panels, each VIS | IR | GT | SAGE-pred | C3-T-pred | error overlay / C3-T fusion RGB. Include improvement, regression, small objects, night and boundary examples. Current Kaggle report used `--khong_luu_anh` and the archive contains per-image confusion only, so neither segmentation pixels nor original RGB images can be reconstructed from it.

To complete *without selecting a new method from Test*: recover already generated SAGE/C3-T fusion RGB and 3-seed `last.pt` from the original Kaggle Output. Save predictions deterministically from frozen weights for a clearly declared fixed set of IDs; record the prediction PNG hashes before looking at labels. Since FMB GT has already been viewed in research, mark every follow-up as **post-exposure exploratory**, not independent Test. Use panels for explanation, not cherry-picking claimed generalization. Do not regenerate or modify thresholds to improve Test.

The included `source/render_qualitative_panels.py` creates real panels from seven aligned PNG inputs when available; it refuses missing or invalid data.
