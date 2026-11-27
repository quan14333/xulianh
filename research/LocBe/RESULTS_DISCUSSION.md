# Results discussion — FMB 273 clean / post-exposure exploratory


**C3-T mean mIoU13: 63.1994%** across 3 seeds. Compared with SAGE mean 63.3700%, TB mean 64.4233%. Group baseline numbers are locked and rounded in the original CSV.

**Top 3 class deltas vs SAGE (mean IoU, pp)**
- `bus`: +2.145 pp; C3-T 35.98 vs SAGE 33.84
- `car`: +0.073 pp; C3-T 79.05 vs SAGE 78.97
- `vegetation`: +0.056 pp; C3-T 87.45 vs SAGE 87.40

**Bottom 3 class deltas vs SAGE (mean IoU, pp)**
- `truck`: -2.627 pp; C3-T 30.80 vs SAGE 33.42
- `traffic_light`: -0.547 pp; C3-T 40.63 vs SAGE 41.17
- `motorcycle`: -0.422 pp; C3-T 37.70 vs SAGE 38.13

**Interpretation (hypotheses, not validated causal conclusions):** C3-T reverts C2 to SAGE where relative IR edges dominate; that can preserve IR structure but may suppress helpful VIS contrast. Sparse object classes have high variance from seed to seed (especially truck/bus), potentially changing mean scores. To attribute class changes to the guard, an exact paired *C2 vs C3-T* test with frozen checkpoints and qualitative examples is still required. A matched baseline+TTA study was not completed.

**Missing evidence:** The archive contains confusion per image, not predicted PNG masks or original VIS/IR/SAGE/C3-T RGB images. Therefore the 4–6 qualitative cases required by the group cannot be honestly produced from this package; `figures/quant_*` only display numerical results, and `QUALITATIVE_MISSING.md` specifies required inputs.
