# Quân method integration slot

Status: **WAITING_FOR_METHOD_SPEC**. Card docs/problem_cards/quan.md on main and
origin/feature/quan-baseline still contains placeholders at the15Nov audit.
The available Quân notebook reproduces pretrained SAGE; it is not a supplied new method.

Required handoff from Quân:

1. Hypothesis/failure mode and falsifiable expected change.
2. Algorithm or mathematical formula, preprocessing and output contract.
3. Versioned configuration and checkpoint/data dependencies.
4. Prototype entrypoint/source commit.
5. Ablation protocol,controls,seeds,selection/evaluation rules.

QuanMethodSpec describes these fields; it does not approve or implement an algorithm.
QuanMethodPending and factory variant quan_improved fail with WAITING_FOR_METHOD_SPEC.
No SAGE/identity fallback and no invented architecture is attributed to Quân.

After a real spec is reviewed, implement FusionMethod.fuse(FusionPair)->FusionPrediction
from [shared contract](../fusion_interface.py). Sources are aligned BGRuint8; result
is same-size RGBuint8. The common exporter writes B3 folders and provenance.
Keep original sage unchanged,write unit tests and Development smoke before performance runs.

Attribution: Quân owns the future method design; Lộc bé implements integration and tests.
Current interface/stub is Lộc bé's engineering work,not Quân's proposed contribution.
Full B3 runs and Test evaluation require their own locked protocol and explicit approval.
