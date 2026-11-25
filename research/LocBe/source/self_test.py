"""CPU self-test for frozen C2 + C3-T and archived test confusion validation."""
import sys, numpy as np, cv2
from pathlib import Path
from dctr_c2_core import c2_fuse,synthetic_self_test
from c3t_fusion import c3t_fuse,C3T_THERMAL_THRESHOLD,signal_maps
from evaluate import evaluate_npz

assert synthetic_self_test()['status']=='PASS'
rng=np.random.default_rng(406)
v=rng.integers(0,256,(64,80,3),dtype=np.uint8)
i=rng.integers(0,256,(64,80,3),dtype=np.uint8)
s=cv2.GaussianBlur(v,(0,0),2.3)
c2=c2_fuse(v,s)
a,mask=c3t_fuse(v,i,s,c2,return_mask=True)
b,mask2=c3t_fuse(v,i,s,c2,return_mask=True)
assert C3T_THERMAL_THRESHOLD==0.30
assert a.dtype==np.uint8 and a.shape==v.shape
assert np.array_equal(a,b) and np.array_equal(mask,mask2)
assert np.array_equal(a[mask],s[mask])
assert np.array_equal(a[~mask],c2[~mask])
assert 0<=mask.mean()<=1
if len(sys.argv)>1:
 root=Path(sys.argv[1]); expected=[63.054613342650406,63.71760457874735,62.82604697680844]
 for seed,target in enumerate(expected):
  res=evaluate_npz(root/f'seed{seed}_test_nham_last.npz')
  assert abs(res['273']['mIoU13']-target)<1e-9
print('PASS: C2/C3T deterministic and 3-seed confusion scorer')
