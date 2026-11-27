"""Reproduce the team's mIoU13 from archived per-image confusion matrices.

Scoring: remove rows whose GT=0, but keep predictions of class 0 as errors;
exclude class 0 and absent classes from mIoU. C3-T evidence has 13 present classes.
This does NOT generate segmentation predictions or re-open test labels.
"""
import argparse, io, json
from pathlib import Path
import numpy as np

CLASSES=['unlabeled','road','sidewalk','building','traffic_light','traffic_sign',
         'vegetation','sky','person','car','truck','bus','motorcycle','bicycle','pole']

def score(conf):
    c=conf.copy()
    c[0,:]=0  # ignore unlabeled GT pixels exactly as group pipeline
    den=c.sum(axis=0)+c.sum(axis=1)-np.diag(c)
    iou=np.divide(np.diag(c)*100.,den,out=np.full(15,np.nan),where=den>0)
    return {'mIoU13':float(np.nanmean(iou[1:])),
            'pixel_acc':float(100*np.trace(c)/c.sum()),
            'per_class':{k:(float(v) if np.isfinite(v) else None) for k,v in zip(CLASSES,iou)}}

def evaluate_npz(path):
    with np.load(path,allow_pickle=False) as o:
        cm=o['conf']; ids=o['image_id']; leak=o['leak']
    assert cm.shape==(280,15,15) and len(ids)==280 and leak.sum()==7
    assert len(set(ids.tolist()))==280
    return {'280':score(cm.sum(axis=0)), '273':score(cm[~leak].sum(axis=0))}

def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence-dir',type=Path,required=True);p.add_argument('--output',type=Path)
    a=p.parse_args();res={}
    for seed in (0,1,2):
        v=evaluate_npz(a.evidence_dir/f'seed{seed}_test_nham_last.npz')
        orig=json.loads((a.evidence_dir/f'seed{seed}_test_ket_qua_last.json').read_text())
        for split in ('280','273'):
            assert abs(v[split]['mIoU13']-orig[split]['mIoU_co_mat'])<1e-9
            assert abs(v[split]['pixel_acc']-orig[split]['pixel_acc'])<1e-9
        res[str(seed)]=v
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(res,indent=2),encoding='utf8')
    for seed,x in res.items(): print(f"seed {seed}: 273 mIoU13={x['273']['mIoU13']:.6f}% 280={x['280']['mIoU13']:.6f}%")
if __name__=='__main__':main()
