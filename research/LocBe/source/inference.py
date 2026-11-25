"""Create C3-T PNG from corresponding VIS, IR and previously exported SAGE PNG.

NOTE: Generate SAGE PNG using pinned group SAGE Student checkpoint beforehand.
This script never sees labels or segmentation predictions.
"""
import argparse, hashlib, json
from pathlib import Path
import cv2
from c3t_fusion import c3t_fuse

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda: f.read(2**20),b''):h.update(b)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser()
 for k in ('vis_dir','ir_dir','sage_dir','output_dir'): p.add_argument('--'+k.replace('_','-'), required=True, type=Path)
 a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
 manifest=[]
 for vis in sorted(a.vis_dir.glob('*.png')):
  ir=a.ir_dir/vis.name; sage=a.sage_dir/vis.name
  if not ir.exists() or not sage.exists(): raise FileNotFoundError(f'Missing paired IR/SAGE for {vis.name}')
  v=cv2.imread(str(vis),cv2.IMREAD_COLOR);i=cv2.imread(str(ir),cv2.IMREAD_COLOR);s=cv2.imread(str(sage),cv2.IMREAD_COLOR)
  if any(x is None for x in (v,i,s)): raise ValueError(f'Corrupted image {vis.name}')
  out,guard=c3t_fuse(v,i,s,return_mask=True)
  dst=a.output_dir/vis.name
  if not cv2.imwrite(str(dst),out): raise IOError(f'Cannot write {dst}')
  manifest.append({'image_id':vis.stem,'filename':vis.name,'sha256':sha(dst),'guard_fraction':float(guard.mean())})
 if not manifest: raise RuntimeError('No PNG found')
 (a.output_dir/'C3T_export_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
 print('DONE:',len(manifest),'images in',a.output_dir)
if __name__=='__main__':main()
