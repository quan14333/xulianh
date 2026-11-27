"""Optional GT-free fusion metrics EN/SD/SCD for matched PNG pairs.

SCD convention here: corr(F-V,IR)+corr(F-IR,V) on [0,1] grayscale.
EN/SD/SCD implementations are explicit and not claimed to match other benchmarks
unless definitions and preprocessing have been harmonized.
MS-SSIM is NOT provided by this script, since no predeclared reference convention
was established for dual-source fusion; report this as missing instead.
"""
import argparse, csv
from pathlib import Path
import cv2, numpy as np

def gray(p):
 x=cv2.imread(str(p),cv2.IMREAD_COLOR)
 if x is None: raise ValueError(p)
 return cv2.cvtColor(x,cv2.COLOR_BGR2GRAY).astype(np.float64)/255.

def corr(a,b):
 a=a.ravel();b=b.ravel();return float(np.corrcoef(a,b)[0,1]) if a.std()>1e-9 and b.std()>1e-9 else 0.

def metrics(v,i,f):
 hist,_=np.histogram(f,bins=256,range=(0,1));p=hist[hist>0]/hist.sum()
 return {'EN_bits':float(-np.sum(p*np.log2(p))),'SD_0_1':float(np.std(f)),
         'SCD_explicit':corr(f-v,i)+corr(f-i,v)}

def main():
 ap=argparse.ArgumentParser()
 for k in ('vis','ir','fusion','output'):ap.add_argument('--'+k,type=Path,required=True)
 a=ap.parse_args();rows=[]
 for f in sorted(a.fusion.glob('*.png')):
  v=a.vis/f.name;i=a.ir/f.name
  if not (v.exists() and i.exists()): raise FileNotFoundError(f.name)
  vf=gray(v);inf=gray(i);ff=gray(f)
  if vf.shape != inf.shape or vf.shape != ff.shape:raise ValueError(f.name)
  rows.append({'image_id':f.stem,**metrics(vf,inf,ff)})
 if not rows:raise RuntimeError('No matched PNG')
 with a.output.open('w',newline='') as out:
  w=csv.DictWriter(out,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 print('DONE:',len(rows),'GT-free metric records')
if __name__=='__main__':main()
