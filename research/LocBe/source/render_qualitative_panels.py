"""Make properly labelled VIS/IR/fusion/GT/pred panels ONLY when real files exist.

This code does not fabricate missing test predictions, labels or sample selection.
Select a fixed set of IDs prior to looking at scoring; document them. Example:
  python source/render_qualitative_panels.py --vis path/Visible --ir path/Infrared
   --sage path/SAGE --c3t path/C3T --gt path/Label
   --sage-pred path/sage_pred --c3t-pred path/c3t_pred
   --output figures/qualitative --ids 00043 00044 00045 00046
"""
import argparse
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PALETTE=np.random.default_rng(406).integers(25,235,size=(15,3),dtype=np.uint8)
PALETTE[0]=0

def load_img(folder,image_id,color):
    p=folder/(image_id+'.png')
    if not p.is_file():raise FileNotFoundError(p)
    x=cv2.imread(str(p),cv2.IMREAD_COLOR if color else cv2.IMREAD_UNCHANGED)
    if x is None:raise RuntimeError('Unreadable image: '+str(p))
    if color:return cv2.cvtColor(x,cv2.COLOR_BGR2RGB)
    if x.ndim==3:
        if x.shape[2]!=3:raise ValueError('Not a 2D class-id mask: '+str(p))
        if np.array_equal(x[:,:,0],x[:,:,1]) and np.array_equal(x[:,:,1],x[:,:,2]):x=x[:,:,0]
        else:raise ValueError('GT/pred must be integer class-ID masks, not colorized PNG: '+str(p))
    if x.min()<0 or x.max()>14:raise ValueError('Mask class ID outside 0..14: '+str(p))
    return x

def colorize(mask):return PALETTE[mask]

def main():
 p=argparse.ArgumentParser()
 for x in ('vis','ir','sage','c3t','gt','sage_pred','c3t_pred','output'):p.add_argument('--'+x.replace('_','-'),required=True,type=Path)
 p.add_argument('--ids',nargs='+',required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 for iid in a.ids:
  vis=load_img(a.vis,iid,True);ir=load_img(a.ir,iid,True);sage=load_img(a.sage,iid,True);c3t=load_img(a.c3t,iid,True)
  gt=load_img(a.gt,iid,False);s_pred=load_img(a.sage_pred,iid,False);c_pred=load_img(a.c3t_pred,iid,False)
  if len({x.shape[:2] for x in (vis,ir,sage,c3t,gt,s_pred,c_pred)})!=1:raise ValueError('Unaligned shape: '+iid)
  diff=np.zeros((*gt.shape,3),dtype=np.uint8)
  diff[(gt!=0)&(c_pred!=gt)]=[230,40,40]
  diff[(gt!=0)&(s_pred!=gt)&(c_pred==gt)]=[35,170,65]
  fig,axs=plt.subplots(1,8,figsize=(22,3.5))
  arrs=[vis,ir,sage,c3t,colorize(gt),colorize(s_pred),colorize(c_pred),diff]
  titles=['VIS','IR','SAGE Fusion','C3-T Fusion','GT label IDs','SAGE pred','C3-T pred','C3-T error/red, win/green']
  for ax,im,t in zip(axs,arrs,titles):ax.imshow(im);ax.set_title(t,fontsize=8);ax.axis('off')
  fig.suptitle(f'{iid} — post-exposure exploratory / fixed output only',fontsize=11)
  fig.tight_layout();fig.savefig(a.output/(iid+'.png'),dpi=150);plt.close(fig)
 print('Created',len(a.ids),'real-data panels (no pixels fabricated)')
if __name__=='__main__':main()
