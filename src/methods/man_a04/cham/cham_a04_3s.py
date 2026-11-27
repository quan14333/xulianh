"""26/11: a04 3 seed (s0,s1,s2) Dev228 B3, biên chốt trước: 13 lớp cận dưới CI a04-TB > -0,5."""
import numpy as np
G='D:/CS406/DoAn/Sage/tai_lap/_kaggle_out/dev228_gop/'
T=["0","road","sidewalk","building","light","sign","veg","sky","person","car","truck","bus","moto","bicycle","pole"]
C8=[1,3,6,7,8,9,14,5]; C5=[2,4,10,11,12]; S=(0,1,2)
def nap(b,s):
    z=np.load(f'{G}{b}_b3_s{s}_15lop/dev_nham_last.npz'); a=z['image_id'].astype(int); o=np.argsort(a); return a[o],z['conf'][o].astype(float)
def iou(c):
    c=c.copy(); c[0]=0; u=c.sum(0)+c.sum(1)-np.diag(c)
    return np.where(c.sum(1)>0,100*np.diag(c)/np.maximum(u,1),np.nan)
F={'13':lambda v:np.nanmean(v[1:]),'8':lambda v:v[C8].mean(),'5':lambda v:v[C5].mean()}
BT=['tbsua_a04','tbsua_992','tb','nosam_992','sage','vis']; A='tbsua_a04'
D={(b,s):nap(b,s) for b in BT for s in S}; ids=D['tb',0][0]; assert all((v[0]==ids).all() for v in D.values()) and len(ids)==228
V={k:iou(v[1].sum(0)) for k,v in D.items()}
print('kiểm TB s0 13 phải 60,29:',round(F['13'](V['tb',0]),2),'| TB s2 13/8 phải 60,07/77,23:',round(F['13'](V['tb',2]),2),round(F['8'](V['tb',2]),2))
for n,f in F.items():
    print(f'--- {n} lớp')
    for b in BT:
        x=[f(V[b,s]) for s in S]; print(f'{b:10s}',' / '.join(f'{v:.2f}' for v in x),f'TB {np.mean(x):.2f}')
k8=np.mean([F['8'](V[A,s]) for s in S]); h5=np.mean([F['5'](V[A,s])-F['5'](V['tb',s]) for s in S])
print(f"LUẬT: 8 lớp {k8:.3f} (>=77,62: {k8>=77.62}) | 5 lớp hiệu so TB {h5:+.3f} (>=0: {h5>=0})")
print('lớp       ',' '.join(BT),'| a04-tb s0/s1/s2 | a04-sage s0/s1/s2')
for i in range(1,15):
    if i==13: continue
    g=lambda b:' / '.join(f"{V[A,s][i]-V[b,s][i]:+.1f}" for s in S)
    print(f'{T[i]:9s}',' '.join(f'{np.mean([V[b,s][i] for s in S]):6.1f}' for b in BT),' |',g('tb'),' |',g('sage'))
N=len(ids); bl=[np.arange(i,min(i+10,N)) for i in range(0,N,10)]; rng=np.random.default_rng(0)
for b in ('sage','tb','nosam_992','tbsua_992'):
    for n,f in F.items():
        d=[f(V[A,s])-f(V[b,s]) for s in S]; X=[]
        for _ in range(500):
            ii=np.concatenate([bl[j] for j in rng.integers(0,len(bl),len(bl))])
            X.append(np.mean([f(iou(D[A,s][1][ii].sum(0)))-f(iou(D[b,s][1][ii].sum(0))) for s in S]))
        print(f"a04-{b:9s} {n:>2s} lớp: {' / '.join(f'{x:+.2f}' for x in d)} TB {np.mean(d):+.2f} CI ảnh {np.nanpercentile(X,[2.5,97.5]).round(2)}")
