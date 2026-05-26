"""Unified evaluation: every checkpoint in rgb2thermal/checkpoints/* with best.pth is scored
on the TEST split at 512x640. Scalar metrics + color-LPIPS. Leaderboard + gallery + ensemble."""
import os, json, glob, csv
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import r2t_common as C
from train_a1 import UNetReg

dev='cuda' if torch.cuda.is_available() else 'cpu'
OUT=f"{C.BASE}/rgb2thermal/outputs"; os.makedirs(OUT,exist_ok=True)

def load_model(ckpt):
    d=torch.load(ckpt,map_location=dev); ar=d['args']
    in_ch=3+(1 if ar.get('use_depth') else 0)
    if ar.get('arch')=='physics':
        from train_a4 import PhysicsNet
        m=PhysicsNet(ar.get('encoder','convnext_tiny'),in_ch,K=ar.get('K',6)).to(dev)
    else:
        m=UNetReg(ar.get('encoder','convnext_tiny'),in_ch,bool(ar.get('use_alpha'))).to(dev)
    m.load_state_dict(d['model']); m.eval(); return m,ar

@torch.no_grad()
def predict(m,ar,ds,i):
    b={k:(v.unsqueeze(0) if torch.is_tensor(v) else v) for k,v in ds[i].items()}
    x=C.make_input(b,ar.get('use_depth',0)).to(dev)
    al=b['alpha'].to(dev) if m.use_alpha else None
    return m(x,al)[0,0].cpu().numpy(), b['target'][0,0].numpy()

def color_lpips(lpfn,pred,gt):
    if lpfn is None: return float('nan')
    pc=C.to_color(pred).astype(np.float32)/127.5-1; gc=C.to_color(gt).astype(np.float32)/127.5-1
    pt=torch.from_numpy(pc.transpose(2,0,1)[None]).to(dev); gt_=torch.from_numpy(gc.transpose(2,0,1)[None]).to(dev)
    return float(lpfn(pt,gt_).item())

def main():
    sp=C.load_split(); test=sp['test']
    try:
        import lpips; lpfn=lpips.LPIPS(net='alex').to(dev).eval()
    except Exception: lpfn=None
    ckpts=sorted(glob.glob(f"{C.BASE}/rgb2thermal/checkpoints/*/best.pth"))
    print("checkpoints:",[c.split('/')[-2] for c in ckpts],flush=True)
    rows=[]; preds_by_model={}
    # mean-field floor
    tr=C.R2TDataset(sp['train'],augment=False,use_depth=0)
    meanfield=np.mean([np.load(f"{C.CACHE}/scalar/{n}.npy").astype(np.float32) for n in tr.names[:60]],axis=0)
    for ck in ckpts:
        name=ck.split('/')[-2]; m,ar=load_model(ck)
        ds=C.R2TDataset(test,augment=False,use_depth=ar.get('use_depth',0),unreg=bool(ar.get('unreg',0)))
        ms=[]; preds=[]
        for i in range(len(test)):
            p,t=predict(m,ar,ds,i); preds.append(p)
            d=C.metrics_np(p,t); d['ssim']=C.ssim_np(p,t,dev); d['clpips']=color_lpips(lpfn,p,t)
            ms.append(d)
        preds_by_model[name]=preds
        agg={k:float(np.nanmean([x[k] for x in ms])) for k in ms[0]}
        agg['model']=name; rows.append(agg); print(f"  {name}: mae={agg['mae']:.4f} psnr={agg['psnr']:.2f} ssim={agg['ssim']:.3f} clpips={agg['clpips']:.3f}",flush=True)
        del m; torch.cuda.empty_cache()
    # mean-field floor row
    ms=[]
    for i,n in enumerate([x.replace('.JPG','') for x in test]):
        t=np.load(f"{C.CACHE}/scalar/{n}.npy").astype(np.float32)
        d=C.metrics_np(meanfield,t); d['ssim']=C.ssim_np(meanfield,t,dev); d['clpips']=color_lpips(lpfn,meanfield,t); ms.append(d)
    fl={k:float(np.nanmean([x[k] for x in ms])) for k in ms[0]}; fl['model']='[mean-field floor]'; rows.append(fl)
    # ensemble top-3 by mae
    real=[r for r in rows if not r['model'].startswith('[')]
    top=sorted(real,key=lambda r:r['mae'])[:3]
    if len(top)>=2:
        ens=[]; names=[r['model'] for r in top]
        for i in range(len(test)):
            p=np.mean([preds_by_model[nm][i] for nm in names],axis=0)
            t=np.load(f"{C.CACHE}/scalar/{test[i].replace('.JPG','')}.npy").astype(np.float32)
            d=C.metrics_np(p,t); d['ssim']=C.ssim_np(p,t,dev); d['clpips']=color_lpips(lpfn,p,t); ens.append(d)
        er={k:float(np.nanmean([x[k] for x in ens])) for k in ens[0]}; er['model']=f"[ensemble:{'+'.join(names)}]"; rows.append(er)
    rows=sorted(rows,key=lambda r:r['mae'])
    keys=['model','mae','rmse','psnr','ssim','corr','clpips']
    with open(f"{OUT}/leaderboard.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader()
        for r in rows: w.writerow({k:r.get(k) for k in keys})
    print("\n=== LEADERBOARD (test, by MAE) ===")
    print(f"{'model':40s} {'MAE':>7} {'PSNR':>6} {'SSIM':>6} {'corr':>6} {'cLPIPS':>7}")
    for r in rows: print(f"{r['model'][:40]:40s} {r['mae']:7.4f} {r['psnr']:6.2f} {r['ssim']:6.3f} {r['corr']:6.3f} {r['clpips']:7.3f}")
    # gallery
    gn=[t for t in test[:5]]; mods=[r['model'] for r in real][:6]
    fig,ax=plt.subplots(len(gn),2+len(mods),figsize=(3*(2+len(mods)),3*len(gn))); ax=np.atleast_2d(ax)
    for r,tn in enumerate(gn):
        nn=tn.replace('.JPG','')
        rgb=np.asarray(Image.open(f"{C.CACHE}/reg_rgb/{nn}.png")); gt=np.load(f"{C.CACHE}/scalar/{nn}.npy").astype(np.float32)
        ax[r,0].imshow(rgb);ax[r,0].axis('off');ax[r,0].set_title(nn if r==0 else "",fontsize=8)
        ax[r,0].set_ylabel(nn)
        ax[r,1].imshow(gt,cmap='inferno',vmin=0,vmax=1);ax[r,1].axis('off')
        if r==0: ax[r,1].set_title("GT",fontsize=9)
        for c,mod in enumerate(mods):
            idx=test.index(tn); p=preds_by_model[mod][idx]
            ax[r,2+c].imshow(p,cmap='inferno',vmin=0,vmax=1);ax[r,2+c].axis('off')
            if r==0: ax[r,2+c].set_title(mod[:14],fontsize=8)
    plt.tight_layout();plt.savefig(f"{OUT}/comparison_gallery.png",dpi=85,bbox_inches='tight');plt.close()
    json.dump(rows,open(f"{OUT}/leaderboard.json","w"),indent=2)
    print("\nwrote",f"{OUT}/leaderboard.csv",f"{OUT}/comparison_gallery.png")

if __name__=='__main__': main()
