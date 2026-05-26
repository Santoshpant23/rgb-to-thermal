"""v2 — flip-TTA per model + val-tuned WEIGHTED ensemble on official 202. Solar-aware.
Reuses registration/eval helpers from eval_official (imported as EO)."""
import os, json, glob, csv
import numpy as np, torch
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import r2t_common as C
import eval_official as EO
dev=EO.dev; IMN,IST=EO.IMN,EO.IST
SOLAR=json.load(open(f"{C.CACHE}/solar.json")) if os.path.exists(f"{C.CACHE}/solar.json") else {}
ORDER=['a1_rgb','a1_rgb_s2_512','a2_gan','a4_physics8','a1_solar_512','a1_rgbda','a4_physics','a1_rgbd','a1_small','a1_unreg']
OUT=f"{C.BASE}/rgb2thermal_v2/outputs"; os.makedirs(OUT,exist_ok=True)

def alpha_for(split,nm):
    try:
        dji=split[nm][0]; tif=f"{C.BASE}/alphaearth-emb/satellite_embedding_{dji.replace('.JPG','.')}.tif"
        if os.path.exists(tif):
            import rasterio
            with rasterio.open(tif) as s: a=s.read()
            return np.nan_to_num(a,nan=0,posinf=0,neginf=0).mean(axis=(1,2)).astype(np.float32)
    except Exception: pass
    return np.zeros(64,np.float32)

@torch.no_grad()
def tta(m,ar,rgb,depth,alpha,sol2=None):
    ps=[]
    for fl in (None,'h','v'):
        r=rgb.copy(); d=depth.copy()
        if fl=='h': r=r[:,::-1]; d=d[:,::-1]
        if fl=='v': r=r[::-1]; d=d[::-1]
        t=((torch.from_numpy(np.ascontiguousarray(r.transpose(2,0,1))).float()-IMN)/IST)
        chans=[t]
        if ar.get('solar') and sol2 is not None:
            s=torch.tensor([sol2[0]/90.0,sol2[1]/360.0],dtype=torch.float32).view(2,1,1).expand(2,t.shape[1],t.shape[2])
            chans.append(s)
        x=torch.cat(chans,0).unsqueeze(0)
        if ar.get('use_depth'): x=torch.cat([x,torch.from_numpy(np.ascontiguousarray(d[None,None])).float()],1)
        al=torch.from_numpy(np.nan_to_num(alpha)[None]).float().to(dev) if getattr(m,'use_alpha',False) else None
        p=np.nan_to_num(m(x.to(dev),al)[0,0].cpu().numpy()).clip(0,1)
        if fl=='h': p=p[:,::-1]
        if fl=='v': p=p[::-1]
        ps.append(np.ascontiguousarray(p))
    return np.mean(ps,0)

def metrics(p,gt,lpfn):
    d=C.metrics_np(p,gt); d['ssim']=C.ssim_np(p,gt,dev)
    if lpfn is not None:
        pc=C.to_color(p).astype(np.float32)/127.5-1; gc=C.to_color(gt).astype(np.float32)/127.5-1
        d['clpips']=float(lpfn(torch.from_numpy(pc.transpose(2,0,1)[None]).to(dev),torch.from_numpy(gc.transpose(2,0,1)[None]).to(dev)).item())
    else: d['clpips']=float('nan')
    return d

@torch.no_grad()
def main():
    split=json.load(open(f"{C.BASE}/code/train_test_split.json"))
    models={ck.split('/')[-2]:EO.load_model(ck) for ck in sorted(glob.glob(f"{C.BASE}/rgb2thermal/checkpoints/*/best.pth"))}
    ALL=[k for k in ORDER if k in models]
    CAND=[k for k in ['a1_rgb','a1_rgb_s2_512','a2_gan','a4_physics8','a1_solar_512','a1_rgbda'] if k in models]
    ENS3=[k for k in ['a1_rgb','a2_gan','a4_physics8'] if k in models]
    print("models:",list(models.keys()),"| CAND:",CAND,flush=True)
    try: import lpips; lpfn=lpips.LPIPS(net='alex').to(dev).eval()
    except Exception: lpfn=None
    from transformers import pipeline
    dpipe=pipeline("depth-estimation",model="depth-anything/Depth-Anything-V2-Small-hf",device=0 if dev=='cuda' else -1)

    # ---- tune ensemble weights on internal VAL (cached) ----
    sp=C.load_split(); val=sp['val']; Pv={k:[] for k in CAND}; Gv=[]
    for nm in val:
        n=nm.replace('.JPG','')
        rgb=np.asarray(Image.open(f"{C.CACHE}/reg_rgb/{n}.png").convert("RGB")).astype(np.float32)/255
        dp=f"{C.CACHE}/depth/{n}.npy"; depth=np.load(dp).astype(np.float32) if os.path.exists(dp) else np.zeros((512,640),np.float32)
        ap=f"{C.CACHE}/alpha/{n}.npy"; alpha=np.load(ap).astype(np.float32).reshape(64,-1).mean(1) if os.path.exists(ap) else np.zeros(64,np.float32)
        sol2=SOLAR.get(n); Gv.append(np.load(f"{C.CACHE}/scalar/{n}.npy").astype(np.float32))
        for k in CAND: Pv[k].append(tta(*models[k],rgb,depth,alpha,sol2))
    P=np.stack([np.stack(Pv[k]) for k in CAND],0); G=np.stack(Gv)
    from scipy.optimize import minimize
    def obj(w):
        w=np.clip(w,0,None); w=w/(w.sum()+1e-8); return float(np.abs(np.tensordot(w,P,axes=(0,0))-G).mean())
    res=minimize(obj,np.ones(len(CAND))/len(CAND),method='Nelder-Mead',options=dict(maxiter=800,xatol=1e-3,fatol=1e-6))
    W=np.clip(res.x,0,None); W=W/(W.sum()+1e-8)
    print("tuned weights (val):",{k:round(float(w),3) for k,w in zip(CAND,W)},"val MAE",round(obj(res.x),4),flush=True)

    # ---- official 202 with TTA ----
    names=sorted([f for f in os.listdir(EO.TH) if f.upper().endswith('.JPG')],key=lambda x:int(x.split('.')[0]))
    acc={k:[] for k in ALL}; acc['ens_w']=[]; acc['ens3']=[]; regq=[]
    gidx=set(np.linspace(0,len(names)-1,12).astype(int)); gstash={}
    for i,nm in enumerate(names):
        th=Image.open(f"{EO.TH}/{nm}").convert("RGB"); gt=EO.to_scalar(np.array(th.resize((640,512))))
        e_th=EO.edges(EO.to_scalar(np.array(th.resize((320,256)))))
        rgb_pil=Image.open(f"{EO.RGB}/{nm}").convert("RGB"); box,q=EO.register(rgb_pil,e_th); regq.append(q)
        rgb_reg=np.array(rgb_pil.crop(box).resize((640,512))).astype(np.float32)/255
        rgb_full=np.array(rgb_pil.resize((640,512))).astype(np.float32)/255
        dd=dpipe(Image.fromarray((rgb_reg*255).astype(np.uint8)))["predicted_depth"].squeeze().detach().cpu().numpy().astype(np.float32)
        dd=np.array(Image.fromarray(((dd-dd.min())/(np.ptp(dd)+1e-8)*255).astype(np.uint8)).resize((640,512)))/255.0
        al=alpha_for(split,nm); sol2=SOLAR.get(nm.replace('.JPG',''))
        preds={}
        for k in ALL:
            m,ar=models[k]; rgb=rgb_full if ar.get('unreg') else rgb_reg
            preds[k]=tta(m,ar,rgb,dd,al,sol2); acc[k].append(metrics(preds[k],gt,lpfn))
        ew=np.tensordot(W,np.stack([preds[k] for k in CAND]),axes=(0,0)); acc['ens_w'].append(metrics(ew,gt,lpfn))
        if ENS3: acc['ens3'].append(metrics(np.mean([preds[k] for k in ENS3],0),gt,lpfn))
        if i in gidx: gstash[nm]=(rgb_reg,gt,preds,ew)
        if (i+1)%50==0: print(f"  {i+1}/{len(names)}",flush=True)
    rows=[]
    for k in list(acc.keys()):
        if not acc[k]: continue
        agg={kk:float(np.nanmean([x[kk] for x in acc[k]])) for kk in acc[k][0]}
        agg['model']=('[ens_weighted]' if k=='ens_w' else '[ens3_equal]' if k=='ens3' else k+'+TTA'); rows.append(agg)
    rows=sorted(rows,key=lambda r:r['mae']); keys=['model','mae','rmse','psnr','ssim','corr','clpips']
    with open(f"{OUT}/leaderboard_v2.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader()
        for r in rows: w.writerow({k:r.get(k) for k in keys})
    print(f"\n=== v2 OFFICIAL-202 LEADERBOARD (TTA + weighted ensemble), reg {np.mean(regq):.3f} ===")
    print(f"{'model':20s}{'MAE':>8}{'PSNR':>7}{'SSIM':>7}{'corr':>7}{'cLPIPS':>8}")
    for r in rows: print(f"{r['model']:20s}{r['mae']:8.4f}{r['psnr']:7.2f}{r['ssim']:7.3f}{r['corr']:7.3f}{r['clpips']:8.3f}")
    gn=sorted(gstash.keys(),key=lambda x:int(x.split('.')[0])); show=[k for k in ['a1_rgb','a2_gan','a1_solar_512'] if k in ALL]
    cols=['RGB','GT','ens_weighted']+show
    fig,ax=plt.subplots(len(gn),len(cols),figsize=(2.1*len(cols),2.1*len(gn))); ax=np.atleast_2d(ax)
    for r,nm in enumerate(gn):
        rgb_reg,gt,preds,ew=gstash[nm]
        ax[r,0].imshow((rgb_reg*255).astype(np.uint8)); ax[r,0].axis('off'); ax[r,0].set_ylabel(nm,fontsize=7)
        ax[r,1].imshow(gt,cmap='inferno',vmin=0,vmax=1); ax[r,1].axis('off')
        ax[r,2].imshow(ew,cmap='inferno',vmin=0,vmax=1); ax[r,2].axis('off')
        for c,k in enumerate(show): ax[r,3+c].imshow(preds[k],cmap='inferno',vmin=0,vmax=1); ax[r,3+c].axis('off')
        if r==0:
            for c,t in enumerate(cols): ax[0,c].set_title(t,fontsize=8)
    plt.tight_layout(); plt.savefig(f"{OUT}/v2_official_gallery.png",dpi=95,bbox_inches='tight'); plt.close()
    print("wrote leaderboard_v2.csv + v2_official_gallery.png")

if __name__=='__main__': main()
