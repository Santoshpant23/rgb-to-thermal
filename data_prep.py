#!/usr/bin/env python3
"""
Phase 1 — DATA FOUNDATION for RGB->Thermal.
1) Palette inversion: learn a 1-D LUT from the thermal palette -> recover scalar heat field.
2) Registration: thermal sensor sees a central crop (~0.65 width) of the 12MP RGB; estimate
   global crop fraction + per-image translation refine; produce registered RGB at 640x512.
3) Targets: scalar thermal 640x512. Priors: Depth-Anything depth, AlphaEarth coarse (cleaned).
4) Fixed train/val/test split. 5) Sanity gallery.

Idempotent: skips already-cached items. Defensive: per-image try/except.
"""
import os, json, glob, sys, time
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

BASE = "/home/spant/UMich/umich-hackathon"
RGB_DIR = f"{BASE}/data/Train_2/RGB"
TH_DIR  = f"{BASE}/data/Train_2/Thermal"
EMB_DIR = f"{BASE}/alphaearth-emb"
SPLIT_J = f"{BASE}/code/train_test_split.json"
OUT     = f"{BASE}/rgb2thermal/data_cache"
GAL     = f"{BASE}/rgb2thermal/outputs/data_sanity"
TW, TH_ = 640, 512                      # target width,height (thermal native)
ASPECT  = TW/TH_                        # 1.25
for d in ["reg_rgb","scalar","depth","alpha"]:
    os.makedirs(f"{OUT}/{d}", exist_ok=True)
os.makedirs(GAL, exist_ok=True)

def log(*a):
    print(*a, flush=True)

# ----------------------------------------------------------------- palette LUT
def build_lut(th_files, n=60):
    cols=[]
    for f in th_files[:n]:
        a=np.array(Image.open(f).convert("RGB").resize((160,128))).reshape(-1,3).astype(np.float32)
        cols.append(a)
    C=np.concatenate(cols,0); mean=C.mean(0)
    _,_,Vt=np.linalg.svd(C-mean, full_matrices=False); pc1=Vt[0]
    proj=(C-mean)@pc1
    # orient so LUT[0] is "cool/dark"
    qs=np.quantile(proj, np.linspace(0,1,257))
    binid=np.clip(np.digitize(proj, qs[1:-1]),0,255)
    lut=np.zeros((256,3),np.float32)
    for b in range(256):
        m=binid==b; lut[b]=C[m].mean(0) if m.any() else (lut[b-1] if b>0 else mean)
    # orient by luminance of endpoints (low scalar -> darker)
    if lut[0].mean() > lut[-1].mean():
        lut=lut[::-1].copy()
    return lut

def build_inv_lut(lut, bits=6):
    """3D coarse inverse LUT: bin RGB (2^bits per channel) -> nearest palette index."""
    step=256//(2**bits); n=2**bits
    centers=(np.arange(n)*step+step/2)
    gr,gg,gb=np.meshgrid(centers,centers,centers,indexing='ij')
    grid=np.stack([gr,gg,gb],-1).reshape(-1,3).astype(np.float32)   # (n^3,3)
    # nearest palette idx for each bin center (chunked)
    out=np.empty(len(grid),np.uint8)
    for i in range(0,len(grid),20000):
        ch=grid[i:i+20000]
        d=np.linalg.norm(ch[:,None,:]-lut[None],axis=2)
        out[i:i+len(ch)]=d.argmin(1).astype(np.uint8)
    return out.reshape(n,n,n), step

def to_scalar(img_rgb_uint8, inv, step):
    a=np.asarray(img_rgb_uint8)
    idx=(a//step).astype(np.int32)
    idx=np.clip(idx,0,inv.shape[0]-1)
    return inv[idx[...,0],idx[...,1],idx[...,2]].astype(np.float32)/255.0

# ----------------------------------------------------------------- edges / registration
def edges(x):
    x=x.astype(np.float32)
    gy,gx=np.gradient(x); m=np.hypot(gx,gy)
    return m-m.mean()
def ncc(a,b): return float((a*b).mean()/(a.std()*b.std()+1e-8))

def crop_box(W,H,c,fx=0.0,fy=0.0):
    bw=c*W; bh=bw/ASPECT
    if bh>H: bh=H; bw=bh*ASPECT
    cx=W/2+fx*W; cy=H/2+fy*H
    l=cx-bw/2; t=cy-bh/2
    l=min(max(0,l),W-bw); t=min(max(0,t),H-bh)
    return (l,t,l+bw,t+bh)

def estimate_global_c(pairs, inv, step, sample=24):
    th_e={}; best_c=0.65; best=-9
    for c in np.linspace(0.45,0.95,11):
        cs=[]
        for f in pairs[:sample]:
            try:
                th=Image.open(f"{TH_DIR}/{f}").resize((320,256));
                e_th=th_e.get(f)
                if e_th is None:
                    e_th=edges(to_scalar(np.array(th.convert('RGB')),inv,step)); th_e[f]=e_th
                rg=Image.open(f"{RGB_DIR}/{f}").convert("L"); W,H=rg.size
                box=crop_box(W,H,c); crop=rg.crop(box).resize((320,256))
                cs.append(ncc(e_th,edges(np.array(crop)/255.0)))
            except Exception: pass
        m=float(np.mean(cs)) if cs else -9
        if m>best: best=m; best_c=float(c)
    return best_c, best

def refine_offset(fname, c, inv, step):
    """small translation search around center crop maximizing edge-ncc; return (fx,fy,quality,box)."""
    th=Image.open(f"{TH_DIR}/{fname}").resize((320,256))
    e_th=edges(to_scalar(np.array(th.convert('RGB')),inv,step))
    rg=Image.open(f"{RGB_DIR}/{fname}").convert("L"); W,H=rg.size
    best=(-9,0,0)
    for fx in (-0.08,-0.04,0,0.04,0.08):
        for fy in (-0.08,-0.04,0,0.04,0.08):
            box=crop_box(W,H,c,fx,fy)
            crop=rg.crop(box).resize((320,256))
            q=ncc(e_th,edges(np.array(crop)/255.0))
            if q>best[0]: best=(q,fx,fy)
    q,fx,fy=best
    return fx,fy,q,crop_box(W,H,c,fx,fy)

# ----------------------------------------------------------------- alphaearth
def load_alpha(fname, split):
    try:
        import rasterio
        dji=split[fname][0]
        tif=f"{EMB_DIR}/satellite_embedding_{dji.replace('.JPG','.')}.tif"
        if not os.path.exists(tif): return None
        with rasterio.open(tif) as src: d=src.read()  # (64,24,18)
        d=np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float16)
        return d
    except Exception as e:
        return None

# ----------------------------------------------------------------- main
def main():
    t0=time.time()
    rgbset={f for f in os.listdir(RGB_DIR) if f.upper().endswith('.JPG')}
    thset ={f for f in os.listdir(TH_DIR)  if f.upper().endswith('.JPG')}
    pairs=sorted(rgbset & thset, key=lambda x:int(x.split('.')[0]))
    log(f"usable pairs (RGB & Thermal): {len(pairs)}")
    th_files=[f"{TH_DIR}/{f}" for f in sorted(thset)]
    split=json.load(open(SPLIT_J))

    # ---- LUT ----
    lutp=f"{OUT}/lut.npy"
    if os.path.exists(lutp): lut=np.load(lutp)
    else:
        lut=build_lut(th_files); np.save(lutp,lut); log("built LUT")
    inv,step=build_inv_lut(lut,bits=6); log("built inverse LUT")
    # recon residual check
    res=[]
    for f in pairs[:5]:
        a=np.array(Image.open(f"{TH_DIR}/{f}").convert("RGB").resize((160,128)))
        sc=to_scalar(a,inv,step); rec=(lut[np.clip((sc*255).astype(int),0,255)])
        res.append(float(np.linalg.norm(a.astype(np.float32)-rec,axis=2).mean()))
    log(f"palette recon residual mean={np.mean(res):.2f} (inferno~42)")

    # ---- global c ----
    c,gq=estimate_global_c(pairs,inv,step); log(f"global crop c={c:.3f} (mean edge-ncc {gq:.3f})")

    # ---- depth model (gated) ----
    depth_pipe=None
    try:
        import torch
        from transformers import pipeline
        dev=0 if torch.cuda.is_available() else -1
        depth_pipe=pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=dev)
        log("Depth-Anything loaded")
    except Exception as e:
        log("Depth model unavailable -> skip depth:", repr(e)[:120])

    # ---- per image ----
    meta={}; qualities=[]
    for i,f in enumerate(pairs):
        try:
            regp=f"{OUT}/reg_rgb/{f.replace('.JPG','')}.png"
            scp =f"{OUT}/scalar/{f.replace('.JPG','')}.npy"
            dpp =f"{OUT}/depth/{f.replace('.JPG','')}.npy"
            alp =f"{OUT}/alpha/{f.replace('.JPG','')}.npy"
            fx,fy,q,box=refine_offset(f,c,inv,step); qualities.append(q)
            # registered RGB 640x512
            rg=Image.open(f"{RGB_DIR}/{f}").convert("RGB")
            reg=rg.crop(box).resize((TW,TH_), Image.BILINEAR)
            if not os.path.exists(regp): reg.save(regp)
            # scalar target 640x512
            if not os.path.exists(scp):
                tha=np.array(Image.open(f"{TH_DIR}/{f}").convert("RGB").resize((TW,TH_)))
                sc=to_scalar(tha,inv,step).astype(np.float16); np.save(scp,sc)
            # depth on registered RGB
            if depth_pipe is not None and not os.path.exists(dpp):
                try:
                    dout=depth_pipe(reg)["predicted_depth"]
                    dd=dout.squeeze().detach().cpu().numpy().astype(np.float32)
                    dd=np.array(Image.fromarray(((dd-dd.min())/(np.ptp(dd)+1e-8)*255).astype(np.uint8)).resize((TW,TH_)))/255.0
                    np.save(dpp, dd.astype(np.float16))
                except Exception as e:
                    if i==0: log("depth fail:", repr(e)[:100])
            # alphaearth coarse
            if not os.path.exists(alp):
                a=load_alpha(f,split)
                if a is not None: np.save(alp,a)
            meta[f]=dict(fx=fx,fy=fy,quality=float(q),box=[float(x) for x in box])
            if (i+1)%50==0 or i==0: log(f"  [{i+1}/{len(pairs)}] {f} q={q:.3f} t={time.time()-t0:.0f}s")
        except Exception as e:
            log("ERR", f, repr(e)[:120])
    q=np.array(qualities)
    log(f"registration quality: mean={q.mean():.3f} med={np.median(q):.3f} <0.05frac={(q<0.05).mean():.2f}")

    # ---- split ----
    rng=np.random.RandomState(42); idx=np.arange(len(pairs)); rng.shuffle(idx)
    n=len(pairs); nval=max(8,int(0.1*n)); ntest=max(8,int(0.1*n))
    test=[pairs[i] for i in idx[:ntest]]; val=[pairs[i] for i in idx[ntest:ntest+nval]]
    train=[pairs[i] for i in idx[ntest+nval:]]
    json.dump(dict(train=train,val=val,test=test), open(f"{OUT}/split.json","w"))
    log(f"split: train={len(train)} val={len(val)} test={len(test)}")
    json.dump(dict(global_c=c, step=int(step), meta=meta), open(f"{OUT}/meta.json","w"))

    # ---- sanity gallery ----
    show=val[:4]
    fig,ax=plt.subplots(len(show),4,figsize=(16,4*len(show))); ax=np.atleast_2d(ax)
    for r,f in enumerate(show):
        reg=np.array(Image.open(f"{OUT}/reg_rgb/{f.replace('.JPG','')}.png"))
        sc=np.load(f"{OUT}/scalar/{f.replace('.JPG','')}.npy").astype(np.float32)
        dpf=f"{OUT}/depth/{f.replace('.JPG','')}.npy"; dp=np.load(dpf).astype(np.float32) if os.path.exists(dpf) else np.zeros_like(sc)
        e_o=np.dstack([(edges(np.array(Image.fromarray(reg).convert('L'))/255.0)),(edges(sc)),np.zeros_like(sc)])
        e_o=(e_o-e_o.min())/(np.ptp(e_o)+1e-8)
        ax[r,0].imshow(reg); ax[r,0].set_title(f"{f} registered RGB"); ax[r,0].axis("off")
        ax[r,1].imshow(sc,cmap="inferno",vmin=0,vmax=1); ax[r,1].set_title("scalar target"); ax[r,1].axis("off")
        ax[r,2].imshow(dp,cmap="viridis"); ax[r,2].set_title("depth prior"); ax[r,2].axis("off")
        ax[r,3].imshow(e_o); ax[r,3].set_title("edge overlay R=RGB G=therm"); ax[r,3].axis("off")
    plt.tight_layout(); plt.savefig(f"{GAL}/gallery.png",dpi=90,bbox_inches="tight"); plt.close()
    log(f"sanity gallery -> {GAL}/gallery.png   total {time.time()-t0:.0f}s")

if __name__=="__main__":
    main()
