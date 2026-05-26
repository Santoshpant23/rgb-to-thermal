"""Evaluate all models on the OFFICIAL unseen Test_2 set (202 imgs, names 469-670, RGB<->Thermal same filename).
Per-image registration (RGB cropped+shifted to thermal FOV using the now-available thermal),
scalar GT via palette LUT, depth on the fly. Outputs leaderboard_official.csv + gallery."""
import os, json, glob, csv
import numpy as np, torch
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import r2t_common as C
from train_a1 import UNetReg
dev='cuda' if torch.cuda.is_available() else 'cpu'
TH=f"{C.BASE}/data/Test_2/Thermal"; RGB=f"{C.BASE}/data/Test_2/RGB"; OUT=f"{C.BASE}/rgb2thermal/outputs"
IMN=torch.tensor([0.485,0.456,0.406]).view(3,1,1); IST=torch.tensor([0.229,0.224,0.225]).view(3,1,1)
ORDER=['ensemble','a1_rgb','a2_gan','a4_physics8','a1_rgbda','a4_physics','a1_rgbd','a1_small','a1_unreg']; ENS=['a1_rgb','a2_gan','a4_physics8']
lut=C.load_lut()
def build_inv(lut,bits=6):
    step=256//(2**bits);n=2**bits;c=(np.arange(n)*step+step/2)
    gr,gg,gb=np.meshgrid(c,c,c,indexing='ij');grid=np.stack([gr,gg,gb],-1).reshape(-1,3).astype(np.float32)
    out=np.empty(len(grid),np.uint8)
    for i in range(0,len(grid),20000):
        ch=grid[i:i+20000];d=np.linalg.norm(ch[:,None]-lut[None],axis=2);out[i:i+len(ch)]=d.argmin(1).astype(np.uint8)
    return out.reshape(n,n,n),step
inv,step=build_inv(lut)
def to_scalar(arr):
    idx=np.clip((arr//step).astype(int),0,inv.shape[0]-1);return inv[idx[...,0],idx[...,1],idx[...,2]].astype(np.float32)/255.0
def edges(x):
    gy,gx=np.gradient(x.astype(np.float32));m=np.hypot(gx,gy);return m-m.mean()
def crop_box(W,H,c,fx,fy):
    bw=c*W;bh=bw/1.25
    if bh>H:bh=H;bw=bh*1.25
    cx=W/2+fx*W;cy=H/2+fy*H;l=cx-bw/2;t=cy-bh/2;l=min(max(0,l),W-bw);t=min(max(0,t),H-bh);return (l,t,l+bw,t+bh)
def register(rgb_pil,e_th):
    W,H=rgb_pil.size;gl=rgb_pil.convert("L");best=(-9,crop_box(W,H,0.65,0,0))
    for fx in (-0.08,-0.04,0,0.04,0.08):
        for fy in (-0.08,-0.04,0,0.04,0.08):
            box=crop_box(W,H,0.65,fx,fy)
            e_rg=edges(np.array(gl.crop(box).resize((320,256)))/255.0)
            q=float((e_th*e_rg).mean()/(e_th.std()*e_rg.std()+1e-8))
            if q>best[0]: best=(q,box)
    return best[1],best[0]
def load_model(ck):
    d=torch.load(ck,map_location=dev);ar=d['args'];in_ch=3+(2 if ar.get('solar') else 0)+(1 if ar.get('use_depth') else 0)
    if ar.get('arch')=='physics':
        from train_a4 import PhysicsNet;m=PhysicsNet(ar.get('encoder','convnext_tiny'),in_ch,K=ar.get('K',6)).to(dev)
    else: m=UNetReg(ar.get('encoder','convnext_tiny'),in_ch,bool(ar.get('use_alpha'))).to(dev)
    m.load_state_dict(d['model']);m.eval();return m,ar

@torch.no_grad()
def main():
    split=json.load(open(f"{C.BASE}/code/train_test_split.json"))
    names=sorted([f for f in os.listdir(TH) if f.upper().endswith('.JPG')],key=lambda x:int(x.split('.')[0]))
    models={ck.split('/')[-2]:load_model(ck) for ck in sorted(glob.glob(f"{C.BASE}/rgb2thermal/checkpoints/*/best.pth"))}
    mods=[n for n in ORDER if n in models or n=='ensemble']
    try: import lpips; lpfn=lpips.LPIPS(net='alex').to(dev).eval()
    except Exception: lpfn=None
    from transformers import pipeline
    dpipe=pipeline("depth-estimation",model="depth-anything/Depth-Anything-V2-Small-hf",device=0 if dev=='cuda' else -1)
    def alpha_for(nm):
        try:
            dji=split[nm][0];tif=f"{C.BASE}/alphaearth-emb/satellite_embedding_{dji.replace('.JPG','.')}.tif"
            if os.path.exists(tif):
                import rasterio
                with rasterio.open(tif) as s:a=s.read()
                return np.nan_to_num(a,nan=0,posinf=0,neginf=0).mean(axis=(1,2)).astype(np.float32)
        except Exception: pass
        return np.zeros(64,np.float32)
    def run(rgb_reg,rgb_full,depth,alpha):
        out={}
        for nm,(m,ar) in models.items():
            rgb=rgb_full if ar.get('unreg') else rgb_reg
            t=((torch.from_numpy(rgb.transpose(2,0,1)).float()-IMN)/IST).unsqueeze(0)
            if ar.get('use_depth'): t=torch.cat([t,torch.from_numpy(depth[None,None]).float()],1)
            al=torch.from_numpy(np.nan_to_num(alpha)[None]).float().to(dev) if getattr(m,'use_alpha',False) else None
            out[nm]=np.nan_to_num(m(t.to(dev),al)[0,0].cpu().numpy()).clip(0,1)
        out['ensemble']=np.mean([out[k] for k in ENS],0); return out
    gallery_idx=set(np.linspace(0,len(names)-1,12).astype(int)); gstash={}
    acc={nm:[] for nm in mods}; regq=[]
    for i,nm in enumerate(names):
        th=Image.open(f"{TH}/{nm}").convert("RGB")
        gt=to_scalar(np.array(th.resize((640,512))))
        e_th=edges(to_scalar(np.array(th.resize((320,256)))))
        rgb_pil=Image.open(f"{RGB}/{nm}").convert("RGB")
        box,q=register(rgb_pil,e_th); regq.append(q)
        rgb_reg=np.array(rgb_pil.crop(box).resize((640,512))).astype(np.float32)/255
        rgb_full=np.array(rgb_pil.resize((640,512))).astype(np.float32)/255
        dd=dpipe(Image.fromarray((rgb_reg*255).astype(np.uint8)))["predicted_depth"].squeeze().detach().cpu().numpy().astype(np.float32)
        dd=np.array(Image.fromarray(((dd-dd.min())/(np.ptp(dd)+1e-8)*255).astype(np.uint8)).resize((640,512)))/255.0
        preds=run(rgb_reg,rgb_full,dd,alpha_for(nm))
        for k in mods:
            d=C.metrics_np(preds[k],gt); d['ssim']=C.ssim_np(preds[k],gt,dev)
            if lpfn is not None:
                pc=C.to_color(preds[k]).astype(np.float32)/127.5-1; gc=C.to_color(gt).astype(np.float32)/127.5-1
                d['clpips']=float(lpfn(torch.from_numpy(pc.transpose(2,0,1)[None]).to(dev),torch.from_numpy(gc.transpose(2,0,1)[None]).to(dev)).item())
            else: d['clpips']=float('nan')
            acc[k].append(d)
        if i in gallery_idx: gstash[nm]=(rgb_reg,gt,preds)
        if (i+1)%40==0: print(f"  {i+1}/{len(names)} regq~{np.mean(regq):.3f}",flush=True)
    rows=[]
    for k in mods:
        agg={kk:float(np.nanmean([x[kk] for x in acc[k]])) for kk in acc[k][0]}; agg['model']=k; rows.append(agg)
    rows=sorted(rows,key=lambda r:r['mae'])
    keys=['model','mae','rmse','psnr','ssim','corr','clpips']
    with open(f"{OUT}/leaderboard_official.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in rows: w.writerow({k:r.get(k) for k in keys})
    print(f"\n=== OFFICIAL TEST LEADERBOARD (202 unseen imgs, reg-quality mean {np.mean(regq):.3f}) ===")
    print(f"{'model':14s}{'MAE':>8}{'RMSE':>8}{'PSNR':>7}{'SSIM':>7}{'corr':>7}{'cLPIPS':>8}")
    for r in rows: print(f"{r['model']:14s}{r['mae']:8.4f}{r['rmse']:8.4f}{r['psnr']:7.2f}{r['ssim']:7.3f}{r['corr']:7.3f}{r['clpips']:8.3f}")
    # gallery
    gn=sorted(gstash.keys(),key=lambda x:int(x.split('.')[0])); cols=['RGB','GT']+mods
    fig,ax=plt.subplots(len(gn),len(cols),figsize=(2.0*len(cols),2.0*len(gn)));ax=np.atleast_2d(ax)
    for r,nm in enumerate(gn):
        rgb_reg,gt,preds=gstash[nm]
        ax[r,0].imshow((rgb_reg*255).astype(np.uint8));ax[r,0].axis('off');ax[r,0].set_ylabel(nm,fontsize=7)
        ax[r,1].imshow(gt,cmap='inferno',vmin=0,vmax=1);ax[r,1].axis('off')
        for c,k in enumerate(mods): ax[r,2+c].imshow(preds[k],cmap='inferno',vmin=0,vmax=1);ax[r,2+c].axis('off')
        if r==0:
            for c,t in enumerate(cols): ax[0,c].set_title(t,fontsize=8)
    plt.tight_layout();plt.savefig(f"{OUT}/test_official_withGT_gallery.png",dpi=95,bbox_inches='tight');plt.close()
    print("wrote leaderboard_official.csv + test_official_withGT_gallery.png")

if __name__=='__main__': main()
