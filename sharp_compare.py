"""Ensemble (all combined) vs a2_gan (single best): RGB | GT | Ensemble | a2_gan, on unseen test.
Plus a zoomed crop row to reveal sharpness."""
import os, json, numpy as np, torch
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import r2t_common as C, eval_official as EO
from eval_v2 import tta, alpha_for
dev=EO.dev
ENS=['a1_rgb','a2_gan','a4_physics8']
models={k:EO.load_model(f"{C.BASE}/rgb2thermal/checkpoints/{k}/best.pth") for k in ENS}
SOLAR=json.load(open(f"{C.CACHE}/solar.json")) if os.path.exists(f"{C.CACHE}/solar.json") else {}

@torch.no_grad()
def main():
    split=json.load(open(f"{C.BASE}/code/train_test_split.json"))
    from transformers import pipeline
    dpipe=pipeline("depth-estimation",model="depth-anything/Depth-Anything-V2-Small-hf",device=0 if dev=='cuda' else -1)
    tnames=sorted([f for f in os.listdir(EO.TH) if f.upper().endswith('.JPG')],key=lambda x:int(x.split('.')[0]))
    test=[tnames[i] for i in np.linspace(0,len(tnames)-1,4).astype(int)]
    cols=['RGB','Ground truth','Ensemble (combined)','a2_gan (single best)']
    fig,ax=plt.subplots(len(test),4,figsize=(13,3.2*len(test))); ax=np.atleast_2d(ax)
    sharp_e=[]; sharp_g=[]
    for r,fn in enumerate(test):
        th=Image.open(f"{EO.TH}/{fn}").convert("RGB"); gt=EO.to_scalar(np.array(th.resize((640,512))))
        e_th=EO.edges(EO.to_scalar(np.array(th.resize((320,256)))))
        rgb_pil=Image.open(f"{EO.RGB}/{fn}").convert("RGB"); box,_=EO.register(rgb_pil,e_th)
        rgb_reg=np.array(rgb_pil.crop(box).resize((640,512))).astype(np.float32)/255
        rgb_full=np.array(rgb_pil.resize((640,512))).astype(np.float32)/255
        dd=dpipe(Image.fromarray((rgb_reg*255).astype(np.uint8)))["predicted_depth"].squeeze().detach().cpu().numpy().astype(np.float32)
        depth=np.array(Image.fromarray(((dd-dd.min())/(np.ptp(dd)+1e-8)*255).astype(np.uint8)).resize((640,512)))/255.0
        alpha=alpha_for(split,fn); sol2=SOLAR.get(fn.replace('.JPG',''))
        ps={}
        for k in ENS:
            m,ar=models[k]; rgb=rgb_full if ar.get('unreg') else rgb_reg
            ps[k]=tta(m,ar,rgb,depth,alpha,sol2)
        ens=np.mean([ps[k] for k in ENS],0); gan=ps['a2_gan']
        # sharpness proxy = mean gradient magnitude (higher = sharper)
        def sg(x): gy,gx=np.gradient(x); return float(np.hypot(gx,gy).mean())
        sharp_e.append(sg(ens)); sharp_g.append(sg(gan))
        ax[r,0].imshow((rgb_reg*255).astype(np.uint8)); ax[r,0].axis('off'); ax[r,0].set_ylabel(fn,fontsize=8)
        ax[r,1].imshow(gt,cmap='inferno',vmin=0,vmax=1); ax[r,1].axis('off')
        ax[r,2].imshow(ens,cmap='inferno',vmin=0,vmax=1); ax[r,2].axis('off'); ax[r,2].set_title(f"grad {sg(ens):.3f}",fontsize=7)
        ax[r,3].imshow(gan,cmap='inferno',vmin=0,vmax=1); ax[r,3].axis('off'); ax[r,3].set_title(f"grad {sg(gan):.3f}",fontsize=7)
        if r==0:
            for c,t in enumerate(cols): ax[0,c].set_title((cols[c]+("" if c<2 else f"  (grad {[sg(ens),sg(gan)][c-2]:.3f})")),fontsize=10)
    fig.suptitle("Combined ensemble vs single best (a2_gan) — unseen test.  'grad' = sharpness proxy (higher=sharper)",fontsize=11,y=1.003)
    plt.tight_layout(); plt.savefig(f"{C.BASE}/rgb2thermal_v2/outputs/sharp_compare.png",dpi=120,bbox_inches='tight'); plt.close()
    print(f"mean sharpness  ensemble={np.mean(sharp_e):.4f}  a2_gan={np.mean(sharp_g):.4f}")
    print("wrote sharp_compare.png")

if __name__=='__main__': main()
