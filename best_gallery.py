"""Best-model (winning ensemble = a1_rgb+a2_gan+a4_physics8, TTA) vs ground truth.
5 TRAIN photos (model saw these) + 5 official-TEST photos (never seen). Cols: RGB | GT | Best pred."""
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
def ens(rgb_reg,rgb_full,depth,alpha,sol2):
    ps=[]
    for k in ENS:
        m,ar=models[k]; rgb=rgb_full if ar.get('unreg') else rgb_reg
        ps.append(tta(m,ar,rgb,depth,alpha,sol2))
    return np.mean(ps,0)

@torch.no_grad()
def main():
    split=json.load(open(f"{C.BASE}/code/train_test_split.json")); sp=C.load_split()
    from transformers import pipeline
    dpipe=pipeline("depth-estimation",model="depth-anything/Depth-Anything-V2-Small-hf",device=0 if dev=='cuda' else -1)
    # pick 5 train + 5 official-test, spread
    train=[sp['train'][i] for i in np.linspace(0,len(sp['train'])-1,5).astype(int)]
    tnames=sorted([f for f in os.listdir(EO.TH) if f.upper().endswith('.JPG')],key=lambda x:int(x.split('.')[0]))
    test=[tnames[i] for i in np.linspace(0,len(tnames)-1,5).astype(int)]
    rows=[("TRAIN",t) for t in train]+[("TEST",t) for t in test]
    fig,ax=plt.subplots(len(rows),3,figsize=(9,3*len(rows))); ax=np.atleast_2d(ax)
    for r,(tag,fn) in enumerate(rows):
        n=fn.replace('.JPG','')
        if tag=="TRAIN":
            rgb_reg=np.asarray(Image.open(f"{C.CACHE}/reg_rgb/{n}.png").convert("RGB")).astype(np.float32)/255
            rgb_full=np.asarray(Image.open(f"{C.BASE}/data/Train_2/RGB/{n}.JPG").convert("RGB").resize((640,512))).astype(np.float32)/255
            depth=np.load(f"{C.CACHE}/depth/{n}.npy").astype(np.float32) if os.path.exists(f"{C.CACHE}/depth/{n}.npy") else np.zeros((512,640),np.float32)
            ap=f"{C.CACHE}/alpha/{n}.npy"; alpha=np.load(ap).astype(np.float32).reshape(64,-1).mean(1) if os.path.exists(ap) else np.zeros(64,np.float32)
            gt=np.load(f"{C.CACHE}/scalar/{n}.npy").astype(np.float32)
        else:
            th=Image.open(f"{EO.TH}/{fn}").convert("RGB"); gt=EO.to_scalar(np.array(th.resize((640,512))))
            e_th=EO.edges(EO.to_scalar(np.array(th.resize((320,256)))))
            rgb_pil=Image.open(f"{EO.RGB}/{fn}").convert("RGB"); box,_=EO.register(rgb_pil,e_th)
            rgb_reg=np.array(rgb_pil.crop(box).resize((640,512))).astype(np.float32)/255
            rgb_full=np.array(rgb_pil.resize((640,512))).astype(np.float32)/255
            dd=dpipe(Image.fromarray((rgb_reg*255).astype(np.uint8)))["predicted_depth"].squeeze().detach().cpu().numpy().astype(np.float32)
            depth=np.array(Image.fromarray(((dd-dd.min())/(np.ptp(dd)+1e-8)*255).astype(np.uint8)).resize((640,512)))/255.0
            alpha=alpha_for(split,fn)
        pred=ens(rgb_reg,rgb_full,depth,alpha,SOLAR.get(n))
        m=C.metrics_np(pred,gt); psnr=m['psnr']
        ax[r,0].imshow((rgb_reg*255).astype(np.uint8)); ax[r,0].axis('off')
        ax[r,0].set_ylabel(f"{tag}\n{fn}",fontsize=9,rotation=0,labelpad=35,va='center')
        ax[r,1].imshow(gt,cmap='inferno',vmin=0,vmax=1); ax[r,1].axis('off')
        ax[r,2].imshow(pred,cmap='inferno',vmin=0,vmax=1); ax[r,2].axis('off')
        ax[r,2].set_title(f"PSNR {psnr:.1f} dB",fontsize=8)
        if r==0:
            ax[0,0].set_title("RGB input",fontsize=11); ax[0,1].set_title("Ground truth",fontsize=11); ax[0,2].set_title("Best model (ensemble)",fontsize=11)
    fig.suptitle("Best model (ensemble: a1_rgb+a2_gan+a4_physics8 + TTA)  —  top 5 = TRAIN (seen),  bottom 5 = TEST (unseen)",fontsize=12,y=1.005)
    plt.tight_layout(); plt.savefig(f"{C.BASE}/rgb2thermal_v2/outputs/best_vs_gt.png",dpi=110,bbox_inches='tight'); plt.close()
    print("wrote rgb2thermal_v2/outputs/best_vs_gt.png")

if __name__=='__main__': main()
