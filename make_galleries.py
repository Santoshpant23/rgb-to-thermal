"""Side-by-side galleries on UNSEEN data, for all models + ensemble.
 (1) internal held-out test split (has GT): RGB | GT | models | ensemble
 (2) official Test_2 set (no local GT):     RGB | models | ensemble  (render as thermal)
"""
import os, glob, json
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import r2t_common as C
from train_a1 import UNetReg
dev='cuda' if torch.cuda.is_available() else 'cpu'
OUT=f"{C.BASE}/rgb2thermal/outputs"; TEST2=f"{C.BASE}/data/Test_2/RGB"
IMN=torch.tensor([0.485,0.456,0.406]).view(3,1,1); IST=torch.tensor([0.229,0.224,0.225]).view(3,1,1)
ORDER=['ensemble','a1_rgb','a2_gan','a4_physics8','a1_rgbda','a4_physics','a1_rgbd','a1_small','a1_unreg']
ENS=['a1_rgb','a2_gan','a4_physics8']

def load_model(ck):
    d=torch.load(ck,map_location=dev); ar=d['args']; in_ch=3+(1 if ar.get('use_depth') else 0)
    if ar.get('arch')=='physics':
        from train_a4 import PhysicsNet; m=PhysicsNet(ar.get('encoder','convnext_tiny'),in_ch,K=ar.get('K',6)).to(dev)
    else:
        m=UNetReg(ar.get('encoder','convnext_tiny'),in_ch,bool(ar.get('use_alpha'))).to(dev)
    m.load_state_dict(d['model']); m.eval(); return m,ar

def center_crop065(pil, c=0.65):
    W,H=pil.size; bw=c*W; bh=bw/(640/512)
    if bh>H: bh=H; bw=bh*(640/512)
    l=(W-bw)/2; t=(H-bh)/2; return pil.crop((l,t,l+bw,t+bh)).resize((640,512),Image.BILINEAR)

@torch.no_grad()
def main():
    sp=C.load_split()
    models={ck.split('/')[-2]:load_model(ck) for ck in sorted(glob.glob(f"{C.BASE}/rgb2thermal/checkpoints/*/best.pth"))}
    names=[n for n in ORDER if n in models or n=='ensemble']
    # depth pipe (for official set, on the fly)
    try:
        from transformers import pipeline
        dpipe=pipeline("depth-estimation",model="depth-anything/Depth-Anything-V2-Small-hf",device=0 if dev=='cuda' else -1)
    except Exception as e:
        print("depth pipe unavailable:",repr(e)[:80]); dpipe=None

    def run_models(rgb_reg_np, rgb_full_np, depth_np, alpha_vec):
        """return dict name->pred. rgb_reg_np/rgb_full_np in [0,1] HxWx3; depth HxW [0,1]."""
        out={}
        def to_in(rgb_np, use_depth):
            t=torch.from_numpy(rgb_np.transpose(2,0,1)).float()
            t=((t-IMN)/IST)
            x=t.unsqueeze(0)
            if use_depth: x=torch.cat([x, torch.from_numpy(depth_np[None,None]).float()],1)
            return x.to(dev)
        for nm,(m,ar) in models.items():
            rgb=rgb_full_np if ar.get('unreg') else rgb_reg_np
            x=to_in(rgb, ar.get('use_depth',0))
            av=np.nan_to_num(alpha_vec, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            al=torch.from_numpy(av[None]).float().to(dev) if getattr(m,'use_alpha',False) else None
            out[nm]=np.nan_to_num(m(x,al)[0,0].cpu().numpy(), nan=0.0, posinf=1.0, neginf=0.0).clip(0,1)
        out['ensemble']=np.mean([out[k] for k in ENS],axis=0)
        return out

    # ---------- (1) internal held-out test (GT available) ----------
    test=sp['test'][:12]
    cols=['RGB','GT']+names
    fig,ax=plt.subplots(len(test),len(cols),figsize=(2.0*len(cols),2.0*len(test))); ax=np.atleast_2d(ax)
    for r,fn in enumerate(test):
        n=fn.replace('.JPG','')
        rgb_reg=np.asarray(Image.open(f"{C.CACHE}/reg_rgb/{n}.png").convert("RGB")).astype(np.float32)/255
        rgb_full=np.asarray(Image.open(f"{C.BASE}/data/Train_2/RGB/{n}.JPG").convert("RGB").resize((640,512))).astype(np.float32)/255
        dp_p=f"{C.CACHE}/depth/{n}.npy"; depth=np.load(dp_p).astype(np.float32) if os.path.exists(dp_p) else np.zeros((512,640),np.float32)
        al_p=f"{C.CACHE}/alpha/{n}.npy"; alpha=np.load(al_p).astype(np.float32).reshape(64,-1).mean(1) if os.path.exists(al_p) else np.zeros(64,np.float32)
        gt=np.load(f"{C.CACHE}/scalar/{n}.npy").astype(np.float32)
        preds=run_models(rgb_reg,rgb_full,depth,alpha)
        ax[r,0].imshow((rgb_reg*255).astype(np.uint8)); ax[r,0].axis('off')
        ax[r,1].imshow(gt,cmap='inferno',vmin=0,vmax=1); ax[r,1].axis('off')
        for c,nm in enumerate(names): ax[r,2+c].imshow(preds[nm],cmap='inferno',vmin=0,vmax=1); ax[r,2+c].axis('off')
        if r==0:
            for c,t in enumerate(cols): ax[0,c].set_title(t,fontsize=8)
        ax[r,0].set_ylabel(n,fontsize=7)
    plt.tight_layout(); plt.savefig(f"{OUT}/test_internal_gallery.png",dpi=95,bbox_inches='tight'); plt.close()
    print("wrote test_internal_gallery.png  (",len(test),"unseen held-out images, with GT)")

    # ---------- (2) official Test_2 (no local GT) ----------
    t2=sorted([f for f in os.listdir(TEST2) if f.upper().endswith('.JPG')], key=lambda x:int(x.split('.')[0]))
    pick=[t2[i] for i in np.linspace(0,len(t2)-1,12).astype(int)]
    cols2=['RGB']+names
    fig,ax=plt.subplots(len(pick),len(cols2),figsize=(2.0*len(cols2),2.0*len(pick))); ax=np.atleast_2d(ax)
    try: split=json.load(open(f"{C.BASE}/code/train_test_split.json"))
    except Exception: split={}
    for r,fn in enumerate(pick):
        pil=Image.open(f"{TEST2}/{fn}").convert("RGB")
        reg=center_crop065(pil); rgb_reg=np.asarray(reg).astype(np.float32)/255
        rgb_full=np.asarray(pil.resize((640,512))).astype(np.float32)/255
        if dpipe is not None:
            dd=dpipe(reg)["predicted_depth"].squeeze().detach().cpu().numpy().astype(np.float32)
            dd=np.array(Image.fromarray(((dd-dd.min())/(np.ptp(dd)+1e-8)*255).astype(np.uint8)).resize((640,512)))/255.0
        else: dd=np.zeros((512,640),np.float32)
        alpha=np.zeros(64,np.float32)
        try:
            dji=split[fn][0]; tif=f"{C.BASE}/alphaearth-emb/satellite_embedding_{dji.replace('.JPG','.')}.tif"
            if os.path.exists(tif):
                import rasterio
                with rasterio.open(tif) as s: a=s.read()
                alpha=np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0).mean(axis=(1,2)).astype(np.float32)
        except Exception: pass
        preds=run_models(rgb_reg,rgb_full,dd,alpha)
        ax[r,0].imshow((rgb_reg*255).astype(np.uint8)); ax[r,0].axis('off'); ax[r,0].set_ylabel(fn,fontsize=7)
        for c,nm in enumerate(names): ax[r,1+c].imshow(preds[nm],cmap='inferno',vmin=0,vmax=1); ax[r,1+c].axis('off')
        if r==0:
            for c,t in enumerate(cols2): ax[0,c].set_title(t,fontsize=8)
    plt.tight_layout(); plt.savefig(f"{OUT}/test_official_gallery.png",dpi=95,bbox_inches='tight'); plt.close()
    print("wrote test_official_gallery.png  (",len(pick),"official Test_2 images, RGB cropped to thermal FOV)")

if __name__=='__main__': main()
