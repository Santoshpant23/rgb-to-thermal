"""A1 — foundation dense regression: timm pretrained encoder + U-Net decoder -> scalar thermal."""
import os, json, time, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
import timm
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import r2t_common as C

class ConvBlock(nn.Module):
    def __init__(s,i,o):
        super().__init__(); s.b=nn.Sequential(
            nn.Conv2d(i,o,3,padding=1),nn.BatchNorm2d(o),nn.GELU(),
            nn.Conv2d(o,o,3,padding=1),nn.BatchNorm2d(o),nn.GELU())
    def forward(s,x): return s.b(x)

class FiLM(nn.Module):
    def __init__(s,d,ch):
        super().__init__(); s.n=nn.Sequential(nn.Linear(d,128),nn.GELU(),nn.Linear(128,ch*2)); s.ch=ch
    def forward(s,x,v):
        g,b=s.n(v).chunk(2,1); return (1+g[...,None,None])*x+b[...,None,None]

class UNetReg(nn.Module):
    def __init__(s, encoder='convnext_tiny', in_ch=3, use_alpha=False):
        super().__init__()
        s.enc=timm.create_model(encoder,pretrained=True,features_only=True,in_chans=in_ch)
        chs=s.enc.feature_info.channels()   # e.g. [96,192,384,768]
        s.use_alpha=use_alpha
        if use_alpha: s.film=FiLM(64,chs[-1])
        s.up3=ConvBlock(chs[3]+chs[2],chs[2])
        s.up2=ConvBlock(chs[2]+chs[1],chs[1])
        s.up1=ConvBlock(chs[1]+chs[0],chs[0])
        s.u0 =ConvBlock(chs[0],64)
        s.u_1=ConvBlock(64,32)
        s.head=nn.Conv2d(32,1,1)
    def _up(s,x,ref): return F.interpolate(x,size=ref.shape[2:],mode='bilinear',align_corners=False)
    def _up2(s,x): return F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False)
    def forward(s,x,alpha=None):
        f1,f2,f3,f4=s.enc(x)           # strides 4,8,16,32
        if s.use_alpha and alpha is not None: f4=s.film(f4,alpha)
        d3=s.up3(torch.cat([s._up(f4,f3),f3],1))
        d2=s.up2(torch.cat([s._up(d3,f2),f2],1))
        d1=s.up1(torch.cat([s._up(d2,f1),f1],1))
        u0=s.u0(s._up2(d1))            # /2
        u1=s.u_1(s._up2(u0))           # /1
        return torch.sigmoid(s.head(u1))

def resize_batch(b, res):
    if res>=C.RES_H: return b
    h=res; w=int(round(res*C.RES_W/C.RES_H/2)*2)
    for k in ['rgb','depth','target']:
        b[k]=F.interpolate(b[k],size=(h,w),mode='bilinear',align_corners=False)
    return b

def run_eval(model,loader,dev,use_depth,res):
    model.eval(); ms=[]; ss=[]
    with torch.no_grad():
        for b in loader:
            b=resize_batch(b,res); x=C.make_input(b,use_depth).to(dev)
            al=b['alpha'].to(dev) if model.use_alpha else None
            p=model(x,al).cpu().numpy(); t=b['target'].numpy()
            for i in range(len(p)):
                m=C.metrics_np(p[i,0],t[i,0]); m['ssim']=C.ssim_np(p[i,0],t[i,0])
                ms.append(m)
    keys=ms[0].keys(); return {k:float(np.mean([m[k] for m in ms])) for k in keys}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--name',required=True); ap.add_argument('--encoder',default='convnext_tiny')
    ap.add_argument('--use_depth',type=int,default=0); ap.add_argument('--use_alpha',type=int,default=0)
    ap.add_argument('--res',type=int,default=512); ap.add_argument('--epochs',type=int,default=80)
    ap.add_argument('--bs',type=int,default=6); ap.add_argument('--lr',type=float,default=3e-4)
    ap.add_argument('--unreg',type=int,default=0); ap.add_argument('--solar',type=int,default=0)
    a=ap.parse_args()
    dev='cuda' if torch.cuda.is_available() else 'cpu'
    sp=C.load_split()
    in_ch=3+(2 if a.solar else 0)+(1 if a.use_depth else 0)
    tr=DataLoader(C.R2TDataset(sp['train'],augment=True,use_depth=a.use_depth,unreg=bool(a.unreg),solar=bool(a.solar)),batch_size=a.bs,shuffle=True,num_workers=4,drop_last=True,pin_memory=True)
    va=DataLoader(C.R2TDataset(sp['val'],augment=False,use_depth=a.use_depth,unreg=bool(a.unreg),solar=bool(a.solar)),batch_size=a.bs,shuffle=False,num_workers=2)
    model=UNetReg(a.encoder,in_ch,bool(a.use_alpha)).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs)
    out=f"{C.BASE}/rgb2thermal/checkpoints/{a.name}"; os.makedirs(out,exist_ok=True)
    nparams=sum(p.numel() for p in model.parameters())/1e6
    print(f"[{a.name}] enc={a.encoder} in_ch={in_ch} alpha={a.use_alpha} res={a.res} params={nparams:.1f}M",flush=True)
    best=1e9; hist=[]; t0=time.time()
    for ep in range(1,a.epochs+1):
        model.train(); tl=0; nb=0
        for b in tr:
            b=resize_batch(b,a.res); x=C.make_input(b,a.use_depth).to(dev); y=b['target'].to(dev)
            al=b['alpha'].to(dev) if model.use_alpha else None
            p=model(x,al); loss=C.combined_loss(p,y)
            opt.zero_grad(); loss.backward(); opt.step(); tl+=loss.item(); nb+=1
        sched.step()
        if ep%5==0 or ep==1 or ep==a.epochs:
            m=run_eval(model,va,dev,a.use_depth,a.res)
            hist.append(dict(epoch=ep,train_loss=tl/nb,**m))
            print(f"  ep{ep:3d} loss={tl/nb:.4f} val_mae={m['mae']:.4f} psnr={m['psnr']:.2f} ssim={m['ssim']:.3f} corr={m['corr']:.3f} t={time.time()-t0:.0f}s",flush=True)
            if m['mae']<best:
                best=m['mae']; torch.save(dict(model=model.state_dict(),args=vars(a),val=m),f"{out}/best.pth")
    json.dump(dict(history=hist,best_mae=best,args=vars(a)),open(f"{out}/metrics.json","w"))
    # sample gallery on a few val
    model.load_state_dict(torch.load(f"{out}/best.pth",map_location=dev)['model']); model.eval()
    names=sp['val'][:4]; ds=C.R2TDataset(names,augment=False,use_depth=a.use_depth,unreg=bool(a.unreg),solar=bool(a.solar))
    fig,ax=plt.subplots(len(names),3,figsize=(12,4*len(names))); ax=np.atleast_2d(ax)
    with torch.no_grad():
        for r,n in enumerate(names):
            b={k:(v.unsqueeze(0) if torch.is_tensor(v) else v) for k,v in ds[r].items()}
            b=resize_batch(b,a.res); x=C.make_input(b,a.use_depth).to(dev)
            al=b['alpha'].to(dev) if model.use_alpha else None
            p=model(x,al)[0,0].cpu().numpy(); t=b['target'][0,0].numpy()
            rgb=np.asarray(Image.open(f"{C.CACHE}/reg_rgb/{ds.names[r]}.png"))
            ax[r,0].imshow(rgb);ax[r,0].set_title(ds.names[r]);ax[r,0].axis('off')
            ax[r,1].imshow(t,cmap='inferno',vmin=0,vmax=1);ax[r,1].set_title('GT');ax[r,1].axis('off')
            ax[r,2].imshow(p,cmap='inferno',vmin=0,vmax=1);ax[r,2].set_title('pred');ax[r,2].axis('off')
    plt.tight_layout();plt.savefig(f"{out}/sample.png",dpi=80,bbox_inches='tight');plt.close()
    print(f"[{a.name}] DONE best_val_mae={best:.4f}",flush=True)

if __name__=='__main__': main()
