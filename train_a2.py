"""A2 — conditional GAN (pix2pix-style). Generator=UNetReg, PatchGAN discriminator.
Loss = LSGAN adv + L1 + LPIPS perceptual. Predicts scalar thermal."""
import os, json, time, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import r2t_common as C
from train_a1 import UNetReg, resize_batch, run_eval

class PatchD(nn.Module):
    def __init__(s,in_ch):
        super().__init__()
        def blk(i,o,st=2,norm=True):
            l=[nn.Conv2d(i,o,4,st,1)]
            if norm: l.append(nn.InstanceNorm2d(o))
            l.append(nn.LeakyReLU(0.2,True)); return l
        s.m=nn.Sequential(*blk(in_ch,64,norm=False),*blk(64,128),*blk(128,256),
                          *blk(256,512,st=1),nn.Conv2d(512,1,4,1,1))
    def forward(s,x): return s.m(x)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--name',required=True); ap.add_argument('--encoder',default='convnext_tiny')
    ap.add_argument('--use_depth',type=int,default=0); ap.add_argument('--use_alpha',type=int,default=0)
    ap.add_argument('--res',type=int,default=512); ap.add_argument('--epochs',type=int,default=100)
    ap.add_argument('--bs',type=int,default=5); ap.add_argument('--lr',type=float,default=2e-4)
    ap.add_argument('--l1',type=float,default=50.0); ap.add_argument('--lp',type=float,default=1.0)
    a=ap.parse_args(); dev='cuda' if torch.cuda.is_available() else 'cpu'
    sp=C.load_split(); in_ch=3+(1 if a.use_depth else 0)
    tr=DataLoader(C.R2TDataset(sp['train'],augment=True,use_depth=a.use_depth),batch_size=a.bs,shuffle=True,num_workers=4,drop_last=True,pin_memory=True)
    va=DataLoader(C.R2TDataset(sp['val'],augment=False,use_depth=a.use_depth),batch_size=a.bs,shuffle=False,num_workers=2)
    G=UNetReg(a.encoder,in_ch,bool(a.use_alpha)).to(dev)
    D=PatchD(in_ch+1).to(dev)
    try:
        import lpips; lpfn=lpips.LPIPS(net='alex').to(dev).eval()
        for p in lpfn.parameters(): p.requires_grad=False
    except Exception as e:
        print("lpips unavailable:",repr(e)[:80]); lpfn=None
    optG=torch.optim.AdamW(G.parameters(),lr=a.lr,betas=(0.5,0.999),weight_decay=1e-4)
    optD=torch.optim.AdamW(D.parameters(),lr=a.lr,betas=(0.5,0.999))
    schG=torch.optim.lr_scheduler.CosineAnnealingLR(optG,T_max=a.epochs)
    out=f"{C.BASE}/rgb2thermal/checkpoints/{a.name}"; os.makedirs(out,exist_ok=True)
    print(f"[{a.name}] GAN enc={a.encoder} in_ch={in_ch} alpha={a.use_alpha} res={a.res}",flush=True)
    best=1e9; hist=[]; t0=time.time()
    def lp_loss(p,y):
        if lpfn is None: return torch.tensor(0.0,device=dev)
        return lpfn(p.repeat(1,3,1,1)*2-1, y.repeat(1,3,1,1)*2-1).mean()
    for ep in range(1,a.epochs+1):
        G.train();D.train(); gl=dl=0; nb=0
        for b in tr:
            b=resize_batch(b,a.res); x=C.make_input(b,a.use_depth).to(dev); y=b['target'].to(dev)
            al=b['alpha'].to(dev) if G.use_alpha else None
            fake=G(x,al)
            # D step
            optD.zero_grad()
            dr=D(torch.cat([x,y],1)); df=D(torch.cat([x,fake.detach()],1))
            ld=0.5*(F.mse_loss(dr,torch.ones_like(dr))+F.mse_loss(df,torch.zeros_like(df)))
            ld.backward(); optD.step()
            # G step
            optG.zero_grad()
            df2=D(torch.cat([x,fake],1))
            lg=F.mse_loss(df2,torch.ones_like(df2))+a.l1*F.l1_loss(fake,y)+a.lp*lp_loss(fake,y)+0.5*(1-C.ssim(fake,y))
            lg.backward(); optG.step()
            gl+=lg.item(); dl+=ld.item(); nb+=1
        schG.step()
        if ep%5==0 or ep==1 or ep==a.epochs:
            m=run_eval(G,va,dev,a.use_depth,a.res); hist.append(dict(epoch=ep,g=gl/nb,d=dl/nb,**m))
            print(f"  ep{ep:3d} G={gl/nb:.3f} D={dl/nb:.3f} val_mae={m['mae']:.4f} psnr={m['psnr']:.2f} ssim={m['ssim']:.3f} t={time.time()-t0:.0f}s",flush=True)
            if m['mae']<best:
                best=m['mae']; torch.save(dict(model=G.state_dict(),args=vars(a),val=m),f"{out}/best.pth")
    json.dump(dict(history=hist,best_mae=best,args=vars(a)),open(f"{out}/metrics.json","w"))
    print(f"[{a.name}] DONE best_val_mae={best:.4f}",flush=True)

if __name__=='__main__': main()
