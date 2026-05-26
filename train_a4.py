"""A4 — physics-structured decomposition. Predict K soft material masks + illumination;
compose temperature = sum_k mask_k*base_k - mask_k*shadowgain_k*(1-illum) + small residual.
Interpretable (material signatures + shadow), still fits detail via residual."""
import os, json, time, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import timm
import r2t_common as C
from train_a1 import ConvBlock, resize_batch, run_eval

class PhysicsNet(nn.Module):
    def __init__(self, encoder='convnext_tiny', in_ch=3, K=6, use_alpha=False):
        super().__init__(); self.use_alpha=False; self.K=K
        self.enc=timm.create_model(encoder,pretrained=True,features_only=True,in_chans=in_ch)
        chs=self.enc.feature_info.channels()
        self.up3=ConvBlock(chs[3]+chs[2],chs[2]); self.up2=ConvBlock(chs[2]+chs[1],chs[1])
        self.up1=ConvBlock(chs[1]+chs[0],chs[0]); self.u0=ConvBlock(chs[0],64); self.u1=ConvBlock(64,32)
        self.mask_head=nn.Conv2d(32,K,1); self.illum_head=nn.Conv2d(32,1,1); self.res_head=nn.Conv2d(32,1,1)
        self.base=nn.Parameter(torch.linspace(0.3,0.8,K))      # per-material temperature signature
        self.sgain=nn.Parameter(torch.ones(K)*0.2)             # per-material shadow cooling
    def _up(self,x,r): return F.interpolate(x,size=r.shape[2:],mode='bilinear',align_corners=False)
    def _u2(self,x): return F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False)
    def forward(self,x,alpha=None,aux=False):
        f1,f2,f3,f4=self.enc(x)
        d3=self.up3(torch.cat([self._up(f4,f3),f3],1)); d2=self.up2(torch.cat([self._up(d3,f2),f2],1))
        d1=self.up1(torch.cat([self._up(d2,f1),f1],1))   # /4
        u0=self.u0(d1)                                    # /4, 64ch
        u1=self.u1(self._u2(u0))                          # /2, 32ch
        feat=self._u2(u1)                                 # /1, 32ch
        M=F.softmax(self.mask_head(feat),1)                    # [B,K,H,W]
        illum=torch.sigmoid(self.illum_head(feat))             # [B,1,H,W] 1=lit
        res=torch.tanh(self.res_head(feat))*0.15
        base=(M*self.base.view(1,-1,1,1)).sum(1,keepdim=True)
        cool=(M*self.sgain.view(1,-1,1,1)).sum(1,keepdim=True)*(1-illum)
        out=torch.clamp(base-cool+res,0,1)
        if aux: return out,M,illum
        return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--name',default='a4_physics'); ap.add_argument('--encoder',default='convnext_tiny')
    ap.add_argument('--use_depth',type=int,default=1); ap.add_argument('--K',type=int,default=6)
    ap.add_argument('--res',type=int,default=512); ap.add_argument('--epochs',type=int,default=80)
    ap.add_argument('--bs',type=int,default=6); ap.add_argument('--lr',type=float,default=3e-4)
    a=ap.parse_args(); dev='cuda' if torch.cuda.is_available() else 'cpu'
    sp=C.load_split(); in_ch=3+(1 if a.use_depth else 0)
    tr=DataLoader(C.R2TDataset(sp['train'],augment=True,use_depth=a.use_depth),batch_size=a.bs,shuffle=True,num_workers=4,drop_last=True,pin_memory=True)
    va=DataLoader(C.R2TDataset(sp['val'],augment=False,use_depth=a.use_depth),batch_size=a.bs,shuffle=False,num_workers=2)
    model=PhysicsNet(a.encoder,in_ch,a.K).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs)
    out=f"{C.BASE}/rgb2thermal/checkpoints/{a.name}"; os.makedirs(out,exist_ok=True)
    print(f"[{a.name}] physics K={a.K} in_ch={in_ch}",flush=True); best=1e9; hist=[]; t0=time.time()
    for ep in range(1,a.epochs+1):
        model.train(); tl=0; nb=0
        for b in tr:
            b=resize_batch(b,a.res); x=C.make_input(b,a.use_depth).to(dev); y=b['target'].to(dev)
            p=model(x); loss=C.combined_loss(p,y)
            opt.zero_grad(); loss.backward(); opt.step(); tl+=loss.item(); nb+=1
        sched.step()
        if ep%5==0 or ep==1 or ep==a.epochs:
            m=run_eval(model,va,dev,a.use_depth,a.res); hist.append(dict(epoch=ep,train_loss=tl/nb,**m))
            print(f"  ep{ep:3d} loss={tl/nb:.4f} val_mae={m['mae']:.4f} psnr={m['psnr']:.2f} ssim={m['ssim']:.3f} t={time.time()-t0:.0f}s",flush=True)
            if m['mae']<best: best=m['mae']; torch.save(dict(model=model.state_dict(),args={**vars(a),'arch':'physics'},val=m),f"{out}/best.pth")
    json.dump(dict(history=hist,best_mae=best,args={**vars(a),'arch':'physics'}),open(f"{out}/metrics.json","w"))
    # interpretability viz: material masks + illumination for 3 val
    model.load_state_dict(torch.load(f"{out}/best.pth",map_location=dev)['model']); model.eval()
    ds=C.R2TDataset(sp['val'][:3],augment=False,use_depth=a.use_depth)
    fig,ax=plt.subplots(3,4,figsize=(16,12)); ax=np.atleast_2d(ax)
    with torch.no_grad():
        for r in range(3):
            bb={k:(v.unsqueeze(0) if torch.is_tensor(v) else v) for k,v in ds[r].items()}
            bb=resize_batch(bb,a.res); x=C.make_input(bb,a.use_depth).to(dev)
            p,M,il=model(x,aux=True); p=p[0,0].cpu().numpy(); il=il[0,0].cpu().numpy()
            matmap=M[0].argmax(0).cpu().numpy()
            rgb=np.asarray(Image.open(f"{C.CACHE}/reg_rgb/{ds.names[r]}.png"))
            ax[r,0].imshow(rgb);ax[r,0].axis('off');ax[r,0].set_title(ds.names[r])
            ax[r,1].imshow(bb['target'][0,0].numpy(),cmap='inferno',vmin=0,vmax=1);ax[r,1].axis('off');ax[r,1].set_title('GT')
            ax[r,2].imshow(p,cmap='inferno',vmin=0,vmax=1);ax[r,2].axis('off');ax[r,2].set_title('pred')
            ax[r,3].imshow(matmap,cmap='tab10');ax[r,3].axis('off');ax[r,3].set_title('material map')
    plt.tight_layout();plt.savefig(f"{out}/sample.png",dpi=80,bbox_inches='tight');plt.close()
    print(f"[{a.name}] DONE best_val_mae={best:.4f} bases={model.base.detach().cpu().numpy().round(3)}",flush=True)

if __name__=='__main__': main()
