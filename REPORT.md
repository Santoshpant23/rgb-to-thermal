# RGB → Thermal, From Scratch — Findings & Results
*UMich Heat-Resilience follow-up. Autonomous run on the Knox A6000, 2026-05-24.*

## TL;DR
I rebuilt the RGB→thermal task from the data up. **Two data-level fixes the hackathon
pipeline missed mattered more than any architecture choice**, and on top of them I trained
**three independent model families** + ablations + an ensemble. Best model (ensemble) reaches
**PSNR 18.9 / SSIM 0.71 / corr 0.90** on a held-out test split — versus a mean-field floor of
10.7 and the hackathon's ~13.7 (which used the wrong target on misaligned pairs).

### Leaderboard (test split, 41 images, scalar heat field)
| model | MAE↓ | RMSE↓ | PSNR↑ | SSIM↑ | corr↑ | color-LPIPS↓ |
|---|---|---|---|---|---|---|
| **ensemble (a1_rgb + a2_gan + a4_physics8)** | **0.0730** | — | **18.90** | **0.705** | **0.902** | 0.355 |
| a1_rgb (ConvNeXt-T + U-Net, RGB only) | 0.0763 | — | 18.37 | 0.699 | 0.887 | 0.370 |
| a2_gan (pix2pix cGAN) | 0.0777 | — | 18.32 | 0.678 | 0.888 | **0.307** |
| a4_physics8 (physics, K=8 materials) | 0.0788 | — | 18.03 | 0.693 | 0.882 | 0.393 |
| a1_rgbda (RGB+depth+AlphaEarth) | 0.0800 | — | 17.94 | 0.693 | 0.879 | 0.394 |
| a4_physics (physics, K=6) | 0.0831 | — | 17.44 | 0.682 | 0.866 | 0.432 |
| a1_rgbd (RGB+depth) | 0.0837 | — | 17.34 | 0.682 | 0.861 | 0.435 |
| a1_small (ConvNeXt-S) | 0.0841 | — | 17.35 | 0.682 | 0.864 | 0.445 |
| a1_unreg (**no registration**, same target) | 0.0892 | — | 16.95 | 0.674 | 0.845 | 0.494 |
| [mean-field floor] | 0.2573 | — | 10.73 | 0.527 | 0.018 | 0.585 |

## The two discoveries (headline)
**1. We were optimizing the wrong target.** The thermal GTs are a single colormapped palette
(DJI-style), not raw temperature. A learned 1-D LUT inverts them to a **scalar heat field** with
reconstruction residual **~5/255** (matplotlib inferno gives ~42; PCA shows the colors lie on one
curve). The hackathon trained on the **red channel**, which is *not monotonic* in temperature, so
it was fitting a distorted objective. Evidence of impact: even the weakest real model here (16.95
dB on the correct scalar target) beats the hackathon's red-channel result, and the floor is 10.7.

**2. RGB and thermal were never spatially aligned.** RGB is a 12 MP wide camera (4000×3000, 4:3);
thermal is 640×512 (5:4) from a narrower-FOV sensor. At native framing edge-correlation ≈ 0.
A fixed **central ~0.65×width crop** of the RGB registers them (mean edge-corr 0.01 → 0.17,
consistent across images). The hackathon resized both to 256 → trained pixel losses on misaligned
pairs. **Ablation:** identical model & target, registered vs unregistered RGB → **18.37 vs 16.95
dB (+1.42 dB, −0.013 MAE)**. (At convergence a CNN partly *learns* the consistent global crop, so
the gap is smaller than the early-epoch gap; explicit registration still wins and is principled.)

## Data foundation
- 418 usable RGB/thermal pairs (51 thermals have no RGB; excluded). Split **336 / 41 / 41** (seed 42).
- Palette inversion (PCA principal-curve + quantile-binned LUT) → scalar target at 640×512, renderable back to color for viewing/LPIPS.
- Registration: global crop c≈0.65 + per-image translation refine; quality (edge-NCC) mean 0.171, none <0.05.
- Priors precomputed: Depth-Anything-V2 depth, AlphaEarth 64-D (cleaned), shadow/veg proxies.

## Approaches (all predict the scalar field, 512×640)
- **A1 — foundation dense regression:** ImageNet-pretrained ConvNeXt encoder + U-Net decoder; loss L1 + (1−SSIM) + gradient. Variants: RGB / +depth / +depth+AlphaEarth-FiLM / ConvNeXt-S.
- **A2 — conditional GAN (pix2pix):** same generator + PatchGAN; LSGAN + L1 + LPIPS + SSIM.
- **A4 — physics-structured:** predicts K soft **material masks** + **illumination**, composes
  temperature from learned per-material signatures cooled by shadow + small residual. Interpretable.
- **Ensemble** of the top-3 + a mean-field floor.

## What worked / surprised us
- **Ensemble is best** (18.90 dB) — regression + GAN + physics are complementary.
- **GAN is the most perceptually realistic** (color-LPIPS 0.307, clearly best) while matching regression on PSNR — the right pick if visual realism matters.
- **The physics model is competitive *and* interpretable** (18.03 dB). It learns ordered material
  temperature signatures (K=8: `[0.28, 0.39, 0.44, 0.51, 0.61, 0.68, 0.76, 0.85]`) and produces
  clean material maps — directly operationalizing Prof. Siwo's "material-conditioned" intuition.
- **Priors did NOT help.** Adding Depth-Anything depth *hurt* (17.34 vs 18.37) — unsurprising, since
  the depth is itself derived from the same RGB, adding little new signal and some noise. AlphaEarth
  helped depth slightly (rgbda 17.94 > rgbd 17.34) but neither beat **RGB-only** (18.37).
- **Bigger encoder didn't help** (ConvNeXt-S 17.35 < ConvNeXt-T 18.37) — expected with only 336 training images; the smaller pretrained net generalizes better.
- **Net:** with 12 MP RGB detail + correct target + registration, a *plain* RGB→scalar regressor is
  hard to beat; gains now come from ensembling and realism (GAN), not from more inputs or bigger nets.

## Limitations & next steps
- Residual cross-modal misalignment remains (caps pixel metrics). Next: per-image ECC/affine or a
  learned registration head; or alignment-robust losses (contextual/feature-space).
- Data scarcity (336 train). Next: transfer-learn from public aligned aerial RGB-T
  (**Kust4K** 4k pairs, **CART**) before fine-tuning.
- **A3 (conditional diffusion / ControlNet)** not run here — natural realism upgrade for future work.
- We predict a *relative* heat scalar. If radiometric R-JPEGs (per-pixel °C) are recoverable,
  we could calibrate to absolute temperature.

## Reproduce
Remote `rgb2thermal/`: `data_prep.py`, `r2t_common.py`, `train_a1.py`, `train_a2.py`,
`train_a4.py`, `evaluate.py`, `run_suite*.sh`. Re-score: `python evaluate.py` →
`outputs/leaderboard.csv` + `outputs/comparison_gallery.png`. Per-model `checkpoints/<name>/`:
`best.pth`, `metrics.json`, `sample.png` (physics also shows material maps).
Local mirror + galleries: `rgb2thermal_out/`.
