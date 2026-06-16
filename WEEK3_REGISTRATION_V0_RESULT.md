# Week 3 Registration v0 Result

Date: 2026-06-15

## What Landed

`week3_registration_v0.py` implements the first learned registration sanity
check:

- synthetic RGB-only misalignment on Ann Arbor;
- a small registration head that consumes misaligned RGB plus scalar thermal
  target and predicts an affine warp plus uncertainty map;
- warping before the existing ConvNeXt+U-Net translator from `train_a1.py`;
- losses for uncertainty-weighted reconstruction, RGB/thermal edge alignment,
  affine identity regularization, and uncertainty smoothness.

## First Ann Arbor Pass

Knox run:

`/home/spant/UMich/umich-hackathon/rgb2thermal_wacv/week3_runs/week3_reg_v0_ann_arbor_sigma03_amp_seed42`

Settings:

- `train_sigma=0.3`, `eval_sigma=0.3`
- `max_translation_frac=0.20`, `max_rotation_deg=20`, `max_scale_frac=0.25`
- `res=256`, `epochs=30`, `batch_size=6`
- train/val: `336 / 41`

Final validation metrics:

| Metric | Value |
|---|---:|
| MAE | 0.1043 |
| PSNR | 15.452 |
| SSIM | 0.544 |
| Corr | 0.787 |
| theta_l2 | 0.0139 |
| uncertainty mean | 1.085 |

## Read

This is a successful v0 sanity check: the learned affine correction and
uncertainty-weighted translator can train stably on the Ann Arbor amplified
misalignment task.

This is not yet the final paper architecture. The registration head is
target-conditioned, because Week 3 explicitly tests RGB/thermal feature-based
registration. A deployable model still needs either a test-time registration
proxy or a refactor where registration supervision improves an RGB-only
translator.

## Open Items

- Replace the separate registration head with shared encoder features or justify
  the separate head as v0-only.
- Audit/improve Kust4K and CART thermal target normalization before using them
  for final cross-dataset claims.
- Add qualitative warp/uncertainty visualizations in Week 4 or Week 8.

