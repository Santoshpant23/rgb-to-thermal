# Week 2.5 Diagnostics

Purpose: determine whether the Week 2 no-go was real or caused by a weak
experimental harness.

## Critical Confounds

- The original perturbation range was small: at `sigma=0.3`, translation was
  only about 6 px at `256x320`.
- Kust4K and CART currently train against raw normalized thermal grayscale,
  which is low-frequency and gentle under PSNR.
- One seed plus GAN dynamics is not enough to interpret sub-dB differences.
- Ann Arbor was not tested, even though prior ablations show alignment matters
  there.

## Required Diagnostics

1. Ann Arbor training-time sweep:
   `sigma = {0, 0.1, 0.2, 0.3, 0.5}`.
2. Validation-time sensitivity:
   train aligned (`train_sigma=0`), then evaluate the saved aligned checkpoint
   with `eval_sigma = {0.1, 0.2, 0.3, 0.5}`.
3. Shuffled-RGB control:
   train/evaluate with RGB paired to the wrong target image. If PSNR is within
   1 dB of normal training, the model is mostly learning thermal priors.
4. Amplified perturbation:
   re-sweep with `max_translation_frac=0.20`, `max_rotation_deg=20`,
   `max_scale_frac=0.25`.
5. L1-only baseline:
   repeat the decisive cells with `--model l1` to remove GAN noise.
6. Multi-seed confirmation:
   use 3 seeds for the cells that drive the final decision.

## Command Patterns

Aligned Ann Arbor:

```bash
CUDA_VISIBLE_DEVICES=0 ../weather_experiments/.venv/bin/python -u week2_pix2pix_baseline.py \
  --dataset ann_arbor \
  --ann-arbor-cache ../rgb2thermal/data_cache \
  --sigma 0.0 \
  --epochs 20 \
  --height 256 \
  --width 320 \
  --batch-size 8 \
  --workers 4 \
  --out-dir week2_5_runs \
  --run-name pix2pix_ann_arbor_sigma_0.00_seed_42
```

Validation-time misalignment from an aligned checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 ../weather_experiments/.venv/bin/python -u week2_pix2pix_baseline.py \
  --dataset ann_arbor \
  --ann-arbor-cache ../rgb2thermal/data_cache \
  --eval-only \
  --checkpoint week2_5_runs/ann_arbor/pix2pix_ann_arbor_sigma_0.00_seed_42/best.pt \
  --train-sigma 0.0 \
  --eval-sigma 0.3 \
  --epochs 0 \
  --height 256 \
  --width 320 \
  --batch-size 8 \
  --workers 4 \
  --out-dir week2_5_eval_runs
```

Amplified perturbation:

```bash
CUDA_VISIBLE_DEVICES=0 ../weather_experiments/.venv/bin/python -u week2_pix2pix_baseline.py \
  --dataset kust4k \
  --kust4k-root data_cache/external/kust4k \
  --sigma 0.3 \
  --max-translation-frac 0.20 \
  --max-rotation-deg 20 \
  --max-scale-frac 0.25 \
  --epochs 20 \
  --height 256 \
  --width 320 \
  --batch-size 8 \
  --workers 4 \
  --out-dir week2_5_runs
```

