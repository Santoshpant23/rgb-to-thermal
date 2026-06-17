# Week 6 Baselines Result

Date: 2026-06-16

## Protocol

Main fair-compute table:

- dataset: Ann Arbor train -> Ann Arbor val
- target normalization: `robust`
- synthetic misalignment: `train_sigma=0.3`, `eval_sigma=0.3`
- amplified range: translation `0.20`, rotation `20 deg`, scale `0.25`
- resolution: `256 x 320`
- seed: `42`
- training budget: `50` epochs, `336` train samples, `41` validation samples

This table is a seed-42 pass. The small margins between ConvNeXt+U-Net,
supervised affine, and Swin-T+U-Net are within the seed variance seen in Week 5,
so the ranking is not final until the Week 7 seed audit runs.

Full machine-readable table: `results/week6_baseline_summary.csv`.

The legacy `19.28 dB` ensemble+TTA result remains valid only for the older
raw-target local protocol. It is documented as a reproduced legacy result, not
included as a same-y-axis row in this robust-normalized Week 6 table.

The pretrained-backbone row uses the available pretrained timm Swin-T image
encoder with a U-Net decoder. It is a transformer-backbone baseline, not a true
SwinIR/Restormer restoration checkpoint.

It also uses the custom decoder in `week6_swin_unet_baseline.py`, while the
ConvNeXt rows use `UNetReg` from `train_a1.py`. Therefore the Swin-T gain is a
transformer-encoder-plus-decoder comparison, not an isolated encoder effect.

## Baseline Table

| Method | Family | PSNR | MAE | SSIM | Pearson r | Note |
|---|---|---:|---:|---:|---:|---|
| CycleGAN | unpaired | 8.598 | 0.302 | 0.269 | 0.180 | unpaired, no paired L1 |
| pix2pix | paired GAN | 11.835 | 0.183 | - | - | Week 2 pix2pix harness |
| Small U-Net L1 | paired L1 | 12.714 | 0.177 | - | - | pix2pix generator, L1 only |
| ConvNeXt+U-Net | paired regression | 15.637 | 0.105 | 0.536 | 0.812 | no registration |
| Ours supervised affine | registration | 15.920 | 0.099 | 0.549 | 0.828 | Week 5 run reused here; same robust 50-epoch protocol |
| Swin-T+U-Net | pretrained transformer | 16.123 | 0.096 | 0.566 | 0.834 | pretrained timm Swin-T encoder |

## Decision

Week 6 is complete as a seed-42 matched-compute baseline pass. The cleanest
registration comparison is within the ConvNeXt family:

- holding the ConvNeXt backbone and decoder family fixed, supervised affine
  registration beats no-registration by `+0.284 dB`;
- swapping to the Swin-T+custom-decoder baseline is larger at `+0.486 dB` over
  ConvNeXt+U-Net, but it changes both encoder and decoder;
- Swin-T+U-Net is `+0.203 dB` above the supervised-affine ConvNeXt method at
  seed 42, but this is within expected seed variance;
- the unpaired CycleGAN baseline is not competitive;
- pix2pix and small L1 U-Net are far below the pretrained/backbone baselines.

The current paper cannot claim state-of-the-art performance under the Week 6
robust protocol from this table alone. The defensible seed-42 claim is narrower:
synthetic warp supervision adds a modest gain over the matched ConvNeXt+U-Net
baseline, while a stronger transformer-backbone proxy is also competitive. Week
7 should prioritize:

- rerunning ConvNeXt+U-Net, supervised affine, and Swin-T+U-Net for seeds `7`
  and `123`;
- adding the supervised-affine registration head on top of Swin-T;
- treating pix2pix/L1 SSIM and Pearson cells as missing, not zero, because the
  legacy Week 2 harness did not log those metrics.

## Post-Audit Update

The Week 7 seed audit resolved the biggest Week 6 uncertainty. Over three
seeds, ConvNeXt supervised affine beats ConvNeXt no-registration by
`+0.260 +/- 0.021 dB`, so the effect is repeatable but below the old `+0.3 dB`
threshold. Swin-T+U-Net remains the strongest audited mean row, but only by
`+0.096 +/- 0.096 dB` over ConvNeXt supervised affine. Swin-T +
supervised-affine registration was negative at seed 42 (`-0.122 dB`), so the
registration head should not be framed as a generic module that stacks on the
stronger backbone.

See `WEEK7_SEED_AUDIT_RESULT.md` and
`results/week7_seed_audit_summary.csv`.
