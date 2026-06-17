# Week 7 Seed Audit Result

Date: 2026-06-16

This follow-up addresses the Week 6 audit concerns about seed variance and the
Swin-T stacking question.

## Protocol

- dataset: Ann Arbor train -> Ann Arbor val
- target normalization: `robust`
- synthetic misalignment: `train_sigma=0.3`, `eval_sigma=0.3`
- amplified range: translation `0.20`, rotation `20 deg`, scale `0.25`
- resolution: `256 x 320`
- training budget: `50` epochs, `336` train samples, `41` validation samples

Full table: `results/week7_seed_audit_summary.csv`.

## Three-Seed Audit

| Family | Seed 42 | Seed 7 | Seed 123 | Mean +/- std |
|---|---:|---:|---:|---:|
| ConvNeXt no-reg | 15.637 | 16.062 | 15.918 | 15.872 +/- 0.217 |
| ConvNeXt supervised affine | 15.920 | 16.315 | 16.162 | 16.133 +/- 0.199 |
| Swin-T no-reg | 16.123 | 16.335 | 16.228 | 16.228 +/- 0.106 |

Matched ConvNeXt registration deltas:

| Seed | ConvNeXt affine - ConvNeXt no-reg |
|---:|---:|
| 42 | +0.284 |
| 7 | +0.253 |
| 123 | +0.244 |
| Mean +/- std | +0.260 +/- 0.021 |

Swin-T no-reg vs ConvNeXt supervised-affine:

| Seed | Swin-T no-reg - ConvNeXt affine |
|---:|---:|
| 42 | +0.203 |
| 7 | +0.019 |
| 123 | +0.065 |
| Mean +/- std | +0.096 +/- 0.096 |

## Swin-T Registration Stacking

Swin-T + supervised-affine registration, seed 42:

| Model | PSNR | SSIM | Pearson r |
|---|---:|---:|---:|
| Swin-T no-reg | 16.123 | 0.566 | 0.834 |
| Swin-T supervised affine | 16.001 | 0.567 | 0.833 |
| Delta | -0.122 | +0.000 | -0.001 |

Since the stacking check is negative at seed 42, seeds 7 and 123 were not run
for Swin-T supervised affine.

## Decision

- The ConvNeXt registration effect is stable and positive, but modest:
  `+0.260 +/- 0.021 dB`. It does not clear the old `+0.3 dB` threshold.
- Swin-T no-reg is the best mean row among the audited families, but its mean
  advantage over ConvNeXt supervised affine is only `+0.096 dB`.
- The supervised-affine head does not stack on Swin-T at seed 42.
- The current registration claim should be framed as a ConvNeXt-family ablation:
  synthetic warp supervision gives a small, repeatable gain over no-registration
  with the same backbone/decoder.
- The Swin-T decoder/backbone confound remains open. If this becomes a paper
  result, run a decoder control before attributing the gain to the encoder.
