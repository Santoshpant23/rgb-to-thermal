# Week 7 Ablations Result

Date: 2026-06-16

This memo extends the early Week 7 seed audit. It resolves the Swin-T stacking
gap, bounds the loss-formulation confound, and runs the first ConvNeXt ablation
table for the locked Ann Arbor robust protocol.

Full table: `results/week7_ablation_summary.csv`.

## Protocol

- dataset: Ann Arbor train -> Ann Arbor val
- target normalization: `robust`
- synthetic misalignment: amplified train/eval sigma unless noted
- resolution: `256 x 320`
- training budget: `50` epochs, `336` train samples, `41` validation samples
- primary comparison: paired by seed over `{42, 7, 123}`

## Main Update

The best current method is not the uncertainty-weighted affine model. It is the
same ConvNeXt + UNetReg + input-space affine head with direct RGB warp
supervision, but with uncertainty decoupled from the reconstruction loss. In
the primary rows, the uncertainty head still exists and is logged, but its
output is neither used to weight L1 reconstruction nor regularized
(`lambda_uncertainty=0`, `lambda_uncertainty_tv=0`).

| Method | Seed 42 | Seed 7 | Seed 123 | Mean +/- std |
|---|---:|---:|---:|---:|
| ConvNeXt no-reg | 15.637 | 16.062 | 15.918 | 15.872 +/- 0.217 |
| ConvNeXt affine + uncertainty | 15.920 | 16.315 | 16.162 | 16.133 +/- 0.199 |
| ConvNeXt affine uncertainty-decoupled | 16.312 | 16.453 | 16.566 | 16.444 +/- 0.127 |
| Swin-T no-reg | 16.123 | 16.335 | 16.228 | 16.228 +/- 0.106 |
| Swin-T affine | 16.001 | 16.507 | 15.984 | 16.164 +/- 0.297 |

Paired deltas:

| Comparison | Mean +/- std |
|---|---:|
| ConvNeXt affine + uncertainty - no-reg | +0.260 +/- 0.021 |
| ConvNeXt affine uncertainty-decoupled - no-reg | +0.571 +/- 0.157 |
| ConvNeXt affine uncertainty-decoupled - affine + uncertainty | +0.311 +/- 0.151 |
| ConvNeXt affine uncertainty-decoupled - Swin-T no-reg | +0.215 +/- 0.113 |
| Swin-T affine - Swin-T no-reg | -0.064 +/- 0.214 |

Interpretation:

- Registration remains real on ConvNeXt, and the uncertainty-decoupled variant
  clears the old `+0.3 dB` continuation threshold over three seeds.
- Uncertainty weighting hurts under this protocol. Keep uncertainty maps for
  diagnostics only unless a later calibration run fixes this.
- Swin-T affine does not show a reliable stacking benefit over Swin-T no-reg;
  its three-seed mean delta is near-null/negative and high variance.
- Swin-T no-reg no longer beats the revised ConvNeXt method; the
  uncertainty-decoupled affine variant is the current top row in this
  controlled table.

## Loss-Formulation Confound

The audit correctly flagged that Swin-T and ConvNeXt originally used different
loss recipes. A ConvNeXt no-registration seed-42 control trained with Swin-T's
`combined_loss` reached `15.875 dB`, compared with `15.637 dB` for the original
ConvNeXt no-registration loss. An explicit repeat with
`lambda_uncertainty=0` and `lambda_uncertainty_tv=0` reached `15.839 dB`.

This means the loss recipe is worth roughly `+0.20` to `+0.24 dB` at seed 42.
It is not negligible, but it does not explain the full set of Week 7 outcomes,
because the revised uncertainty-decoupled affine ConvNeXt reaches `16.312 dB`
at the same seed.

## Loss Ablations

Seed-42 ablations against the uncertainty-decoupled affine primary:

| Variant | PSNR | Delta vs primary |
|---|---:|---:|
| ConvNeXt affine uncertainty-decoupled | 16.312 | +0.000 |
| Add uncertainty weighting/regularizers | 15.920 | -0.392 |
| Remove RGB warp-recovery loss | 15.814 | -0.498 |
| Remove edge loss | 16.072 | -0.240 |
| Remove SSIM loss | 16.187 | -0.125 |
| Remove affine identity regularization | 16.083 | -0.229 |

The load-bearing pieces are direct RGB warp supervision and disabling
uncertainty weighting. Edge, SSIM, and affine regularization each help at seed
42, but less than the warp/uncertainty decisions.

## Severity Sweep

Seed-42 severity curve, using matched train/eval sigma. After audit feedback,
the no-registration curve was rerun with explicit uncertainty-decoupled
metadata (`--disable-uncertainty-weight`, `lambda_uncertainty=0`,
`lambda_uncertainty_tv=0`) so the table is protocol-matched.

| Sigma | No-reg PSNR | Uncertainty-decoupled affine PSNR | Delta |
|---:|---:|---:|---:|
| 0.0 | 17.052 | 17.278 | +0.226 |
| 0.1 | 16.557 | 16.304 | -0.253 |
| 0.2 | 15.618 | 16.414 | +0.796 |
| 0.3 | 15.861 | 16.312 | +0.451 |
| 0.5 | 15.349 | 15.644 | +0.295 |

The rerun makes the comparison protocol-matched, but it also exposes
single-seed instability: the no-registration curve is not monotonic at
`sigma=0.3`, and the affine curve is also non-monotonic. Use this as a
diagnostic table only. It should not become a main paper figure without a
multi-seed repeat. The main robust claim remains the three-seed sigma `0.3`
result.

## Decision

Use `ConvNeXt + UNetReg + input-space affine + lambda_warp_rgb=0.5` with
uncertainty decoupled from reconstruction weighting as the Week 7 primary
method. The paper framing should be:

> Synthetic warp-recovery supervision gives a small but repeatable robustness
> gain on Ann Arbor. The uncertainty map is currently diagnostic, not a useful
> reconstruction weighting mechanism.

Still open before paper lock:

- decoder/backbone control for Swin-T if Swin-T remains a major baseline claim;
- multi-seed severity curve if the curve becomes a main figure;
- qualitative warp/uncertainty visualizations for Week 8.
