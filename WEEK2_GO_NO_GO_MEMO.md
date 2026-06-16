# Week 2 Provisional Go/No-Go Memo

Date: 2026-06-15

Superseded by `WEEK2_5_DIAGNOSTICS_RESULT.md`. The Week 2.5 diagnostics show
that this no-go was not trustworthy as a final pivot decision.

## Decision

**PROVISIONAL NO-GO for the WACV registration-bottleneck story under this
specific pix2pix training-time-perturbation protocol. Do not pivot the paper
until Week 2.5 diagnostics are complete.**

The Week 2 criterion was: PSNR should drop by at least `2 dB` at
`sigma = 0.3` on at least one external dataset. It did not. The drop was below
`1 dB` on both datasets.

## Experiment

- Model: small pix2pix baseline from `week2_pix2pix_baseline.py`
- Training perturbation: synthetic RGB-only affine misalignment
- Evaluation: aligned validation pairs
- Resolution: `256x320`
- Epochs: `20`
- Datasets: usable Kust4K and Caltech CART supervised pairs
- Knox output roots: `week2_runs/` for `sigma=0.0,0.3`; `week2_sweep_runs/`
  for `sigma=0.1,0.2,0.5`

## Results

| Dataset | Sigma 0.0 | Sigma 0.1 | Sigma 0.2 | Sigma 0.3 | Sigma 0.5 | Drop at 0.3 |
|---|---:|---:|---:|---:|---:|---:|
| Kust4K | 16.354 | 16.077 | 15.689 | 16.007 | 15.408 | 0.347 |
| Caltech CART | 16.456 | 16.409 | 15.762 | 16.270 | 15.989 | 0.185 |

Best-epoch PSNR shows the same conclusion:

| Dataset | Sigma 0.0 | Sigma 0.3 | Drop at 0.3 |
|---|---:|---:|---:|
| Kust4K | 16.354 | 16.235 | 0.119 |
| Caltech CART | 16.676 | 16.476 | 0.200 |

## Interpretation

This does not support the claim that alignment uncertainty is the dominant
bottleneck for this pix2pix setup, but the protocol has confounds that bias
toward small PSNR drops:

- `sigma=0.3` only creates about a 6 px max translation at `256x320`, which is
  small relative to the generator's downsampled receptive field.
- Kust4K and CART use raw normalized thermal grayscale targets, which are much
  lower-frequency than the Ann Arbor palette-inverted scalar target.
- The baseline is far below published Kust4K ceilings, and the severity curve is
  non-monotonic, so sub-dB changes are likely within seed/GAN variance.
- Ann Arbor, where prior ablations already show alignment matters, was not part
  of the first sweep.

Before fully pivoting the paper, run Week 2.5 diagnostics:

1. Run the same sweep on Ann Arbor.
2. Evaluate trained aligned models with RGB misalignment applied at validation
   time, to test sensitivity when the model cannot adapt during training.
3. Train/evaluate a shuffled-RGB control, to measure how much PSNR is possible
   from thermal priors alone.
4. Re-run the sweep with amplified perturbations: max translation fraction
   `0.20`, max rotation `20 deg`, max scale fraction `0.25`.
5. Run an L1-only regression baseline to remove adversarial training noise.
6. Use at least three seeds for the cells that determine the final claim.

If Ann Arbor, validation-time perturbation, shuffled-RGB, or amplified-sigma
diagnostics show a clean `>= 2 dB` drop, the WACV story stays alive and Week 3
should proceed. If they still stay below `1 dB`, the no-go is solid and we
should re-scope toward a workshop-first contribution around palette inversion,
multi-dataset RGB-to-thermal benchmarking, and failure analysis.
