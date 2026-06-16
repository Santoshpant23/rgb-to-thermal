# Week 2.5 Diagnostics Result

Date: 2026-06-15

## Decision

**The Week 2 no-go is overturned as a final decision. Do not pivot yet.**

The original external-only training-time sweep was too weak to justify pivoting.
After adding Ann Arbor, validation-time perturbation, shuffled-RGB controls,
amplified perturbations, multi-seed checks, and an L1-only baseline, the evidence
supports a narrower version of the WACV story:

> Alignment is a real bottleneck on the Ann Arbor scalar task and under
> meaningful synthetic perturbations; Kust4K and CART are less sensitive under
> raw grayscale targets and may be too pre-aligned or too low-frequency to carry
> the claim by themselves.

Proceed to Week 3, but keep the external-target normalization audit open before
making final cross-dataset claims.

## Key Findings

### Ann Arbor Control

Ann Arbor shows the expected sensitivity.

| Setting | Sigma | PSNR | Drop |
|---|---:|---:|---:|
| Original train-time | 0.0 | 14.073 | - |
| Original train-time | 0.3 | 12.093 | 1.980 |
| Original train-time | 0.5 | 10.257 | 3.816 |
| Amplified train-time | 0.0 | 14.259 | - |
| Amplified train-time | 0.3 | 9.603 | 4.656 |

With three seeds on amplified `sigma=0.3`, Ann Arbor's mean drop was
`2.606 +/- 1.941 dB`.

### Validation-Time Misalignment

Training aligned and perturbing only at validation time exposes fragility that
the original training-time augmentation partly hid.

| Dataset | Eval sigma 0.3 drop | Eval sigma 0.5 drop |
|---|---:|---:|
| Ann Arbor | 1.696 | 2.430 |
| Kust4K | 1.044 | 1.645 |
| Caltech CART | 0.576 | 0.977 |

### Shuffled-RGB Control

The models are not merely learning thermal priors. Breaking RGB-target pairing
caused large drops:

| Dataset | Normal PSNR | Shuffled-RGB PSNR | Drop |
|---|---:|---:|---:|
| Ann Arbor | 14.073 | 7.186 | 6.887 |
| Kust4K | 16.354 | 12.767 | 3.587 |
| Caltech CART | 16.456 | 11.846 | 4.610 |

### Amplified Perturbation

Using `max_translation_frac=0.20`, `max_rotation_deg=20`, and
`max_scale_frac=0.25` makes `sigma=0.3` meaningful.

| Dataset | 3-seed aligned PSNR | 3-seed sigma 0.3 PSNR | Mean drop |
|---|---:|---:|---:|
| Ann Arbor | 13.816 | 11.210 | 2.606 |
| Kust4K | 16.285 | 15.201 | 1.084 |
| Caltech CART | 16.682 | 16.158 | 0.524 |

### L1-Only Baseline

Removing GAN dynamics keeps the same direction: Ann Arbor remains sensitive,
while Kust4K/CART remain weaker.

| Dataset | Default sigma 0.3 drop | Amplified sigma 0.3 drop |
|---|---:|---:|
| Ann Arbor | 1.544 | 2.497 |
| Kust4K | 0.486 | 0.590 |
| Caltech CART | 0.285 | 0.535 |

## Interpretation

The original no-go was a harness failure, not a reliable hypothesis failure.
The corrected read is:

- Ann Arbor confirms the alignment bottleneck.
- Validation-time perturbation confirms models are sensitive when they cannot
  adapt during training.
- Shuffled-RGB controls confirm the models use RGB signal.
- Amplified perturbations produce the expected severity response, especially on
  Ann Arbor.
- Kust4K and CART should be treated carefully: they may be too well aligned, too
  low-frequency under raw grayscale targets, or too easy for this baseline.

## Next Step

Start Week 3, but frame it as:

1. build the learned registration module using Ann Arbor and amplified synthetic
   perturbations as the primary sanity check;
2. keep Kust4K/CART as external generalization tests, not as the sole go/no-go
   evidence;
3. audit Kust4K/CART target normalization before final external claims.

