# Week 4 Registration v1 Progress

Date: 2026-06-16

## Question

Does learned affine registration beat the same ConvNeXt+U-Net translator without
registration under the Ann Arbor amplified `sigma=0.3` protocol?

## Same-Protocol Comparison

All runs use:

- Ann Arbor train/val: `336 / 41`
- `train_sigma=0.3`, `eval_sigma=0.3`
- `max_translation_frac=0.20`, `max_rotation_deg=20`, `max_scale_frac=0.25`
- `res=256`, `epochs=30`, `seed=42`

| Run | Mechanism | PSNR | Delta vs no-reg |
|---|---|---:|---:|
| `week3_no_registration_ann_arbor_sigma03_amp_seed42` | no registration | 15.886 | 0.000 |
| `week3_reg_shared_rgb_ann_arbor_sigma03_amp_seed42` | shared feature-space affine | 15.814 | -0.072 |
| `week4_input_rgb_affine_ann_arbor_sigma03_amp_seed42` | input-space RGB affine | 15.772 | -0.114 |

## Read

The simple affine variants are not enough. Both feature-space affine and
input-space affine train stably, but neither beats the no-registration
ConvNeXt+U-Net baseline under the same protocol.

This is useful: Week 4 should now move to TPS or small dense flow, and the
no-registration baseline should remain the primary comparison for every v1
registration candidate.

## Target-Conditioned Caveat

The target-conditioned variant remains an internal oracle/sanity check only. It
mostly predicts identity (`theta_l2 ~= 0.014`) and should not be used as evidence
that learned registration works.

