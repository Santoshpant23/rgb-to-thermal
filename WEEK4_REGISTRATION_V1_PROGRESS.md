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

| Run | Mechanism | PSNR | Warp metric | Delta vs no-reg |
|---|---|---:|---:|---:|
| `week3_no_registration_ann_arbor_sigma03_amp_seed42` | no registration | 15.886 | 0.000 | 0.000 |
| `week3_reg_shared_rgb_ann_arbor_sigma03_amp_seed42` | shared feature-space affine | 15.814 | theta_l2 0.036 | -0.072 |
| `week4_input_rgb_affine_ann_arbor_sigma03_amp_seed42` | input-space RGB affine | 15.772 | theta_l2 0.013 | -0.114 |
| `week4_input_rgb_flow_ann_arbor_sigma03_amp_seed42` | input-space dense flow | 15.547 | flow_l2 0.0066 | -0.339 |
| `week4_input_rgb_flow_edge1_ann_arbor_sigma03_amp_seed42` | input-space dense flow, stronger edge loss | 15.384 | flow_l2 0.0214 | -0.502 |

## Read

The simple unsupervised registration variants are not enough. Feature-space
affine, input-space affine, and dense flow all train stably, but none beats the
no-registration ConvNeXt+U-Net baseline under the same protocol.

The default dense-flow model learned very small motion and underperformed by
`0.339 dB`. A stronger edge-loss run increased motion, but performance dropped
further to `-0.502 dB` vs no-registration. Blindly adding warp capacity is not
fixing the problem.

This is useful: the next Week 4 step should add a direct synthetic-warp
diagnostic/supervision path. Because the amplified Ann Arbor perturbation is
synthetic, we can compare against the known inverse transform before spending
more runs on TPS or external multi-dataset training. The no-registration
baseline should remain the primary comparison for every v1 registration
candidate.

## Target-Conditioned Caveat

The target-conditioned variant remains an internal oracle/sanity check only. It
mostly predicts identity (`theta_l2 ~= 0.014`) and should not be used as evidence
that learned registration works.
