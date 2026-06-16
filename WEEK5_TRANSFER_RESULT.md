# Week 5 Transfer Result

Date: 2026-06-16

## Protocol

Locked Week 5 protocol:

- `target_normalization=robust`
- synthetic misalignment: `train_sigma=0.3`, `eval_sigma=0.3`
- amplified range: translation `0.20`, rotation `20 deg`, scale `0.25`
- supervised-affine runs use `lambda_warp_rgb=0.5`
- seed `42`
- Ann Arbor-source runs: `30` epochs, `bs=6`
- external-source runs: `20` epochs, `bs=8`

Each transfer cell has a matched no-registration baseline and supervised affine
run.

## Transfer Matrix

| Train -> Eval | No-reg PSNR | Affine PSNR | Delta | No-reg SSIM | Affine SSIM | Delta |
|---|---:|---:|---:|---:|---:|---:|
| Ann Arbor -> Kust4K | 9.902 | 10.306 | +0.404 | 0.386 | 0.415 | +0.028 |
| Kust4K -> Ann Arbor | 10.168 | 10.250 | +0.082 | 0.415 | 0.416 | +0.001 |
| Kust4K -> CART | 10.054 | 9.859 | -0.195 | 0.443 | 0.435 | -0.009 |
| CART -> Kust4K | 9.672 | 9.264 | -0.408 | 0.394 | 0.379 | -0.015 |

Full run table: `results/week5_transfer_matrix_summary.csv`.

## Pretrain Then Fine-Tune

| Run | Train | Eval | PSNR | Note |
|---|---|---|---:|---|
| from scratch | Ann Arbor | Ann Arbor | 15.681 | robust, seed 42, lambda 0.5 |
| external pretrain only | Kust4K+CART | Ann Arbor | 9.983 | poor direct transfer |
| external pretrain -> AA fine-tune | Ann Arbor | Ann Arbor | 15.845 | +0.165 vs from scratch |

Full run table: `results/week5_pretrain_finetune_summary.csv`.

## Decision

Week 5 is complete, and the cross-dataset story is weak:

- supervised affine helps on Ann Arbor -> Kust4K (`+0.404 dB`);
- Kust4K -> Ann Arbor is near-null (`+0.082 dB`);
- Kust4K -> CART and CART -> Kust4K are negative;
- Kust4K+CART pretraining gives only a small Ann Arbor fine-tuning gain
  (`+0.165 dB`) and remains far below the old `19.28 dB` local target.

Do not claim robust cross-dataset generalization from Week 5. The honest
position is that synthetic warp supervision can help in selected source/target
settings, but target/domain shift dominates the current transfer setup. Unless
Week 6 baselines reveal a stronger comparison, the paper should narrow toward a
workshop-strength empirical story rather than a WACV main-conference
cross-dataset registration method claim.
