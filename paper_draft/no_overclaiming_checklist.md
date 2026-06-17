# No-Overclaiming Checklist for Week 9 Draft

Use this before sending the draft to collaborators.

## Abstract Claims

| Claim | Evidence | Status |
|---|---|---|
| Alignment and target conventions matter for aerial RGB-to-thermal translation. | Week 2.5 diagnostics; Week 5 target-normalization audit. | Supported |
| Method improves over ConvNeXt no-registration by `+0.571 +/- 0.157 dB`. | `results/week7_ablation_summary.csv`, three seeds. | Supported |
| Method beats Swin-T no-registration by `+0.215 +/- 0.113 dB`. | Week 7 ablation summary, paired three-seed comparison. | Supported but modest |
| Kust4K does not show meaningful within-dataset registration gain. | Week 5 preflight, `+0.096 +/- 0.067 dB`. | Supported |
| CART gains are loss-balance-sensitive. | CART lambda sweep from Week 5. | Supported |
| Synthetic warp supervision improves robustness on source dataset. | Ann Arbor Week 7 ablation. | Supported |
| Method generalizes broadly across datasets. | Not supported. | Must not claim |
| Uncertainty weighting helps reconstruction. | Not supported; it hurts. | Must not claim |
| Hero figure shows typical gain. | Not supported; hero is upper-tail. | Must not claim |

## Wording Rules

- Use "uncertainty-decoupled" for the final method.
- Use "diagnostic uncertainty map"; do not call it calibrated uncertainty.
- Use "synthetic warp-recovery supervision"; do not call it real-world
  registration supervision.
- Use "qualitative cross-dataset context"; do not call the cross-dataset
  gallery a unified quantitative experiment.
- Use "small but repeatable gain"; do not imply a large performance jump.

## Figure Caption Checks

- Hero caption must state that `+0.93 dB` is illustrative and table results are
  `+0.571 +/- 0.157 dB`.
- Cross-dataset caption must state that Kust4K is not significant across seeds.
- Failure caption must state that failures are selected by negative paired
  delta, not low absolute PSNR.
- Multi-sigma caption must state that the sigma curve is representative, not a
  monotonic statistical claim.

## Week 11 Abstract Sentence Audit

| Abstract sentence | Evidence | Status |
|---|---|---|
| RGB-to-thermal translation from aerial imagery is sensitive to camera alignment, target representation, and dataset conventions. | Week 2.5 diagnostics, Week 5 target-normalization audit, external-dataset audits. | Supported |
| We study Ann Arbor, Kust4K, and CART. | Unified loader, `EXTERNAL_DATASETS.md`, Tables 1 and 4. | Supported |
| The method adds an RGB-only affine registration head with synthetic warp-recovery supervision. | `week3_registration_v0.py`, Week 7 primary rows. | Supported |
| The strongest variant decouples uncertainty from reconstruction and treats uncertainty maps as diagnostics. | Week 7 ablation rows with `lambda_unc=0`, `lambda_unc_tv=0`, `disable_uncertainty_weight=True`. | Supported |
| Ann Arbor improves by `+0.571 +/- 0.157 dB` over matched ConvNeXt no-registration. | `results/week7_ablation_summary.csv`, three seeds. | Supported |
| The method is `+0.215 +/- 0.113 dB` above Swin-T U-Net. | Week 7 paired table; modest margin. | Supported but should stay hedged |
| Kust4K has no meaningful within-dataset gain and CART is loss-balance-sensitive. | Week 5 preflight, Kust4K `+0.096 +/- 0.067 dB`; CART lambda sweep. | Supported |
| The conclusion is conservative: source-dataset robustness improves, but target normalization and dataset-specific alignment limit cross-dataset claims. | Directly supported by Week 5-7. | Supported |
