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
