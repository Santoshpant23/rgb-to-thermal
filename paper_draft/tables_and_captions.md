# Week 9 Paper Tables and Captions

## Table 1. Datasets and Target Representation

| Dataset | Train / Val / Test | Target | Normalization | Edge mean | Paper use |
| --- | --- | --- | --- | --- | --- |
| Ann Arbor | 336 / 41 / 42 | Recovered scalar thermal | robust | 0.035 | Primary source-dataset protocol |
| Kust4K | 1970 / 283 / 565 | Raw grayscale TIR | robust for transfer; raw in legacy qualitative row | 0.029 | Official splits minus author-flagged broken stems |
| CART | 1822 / 222 / 238 | Raw grayscale thermal | robust for transfer; raw in legacy qualitative row | 0.026 | Full labeled_rgbt_pairs archive |

## Table 2. Baseline Results

| Method | Family | Seed(s) | PSNR | SSIM | Pearson r | Note |
| --- | --- | --- | --- | --- | --- | --- |
| CycleGAN | unpaired | 42 | 8.598 | 0.269 | 0.180 | unpaired CycleGAN, no paired L1 |
| pix2pix | paired_gan | 42 | 11.835 | - | - | vanilla paired pix2pix |
| Small U-Net L1 | paired_l1 | 42 | 12.714 | - | - | pix2pix generator with L1 only |
| ConvNeXt+U-Net | paired_regression | 42 | 15.637 | 0.536 | 0.812 | existing ConvNeXt+U-Net no-registration baseline |
| Swin-T+U-Net | pretrained_transformer | 42 | 16.123 | 0.566 | 0.834 | pretrained timm Swin-T encoder plus U-Net decoder |
| Ours: ConvNeXt affine, uncertainty-decoupled | paired_regression + affine | 42/7/123 | 16.444 +/- 0.127 | - | - | Primary 3-seed method; not directly single-seed comparable to rows above |

## Table 3. Main Registration Ablation

| Variant | Seeds | PSNR mean +/- std | Paired delta vs ConvNeXt no-reg |
| --- | --- | --- | --- |
| ConvNeXt no-registration | 3 | 15.872 +/- 0.217 | - |
| ConvNeXt affine + uncertainty weighting | 3 | 16.133 +/- 0.199 | +0.260 +/- 0.021 |
| ConvNeXt affine, uncertainty-decoupled | 3 | 16.444 +/- 0.127 | +0.571 +/- 0.157 |
| Swin-T no-registration | 3 | 16.228 +/- 0.106 | +0.356 +/- 0.115 |
| Swin-T affine | 3 | 16.164 +/- 0.297 | +0.292 +/- 0.200 |

## Table 4. External and Transfer Results

| External experiment | Delta PSNR | Seeds | Interpretation |
| --- | --- | --- | --- |
| Kust4K within-dataset | +0.096 +/- 0.067 | 3 | No positive claim; CI overlaps zero |
| CART within-dataset | +0.782 +/- 0.368 | 3 | Passes mean threshold but loss-balance-sensitive |
| Ann Arbor -> Kust4K transfer | +0.474 +/- 0.061 | 3 | Only transfer cell that survives a 3-seed audit |
| Kust4K -> Ann Arbor transfer | +0.082 | 1 | Single-seed diagnostic only |
| Kust4K -> CART transfer | -0.195 | 1 | Single-seed diagnostic only |
| CART -> Kust4K transfer | -0.408 | 1 | Single-seed diagnostic only |

## Draft Figure Captions

**Figure 1. Hero qualitative example.** Ann Arbor validation scene with visible
road, tree canopy, and hot roof/building edges. The method improves this sample
from `16.01 dB` to `16.94 dB`, but this is an illustrative upper-tail example;
the main three-seed gain is `+0.571 +/- 0.157 dB`.

**Figure 2. Method overview.** Synthetic RGB misalignment is applied while the
thermal target remains fixed. A lightweight RGB-only affine head predicts an
input-space warp, the warped RGB is translated by a ConvNeXt-tiny U-Net, and an
auxiliary warp-recovery loss supervises the predicted warp. The uncertainty map
is logged for diagnostics but is not used for reconstruction weighting in the
primary variant.

**Figure 3. Multi-sigma recovery.** Same Ann Arbor scene evaluated at synthetic
misalignment sigma `0.0`, `0.2`, and `0.5`. The method helps at sigma `0.2` on
this scene but loses slightly at sigma `0.5`, so this figure should be read as
representative behavior, not a monotonic severity claim.

**Figure 4. Failure cases.** Ann Arbor validation samples with the most negative
paired PSNR deltas for the method relative to the no-registration baseline. The
dominant failure is smoothing of high-frequency thermal structure around roofs,
vehicles, and building edges; uncertainty maps do not fully localize the errors.

**Appendix Figure A. Candidate grid.** PSNR-quantile validation examples showing
RGB input, learned warp, target, prediction, absolute error, and uncertainty.

**Appendix Figure B. Cross-dataset qualitative gallery.** Representative rows
for Ann Arbor, Kust4K, and CART. This is qualitative context only. The Kust4K
within-dataset gain is not statistically significant across seeds
(`+0.096 +/- 0.067 dB`), and Kust4K/CART rows use legacy within-dataset
checkpoints rather than the locked Ann Arbor Week 7 protocol.
