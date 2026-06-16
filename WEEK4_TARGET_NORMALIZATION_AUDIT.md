# Week 4 Target Normalization Audit

Date: 2026-06-16

## Purpose

Quantify the target-convention mismatch before using Kust4K/CART for final
cross-dataset claims. Ann Arbor uses palette-inverted scalar targets; Kust4K and
CART currently use raw normalized thermal grayscale from `unified_dataset.py`.

## Validation Split Summary

Computed at `256x320` using `week4_target_normalization_audit.py`.

| Dataset | Count | Mean | Std | Edge Mean | Edge P90 | Entropy |
|---|---:|---:|---:|---:|---:|---:|
| Ann Arbor | 41 | 0.521 | 0.275 | 0.0330 | 0.0849 | 5.516 |
| Kust4K | 283 | 0.431 | 0.180 | 0.0196 | 0.0455 | 5.211 |
| Caltech CART | 222 | 0.409 | 0.151 | 0.0166 | 0.0319 | 4.807 |

## Read

The external targets are materially smoother under the current loader:

- Kust4K edge mean is about `59%` of Ann Arbor.
- CART edge mean is about `50%` of Ann Arbor.
- CART edge P90 is only about `38%` of Ann Arbor.

This supports the Week 2.5/3 caveat: cross-dataset PSNR-drop magnitudes are not
comparable yet. Kust4K/CART need a normalization or edge-preserving target
representation before Week 5 cross-dataset claims.

## Next Options

- Prefer Option A: implement a per-dataset target normalizer with robust
  percentile scaling plus optional histogram matching to Ann Arbor scalar
  targets.
- Keep Option B as fallback: report only direction/order on external datasets
  and avoid magnitude claims until normalization is defensible.

