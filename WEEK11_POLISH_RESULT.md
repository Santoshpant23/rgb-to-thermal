# Week 11 Polish Result

Date: 2026-06-17

## Completed Locally

- Added `paper_draft/references.bib` with 42 bibliography entries covering:
  - urban heat and thermal remote sensing,
  - RGB-to-thermal and image-to-image translation,
  - spatial transformer / registration methods,
  - uncertainty weighting,
  - ConvNeXt/Swin/restoration backbones,
  - RGB-T, visible-infrared, Kust4K, and CART datasets.
- Added inline citation keys throughout `paper_draft/main_draft.md`.
- Added `paper_draft/citation_audit.md` documenting citation coverage and
  remaining bibliography cleanup.
- Added `week11_polish_figures.py`.
- Generated three paper-preview polished PNGs from the existing Week 8 figures:
  - `figures/week8/hero_ann_arbor_polished_seed42.png`
  - `figures/week8/failure_cases_ann_arbor_polished_seed42.png`
  - `figures/week8/cross_dataset_gallery_polished_seed42.png`
- Extended `paper_draft/no_overclaiming_checklist.md` with an abstract
  sentence-by-sentence audit.
- Added `paper_draft/review_packet.md` for internal review and Prof. Siwo
  top-level feedback.

## Claims Tightened

- The draft now explicitly cites urban heat and thermal remote-sensing work
  before motivating RGB-to-thermal translation.
- Public dataset claims now cite Kust4K and CART, and visible-infrared/RGB-T
  benchmark context.
- The method now cites U-Net, ConvNeXt, spatial transformer, and SSIM sources.
- The Swin-T discussion now explicitly separates Swin-T encoder proxy results
  from true SwinIR/Restormer restoration baselines.
- The uncertainty paragraph now cites uncertainty weighting literature while
  preserving the project-specific finding that uncertainty weighting hurt here.

## Figure Polish

The polished PNGs are preview artifacts, not final camera-ready figures. They
add consistent TrueType title/subtitle typography and explicit guardrail notes
without modifying the original Week 8 figures. This avoids rerunning checkpoint
inference locally, because the original checkpoints live on Knox.

## Remaining Blockers

- Prof. Siwo has not provided feedback yet, so "address Prof. Siwo's feedback"
  remains externally blocked.
- The paper is still markdown-first. LaTeX conversion, citation style, DOI
  completion, and final reference formatting remain Week 12 / camera-ready work.
- Some recent arXiv entries may gain proceedings metadata before submission;
  re-check publication status during the final bibliography pass.

## Validation

- BibTeX entries: 42.
- Inline citation key occurrences in `main_draft.md`: 57.
- Polished figures visually inspected locally.
- No new experiments were added; Week 11 is a paper-polish pass only.
