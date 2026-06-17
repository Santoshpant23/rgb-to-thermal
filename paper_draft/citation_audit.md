# Citation Audit

Week 11 added `paper_draft/references.bib` and inline citation keys to
`paper_draft/main_draft.md`.

## Coverage

- Total BibTeX entries: 42.
- Urban heat / thermal remote sensing: 9 entries.
- Core image-to-image translation, quality metrics, uncertainty, registration,
  and backbones: 16 entries.
- RGB-T / thermal datasets, fusion, and RGB-to-thermal translation: 17 entries.

## Source Checks

The most project-critical public dataset facts were checked against primary
pages:

- Kust4K Figshare page: title, authors, 4,024 pairs, 640x512 alignment, DOI,
  and CC BY 4.0 license.
- Caltech CART GitHub and CaltechDATA pages: arXiv ID, dataset description,
  labeled RGB-T subset, and repository-provided citation.
- NeurIPS/CCAI deadline status remains in `WEEK10_CCAI_WORKSHOP_PLAN.md`.

## Draft Usage

The WACV draft now cites:

- Urban heat motivation in the introduction.
- Paired/unpaired RGB-to-thermal and image-to-image baselines in Related Work.
- Public RGB-T, visible-infrared, and aerial thermal datasets in Related Work
  and Datasets.
- Spatial-transformer, homography, deformable, and learned-registration work in
  the registration paragraph.
- ConvNeXt, Swin, SwinIR, Restormer, U-Net, SSIM, and pix2pix/CycleGAN in the
  method and experiment paragraphs.

## Remaining Bibliography Work

- Convert the markdown citation keys to the final LaTeX citation syntax when
  the paper moves to LaTeX.
- Replace arXiv-only metadata with proceedings metadata where needed.
- Add DOI fields during the final camera-ready bibliography pass.
- Re-check any 2025/2026 preprints before submission because publication
  status may change.
