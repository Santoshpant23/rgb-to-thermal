# Week 11 Review Packet

Use this packet for the internal review pass before sending to Prof. Siwo.

## Files To Review

- Main draft: `paper_draft/main_draft.md`
- Tables and captions: `paper_draft/tables_and_captions.md`
- No-overclaiming checklist: `paper_draft/no_overclaiming_checklist.md`
- Method diagram: `paper_draft/method_diagram.svg`
- Polished preview figures:
  - `figures/week8/hero_ann_arbor_polished_seed42.png`
  - `figures/week8/failure_cases_ann_arbor_polished_seed42.png`
  - `figures/week8/cross_dataset_gallery_polished_seed42.png`

## Internal Review Instructions

Adithya and Santosh should mark every paragraph with one of:

- Clear
- Unclear
- Wrong
- Too strong
- Needs citation

Do not line-edit first. The priority is claim shape, missing evidence, and
whether the paper reads like a WACV submission rather than a hackathon report.

## Prof. Siwo Request

Ask for top-level feedback only:

1. Is the urban/aerial thermal framing credible enough for the target venues?
2. Is the contribution better framed as a robustness method, an empirical
   protocol study, or both?
3. Are the negative external results framed honestly without weakening the
   paper too much?
4. Which figure should become the primary paper figure?
5. Is the CCAI workshop branch worth keeping as a parallel safety-net paper?

## Known Caveats To Disclose Up Front

- The main positive result is source-dataset robustness on Ann Arbor, not broad
  cross-dataset registration.
- Kust4K within-dataset registration is null across three seeds.
- CART gains are loss-balance-sensitive.
- The hero example is an upper-tail qualitative example.
- The polished PNGs are preview figures; final paper figures still need LaTeX
  layout and citation-ready captions.
