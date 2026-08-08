# Iliad Intensive Computational Mechanics Day — Handoff

Updated 2026-08-02. The notebook-and-slide teaching suite is complete and
verified.

## Final state

The six short-lecture decks that frame and synthesize the three guided notebooks
have completed their bounded parent-editor revision, full rebuild, visual
inspection, and narrow independent acceptance check. No work remains on the
current request. Do not restart recursive critique unless the user explicitly
asks for another revision cycle.

A subsequent user-requested epigraph is also complete. Every exercise and
worked-solution notebook opens with the same continuous Latin trio, and the
final synthesis closes with it. The supplied middle clause shortens Seneca's
wording in *Moral Letters* 90.46, so the artifacts accurately say “Seneca the
Younger, adapted from *Moral Letters to Lucilius* 6.5, 90.46, and 81.13” rather
than presenting the trio as one verbatim passage.

## Teaching and spoiler constraints

- Keep the existing 10:00–10:30 overview unchanged pending Ben's edits. The new
  suite begins with the 10:30 HMM primer.
- The notebooks remain the derivational source of truth. Slides orient, gate
  reveals, synthesize, and hand off to the notebooks.
- Do not reveal Notebook 1 matrices before its first student answer, Mess3
  recursive geometry before the hand calculation, Notebook 2 empirical results
  before CORE 5, or Notebook 3 rank/basis/operator answers in its primer.
- Preserve the explicit distinctions between HMM beliefs and observable
  predictive state, finite-block evidence and full-process conclusions,
  decodability and causal use, and linear dimension and nonnegative HMM size.
- The GHMM/minimal-realization detour remains out of scope.

## Implemented slide suite

Sources and tooling:

- `slides/_quarto.yml`
- `slides/theme.scss`
- `slides/export-mode.html`
- `slides/build_slides.py`
- `slides/validate_slides.py`
- `slides/README.md`
- `slides/01_hmm_primer.qmd` — 5 slides
- `slides/02_hmm_msp_synthesis.qmd` — 10 slides
- `slides/03_transformer_primer.qmd` — 6 slides
- `slides/04_transformer_debrief.qmd` — 4 slides
- `slides/05_hankel_psr_primer.qmd` — 7 slides
- `slides/06_final_synthesis.qmd` — 10 slides

There are seven notebook-derived figures in `slides/assets/generated/`. Offline
Reveal HTML and matching PDF handouts are in `slides/dist/`.

HTML is the primary presentation format: it is offline, has speaker notes
(`S`), and supports overview mode (`O`). The PDFs are deterministic, lossless
raster handouts assembled from live 1600×900 Reveal canvases; this avoids the
browser print-layout drift seen in direct PDF printing.

## Final verification

- `python slides/build_slides.py` passed without skip flags. It rechecked
  notebook freshness and content, extracted all seven figures, rendered the six
  offline HTML decks, assembled the PDFs from live canvases, and ran the slide
  validator.
- The full build and all explicit validators were rerun after adding the Seneca
  epigraph. Notebook validation now requires the three clauses and attribution
  in all six generated variants; slide validation requires them in the final
  synthesis source.
- An explicit post-build `python slides/validate_slides.py` also passed counts
  5, 10, 6, 4, 7, and 10: 42 pages total. All PDFs are 16:9, structurally valid,
  and paired with offline HTML containing notes and local runtime assets.
- The changed Deck 02 reveal sequence, Deck 04 control wording, Deck 05
  caution/source slide, Deck 06 optional branch, and all six dark title slides
  were inspected in contact sheets and at selected full resolutions. The first
  pass exposed and fixed a wrapped Mess3 key and initial-frame title-furniture
  contrast; the rebuilt pages are clean.
- The final slide and a rendered exercise notebook were additionally inspected
  at full resolution after the epigraph update; the continuous Latin and its
  attribution are readable and unclipped.
- One fresh read-only verifier independently passed the Mess3 stop/result/
  recursive sequence (PDF pages 6/7/8), the induced word-weight validity
  conditions, and the 42-page validator result. No narrow defects remained.
- The apparent `slides/dist/dist` packaging problem reported by one reviewer was
  a false positive. The only nested `dist` directories are the expected bundled
  Reveal runtime paths such as `*_files/libs/revealjs/dist`. Do not change the
  packaging on that premise.

## Implemented parent-editor revisions

Three isolated reviewers assessed reader/pedagogy, technical correctness, and
projection/execution quality. Strengths worth preserving were the actionable
notebook handoffs, consistent edge/update conventions, careful epistemic
qualifiers, and the clean render. The bounded parent pass implemented all six
accepted findings:

1. Deck 02 now has a genuine text-only Mess3 stop before the enlarged one-symbol
   result and readable native-slide labels.
2. Deck 06's optional WFA notes allocate the advertised 8–10 minutes to setup,
   one supplied word check, validity interpretation, and close; reconstruction
   and matrix inversion are not assigned.
3. WFA validity is stated at the induced word-weight level: nonnegativity,
   length normalization, and extension consistency.
4. Deck 04 describes held-out MSE under random input–activation-pair splits
   without implying structured-history generalization.
5. Deck 05 uses one prospective finite-table caution and cites canonical PSR and
   observable-operator/linear-realization sources.
6. Dark title-slide furniture now renders with light contrast, including the
   initial PDF frame.

Canonical theory anchors already checked against primary publication pages:

- Littman, Sutton, and Singh, “Predictive Representations of State,” NIPS 2001:
  <https://proceedings.neurips.cc/paper_files/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html>
- Herbert Jaeger, “Observable Operator Models for Discrete Stochastic Time
  Series,” *Neural Computation* 12(6), 2000:
  <https://doi.org/10.1162/089976600300015411>
- Transformer evidence remains sourced to Shai et al.:
  <https://arxiv.org/abs/2405.15943>

## Delivery and future changes

- Present from `slides/dist/*.html`; press `S` for notes and `O` for overview.
  The matching PDFs are review/fallback handouts.
- Rebuild from the project root with `python slides/build_slides.py`; use
  `python slides/validate_slides.py` to check existing outputs.
- `PROJECT_CONTEXT.md` and `DAY_PLAN.md` now record the prepared artifacts. The
  normalized 17:00 versus original 17:15 final-block timing remains unresolved
  and should be confirmed with the teaching team.
- Future collaborator feedback should be handled as a new bounded revision. Do
  not change the existing 10:00 overview pending Ben's edits.

## Notebook state and source of truth

`notebooks/build_notebooks.py` remains authoritative for all six generated
exercise/solution notebooks. The completed notebook critique cycle established:

- Notebook 1 has four timed cores plus Mess3 synthesis, with matrices and
  recursive geometry correctly gated.
- Notebook 2 has a 58-minute mandatory task/visual budget, evaluation vocabulary,
  a pedagogical Mess3-family simulation clearly distinguished from paper data,
  and matched-next-token RRXOR as CORE 5.
- Notebook 3 has a 58-minute core, treats the small binary Hankel dependency
  honestly, separates the adopted continuation from finite-block evidence, and
  makes WFA an 8–10-minute instructor demonstration.

The final full slide build reran these checks successfully:

- `python notebooks/build_notebooks.py --check`
- `python notebooks/validate_notebooks.py`

Before slide work, all six variants were also rendered to fresh preview
directories and their figures were visually inspected.

## Repository state

This is a local Git repository with no commits and no remote. All files are
untracked by design. Do not commit, push, or create a remote without a separate
request. Preserve unrelated files and the original slide PDFs under
`XavOriginal/`.
