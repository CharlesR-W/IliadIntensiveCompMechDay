# Iliad Intensive Computational Mechanics Day

Teaching materials for an intensive day on hidden Markov models and
mixed-state presentations, transformer belief geometry, and
Hankel/PSR/WFA representations.

## Start here

For co-presenters:

1. Read [`DAY_PLAN.md`](DAY_PLAN.md) for the schedule and pacing choices.
2. Read [`NOTEBOOK_OUTLINES.md`](NOTEBOOK_OUTLINES.md) for the route through
   each exercise block.
3. Read [`HANDOFF.md`](HANDOFF.md) for presenter notes, spoiler gates, and the
   current verification state.
4. Present the offline HTML decks from [`slides/dist/`](slides/dist/). Press
   `S` for speaker notes and `O` for the slide overview. Matching PDFs are
   included as handouts and projection fallbacks.

For participants, distribute the three `*_exercises.ipynb` files in
[`notebooks/`](notebooks/). Keep the matching `*_solutions.ipynb` files with the
teaching team until each reveal.

## Materials

| Topic | Student notebook | Worked solutions |
|---|---|---|
| HMMs and mixed-state presentations | [`01_hmms_msps_exercises.ipynb`](notebooks/01_hmms_msps_exercises.ipynb) | [`01_hmms_msps_solutions.ipynb`](notebooks/01_hmms_msps_solutions.ipynb) |
| Transformer belief geometry | [`02_transformer_belief_geometry_exercises.ipynb`](notebooks/02_transformer_belief_geometry_exercises.ipynb) | [`02_transformer_belief_geometry_solutions.ipynb`](notebooks/02_transformer_belief_geometry_solutions.ipynb) |
| Hankel matrices, PSRs, and WFA reconstruction | [`03_hankel_psr_wfa_exercises.ipynb`](notebooks/03_hankel_psr_wfa_exercises.ipynb) | [`03_hankel_psr_wfa_solutions.ipynb`](notebooks/03_hankel_psr_wfa_solutions.ipynb) |

The notebooks contain embedded outputs and can be read without rerunning them.
Running their supplied cells requires Python, NumPy, and Matplotlib.

## Repository map

- [`notebooks/`](notebooks/) — generated exercise and solution notebooks,
  source builder, and validators
- [`slides/`](slides/) — Quarto sources, build tooling, offline presenter HTML,
  and PDF handouts
- [`XavOriginal/`](XavOriginal/) — private source/reference material; do not
  distribute this folder to participants
- [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md) — project goals and provenance

## Rebuild and validate

From the repository root:

```bash
python notebooks/build_notebooks.py --check
python notebooks/validate_notebooks.py
python slides/validate_slides.py
```

To rebuild every notebook and slide artifact, including the offline decks and
PDFs:

```bash
python slides/build_slides.py
```

See [`notebooks/README.md`](notebooks/README.md) and
[`slides/README.md`](slides/README.md) for the full editing and presentation
workflow.
