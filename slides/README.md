# Short-lecture slides

These six spoiler-gated micro-decks frame and synthesize the guided notebooks.
They deliberately do **not** replace the existing 10:00–10:30 overview: the day
plan says to retain that overview pending Ben's changes. The new files begin at
the 10:30 HMM primer.

| Deck | When to show it | Active talk |
|---|---|---:|
| `01_hmm_primer` | before Notebook 1 | 8–10 min |
| `02_hmm_msp_synthesis` | after Notebook 1 CORE 4; Mess3 slides only after its reveal | 12–15 min + questions |
| `03_transformer_primer` | before Notebook 2 | 8–10 min |
| `04_transformer_debrief` | only after Notebook 2 CORE 5 | 3–5 min |
| `05_hankel_psr_primer` | before Notebook 3 | 10–12 min + questions |
| `06_final_synthesis` | after Notebook 3's core factorization; WFA branch only if reached | 12–15 min + optional 8–10 min |

The HTML files are the primary presentation artifacts. They use bundled Reveal.js
assets and MathML, so presenting does not require an internet connection. Press
`S` for speaker notes and `O` for the slide overview. Matching PDFs are assembled
from the live 1600×900 Reveal canvases for review and fallback projection.

## Build and verify

From the project root:

```bash
python slides/build_slides.py
```

The build first checks and validates the six generated notebooks, extracts the
code-native figures by stable notebook cell ID, renders every Quarto deck,
captures each live 1600×900 Reveal canvas, assembles the captures into PDF
handouts, and runs the slide validator.

To validate existing outputs without rebuilding:

```bash
python slides/validate_slides.py
```

To serve presenter mode locally:

```bash
python -m http.server 8765 --directory slides/dist
```

Then open, for example,
`http://127.0.0.1:8765/01_hmm_primer.html`. No local server is required for the
PDFs.

## Source discipline

- `notebooks/build_notebooks.py` remains the technical source of truth.
- Slides orient, gate reveals, and synthesize; derivations stay in the notebooks.
- The final synthesis carries the shared Seneca epigraph and labels the supplied
  condensed wording as adapted from *Moral Letters to Lucilius* 6.5, 90.46, and
  81.13.
- No figures are cropped from the original PDFs or the Shai et al. paper. The
  included diagrams are project-authored notebook outputs or native slide
  elements.
- Transformer empirical claims are restricted to the primary paper: Shai et al.,
  *Transformers Represent Belief State Geometry in their Residual Stream*,
  NeurIPS 2024, [arXiv:2405.15943](https://arxiv.org/abs/2405.15943).
- The GHMM/minimal-realization detour remains out of scope.
