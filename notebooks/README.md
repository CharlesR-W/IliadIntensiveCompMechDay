# Draft notebooks

Each teaching notebook has a student exercise version and an instructor
worked-solution version:

1. [`01_hmms_msps_exercises.ipynb`](01_hmms_msps_exercises.ipynb) /
   [`01_hmms_msps_solutions.ipynb`](01_hmms_msps_solutions.ipynb)
2. [`02_transformer_belief_geometry_exercises.ipynb`](02_transformer_belief_geometry_exercises.ipynb) /
   [`02_transformer_belief_geometry_solutions.ipynb`](02_transformer_belief_geometry_solutions.ipynb)
3. [`03_hankel_psr_wfa_exercises.ipynb`](03_hankel_psr_wfa_exercises.ipynb) /
   [`03_hankel_psr_wfa_solutions.ipynb`](03_hankel_psr_wfa_solutions.ipynb)

The concise teaching overview is
[`../NOTEBOOK_OUTLINES.md`](../NOTEBOOK_OUTLINES.md).

## Editing and validation

The `.ipynb` files are generated. Edit
[`build_notebooks.py`](build_notebooks.py), then regenerate them with:

```bash
python notebooks/build_notebooks.py
```

The build executes all cells and embeds static PNG/text outputs, so the main
visual argument remains visible without running Python. Supplied plumbing cells
are marked collapsed; student-editable cells remain exposed.

Confirm that the generated files exactly match a fresh in-memory build with:

```bash
python notebooks/build_notebooks.py --check
```

Execute every code cell in both variants and check inventory, cell IDs, local
links, numbered core routes, simulation labels, embedded execution counts, and
exercise/solution parity with:

```bash
python notebooks/validate_notebooks.py
```

For visual QA, render all Matplotlib displays from one notebook to a new or
empty temporary directory:

```bash
python notebooks/render_previews.py \
  notebooks/01_hmms_msps_solutions.ipynb \
  /tmp/iliad-notebook-previews
```

The renderer rejects non-empty destinations so previews from different revisions
cannot be mixed accidentally. Repeat it for all six variants when doing final
visual QA.

The notebooks use only Python's standard library, NumPy, and Matplotlib. No core
route requires Python editing. Notebook 2 has one optional least-squares line,
and Notebook 1 has three optional numerical parameters in its design studio.
