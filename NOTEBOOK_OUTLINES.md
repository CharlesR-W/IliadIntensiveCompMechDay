# Guided-Derivation Notebook Outlines

These are the quick-look outlines for the three student notebooks. The actual
draft notebooks live in [`notebooks/`](notebooks/), with separate exercise and
worked-solution versions.

## At a glance

| Notebook | Core question | Main example | Hardest core task | Optional extension |
|---|---|---|---|---|
| 1. HMMs and mixed-state presentations | How does an observer turn an HMM and an observed history into a sufficient state for prediction? | Zero-One-Random (Z1R), then Mess3 | Exhibit beliefs with equal next-token predictions and different longer futures, then calculate one Mess3 map composition | Tune a constrained three-state HMM and inspect its belief geometry |
| 2. Discovering belief geometry in transformers | If a transformer has learned predictive state, how could we find it in activations? | Mess3 teaching simulation, then the paper's matched-prediction RRXOR comparison | Design controls and explain why the matched-next-token comparison is more diagnostic than Mess3 alone | Transcribe the probe into one NumPy line or design an intervention |
| 3. Hankel matrices, PSRs, and WFA reconstruction | Can predictive state be defined using only observable word probabilities? | A first-order binary source reconstructed from a finite table | Derive core tests from Hankel rank while separating finite-block evidence from full-process conclusions | Bridge Z1R beliefs to longer tests; reconstruct a two-dimensional WFA |

## Notebook 1 — HMMs and mixed-state presentations

**Target time:** 60–70 minutes through the Mess3 core synthesis: 58 minutes of
timed exercises plus roughly 12 minutes for the primer, supplied visuals, and
transitions.
**Student coding:** none in the core; the optional design studio changes three
supplied numerical parameters.
**End state:** students can derive word probabilities, posterior beliefs,
recursive belief updates, and future predictions, and can explain an MSP as
dynamics in a simplex.

1. **CORE 1/4: reconstruct the generator (10 min).** Read the Z1R diagram,
   reconstruct $T^{(\text{'0'})}$ and $T^{(\text{'1'})}$, form the marginalized
   hidden-state matrix $P$, check normalization and stationarity, and calculate
   one token probability.
2. **CORE 2/4: hidden paths to word probabilities (12 min).** Derive
   $\Pr(w)=\pi T^{(w)}\mathbf 1$ and calculate short Z1R words.
3. **CORE 3/4: Bayes as a geometric map (16 min).** Derive the posterior and
   recursive map $F_x$, calculate several beliefs, and inspect reachable points
   and arrows in the simplex; introduce mixed states and the mixed-state
   presentation explicitly.
4. **CORE 4/4: same next token, different future (12 min).** Compare histories
   '01' and '10', derive a concrete contradiction for a recursively updated
   next-token vector, and state the limited conclusion precisely.
5. **CORE SYNTHESIS: build the Mess3 recursion (8 min).** Read visible Mess3
   matrices, normalize rows to obtain image triangles, calculate
   $\eta^{(01)}$, and answer before the supplied length-two/fractal reveal.
6. **Optional consolidation (5 min).** Reconstruct the complete edge-to-future
   inference pipeline and classify its linear, normalized, and geometric steps.
7. **Optional design studio (12–15 min).** Tune a constrained noisy three-state
   cycle, predict how its update maps should change, and test the prediction
   with the supplied visualizer.

**Natural cut points:** after CORE 2 (word probabilities), after CORE 4
(complete Z1R logic), or after the Mess3 synthesis (full core).

## Notebook 2 — Discovering belief geometry in transformers

**Target time:** 60–70 minutes for the notebook core, leaving room in the
90-minute block for a short introduction and buffer; 8 optional minutes for the
Mess3 paper checkpoint.  
**Student coding:** none in the core; one optional NumPy expression transcribes
the least-squares derivation. All data generation and plotting are supplied.  
**End state:** students can reconstruct the logic of the Shai et al. experiment,
interpret its controls, and distinguish a belief probe from a next-token probe.

1. **CORE 1/5: state the Z1R result cautiously (10 min).** Separate the failure
   of a sole recursive next-token state from any claim that a full-context
   transformer must persistently store an HMM posterior.
2. **CORE 2/5: operationalize the hypothesis (8 min).** Pair each residual
   activation $a(h)$ with its exact HMM belief $b(h)$ and propose the affine
   hypothesis $b(h)\approx Wa(h)+c$.
3. **CORE 3/5: specify the affine probe (10 min).** Add a column of ones, track
   shapes, write the least-squares objective and pseudoinverse solution, and
   choose a meaningful held-out history split. The normal-equation derivation
   and one-line NumPy transcription are stretch tasks.
4. **Control card + CORE 4/5 (10 min).** Define fit and held-out sets, MSE,
   baselines, negative controls, checkpoints, and decodability versus causal
   use; then interpret subtree holdout, shuffled labels, training checkpoints,
   and covariate and next-token baselines.
5. **Core visual (10 min).** Inspect a clearly labelled pedagogical Mess3-family
   parameter variant, compare baselines, and use the full-rank emission map to
   explain why exact next-token probabilities reconstruct its beliefs. Mess3
   alone is therefore inconclusive about information beyond the optimal
   next-token distribution.
6. **Optional paper checkpoint (8 min).** Read the paper's reported Mess3
   evidence, including its random 20/80 splits, and state exactly what affine
   decodability does and does not establish.
7. **CORE 5/5: matched next-token histories (10 min).** Predict the RRXOR layer
   pattern and a diagnostic distance comparison, answer, and only then reveal
   the reported result. This is the core resolution of belief versus
   optimal-next-token-distribution geometry.
8. **Stretch design studio.** Choose a matched-prediction process, belief-direction
   intervention, or objective comparison and specify controls and a falsifier.

All plots generated inside this notebook are labelled **teaching simulation,
not paper data**; the notebook also states that its pedagogical Mess3 parameters
differ from the paper's. It links directly to the primary paper and states the
reported results separately.

**Natural cut points:** after CORE 3 (method only), after the Mess3 visual
(controls understood but the next-token-distribution alternative remains), or
after CORE 5 (complete paper logic).

## Notebook 3 — Hankel matrices, PSRs, and WFA reconstruction

**Target time:** 55 minutes of core tasks, or 60–70 minutes including
orientation, supplied checks, and debrief; 15–25 additional
minutes for the Z1R bridge; 15–20 additional minutes for instructor-led,
guided WFA reconstruction.
**Student coding:** none; supplied cells check arithmetic and plot tables.  
**End state:** students can derive predictive equivalence, Hankel rank, core
tests, and symbol updates from observable word probabilities, then relate the
construction back to HMM beliefs without treating latent states as canonical.

The notebook now opens with a self-contained conceptual primer: histories and
future tests, conditional predictive profiles, the joint Hankel matrix, the
rank/core-test result, and the limited role of the SVD all appear before the
first Hankel construction is assigned.

1. **CORE 1/4: check the observable source (10 min).** Recover two short word
   probabilities, distinguish right-extension consistency from stationarity,
   derive two next-symbol laws, and propose the simplest first-order
   continuation.
2. **CORE 2/4: build a block and read rank honestly (12 min).** Assemble the
   $\{\epsilon,0,1\}$ block, derive the generic identity
   $H_{\cdot,\epsilon}=H_{\cdot,0}+H_{\cdot,1}$, and use a nonzero minor to show
   only that this block—and therefore a lower bound on full rank—is two.
3. **CORE 3/4: rows to predictive states (12 min).** Normalize joint rows,
   distinguish proportional joint rows from equal conditional rows, state why
   finite test agreement is not full equivalence, and then use the adopted
   first-order continuation to prove the two recurrent predictive classes.
4. **CORE 4/4: core tests and updates (13 min).** Derive why independent Hankel
   columns serve as coordinates, use $\{\epsilon,0\}$ as a PSR basis, express all
   length-two predictions in one free coordinate, and derive the normalized
   symbol update.
5. **Core synthesis (8 min).** Reuse Notebook 1's word formula to derive
   $H_{h,t}=\Pr(h)\,\eta^{(h)}T^{(t)}\mathbf 1$ to connect observable predictive
   rows back to HMM beliefs. The adopted two-state realization gives the missing
   full-rank upper bound; together with the block lower bound it establishes
   exact full rank two for that continuation, not for an arbitrary completion of
   the finite table.
6. **Optional A: Z1R belief ↔ PSR bridge (15–25 min).** Show that tests
   $\{0,00\}$ recover Z1R's three-component belief and distinguish histories
   merged by one-step prediction.
7. **Optional B: instructor-led WFA reconstruction (15–20 min).** Construct the
   basis and symbol-shifted blocks, derive $BA_x=B_x$, verify word weights,
   interpret a negative operator entry, and state where truncated SVD helps—and
   what it does not guarantee—with empirical Hankel blocks.

**Natural cut points:** after CORE 2 (Hankel construction), after the core
factorization (complete PSR core), after the Z1R bridge, or after the WFA
reconstruction.

## Cross-notebook design conventions

- Every exercise and worked-solution notebook opens with the same continuous
  Seneca epigraph, explicitly labelled as adapted from *Moral Letters to
  Lucilius* 6.5, 90.46, and 81.13.
- Every mathematical object appears first in prose and a diagram or table, then
  in symbols, then in a small calculation.
- Each exercise has an estimated time, one or two staged hints, a paper-answer
  box, and a worked solution in the instructor version.
- Supplied code is plumbing for pictures and arithmetic checks, not the object
  of assessment.
- The same row-vector convention is used throughout:
  $T^{(x)}_{ij}=\Pr(X=x,S'=j\mid S=i)$.
- Beliefs are explicitly described as relative to a chosen HMM realization;
  equality of full observable future distributions is the process-relative
  predictive notion.
- Hankel rank is presented as guaranteeing a finite-dimensional linear
  realization, not automatically a nonnegative stochastic HMM realization.
