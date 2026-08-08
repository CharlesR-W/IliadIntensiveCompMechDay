# Guided-Derivation Notebook Outlines

These are the quick-look outlines for the three student notebooks. The actual
draft notebooks live in [`notebooks/`](notebooks/), with separate exercise and
worked-solution versions.

## At a glance

| Notebook | Core question | Main example | Hardest core task | Optional extension |
|---|---|---|---|---|
| 1. HMMs and mixed-state presentations | How does an observer turn an HMM and an observed history into a sufficient state for prediction? | Zero-One-Random (Z1R), then Mess3 | Exhibit beliefs with equal next-token predictions and different longer futures, then calculate one Mess3 map composition | Tune a constrained three-state HMM and inspect its belief geometry |
| 2. Discovering belief geometry in transformers | If a transformer has learned predictive state, how could we find it in activations? | Mess3 teaching simulation, then the paper's matched-prediction RRXOR comparison | Design controls and explain why the matched-next-token comparison is more diagnostic than Mess3 alone | Transcribe the probe into one NumPy line or design an intervention |
| 3. Hankel matrices, PSRs, and WFA reconstruction | Can predictive state be defined using only observable word probabilities? | A first-order binary source reconstructed from a finite table | Separate generic small-block dependence from evidence about full Hankel rank, then choose PSR coordinates | Bridge Z1R beliefs to longer tests; interpret a short instructor-led WFA demo |

## Notebook 1 — HMMs and mixed-state presentations

**Target time:** 60–70 minutes through the Mess3 core synthesis.  
**Student coding:** none in the core; the design studio changes three supplied
numerical parameters.  
**End state:** students can derive word probabilities, posterior beliefs,
recursive belief updates, and future predictions, and can explain an MSP as
dynamics in a simplex.

1. **CORE 1/4: reconstruct the generator (10 min).** Read the Z1R diagram,
   reconstruct $T^{(0)}$ and $T^{(1)}$, check normalization and stationarity,
   and calculate one symbol probability.
2. **CORE 2/4: hidden paths to word probabilities (14 min).** Derive
   $\Pr(w)=\pi T^{(w)}\mathbf 1$ and calculate short Z1R words.
3. **CORE 3/4: Bayes as a geometric map (20 min).** Derive the posterior and
   recursive map $F_x$, calculate several beliefs, and inspect reachable points
   and arrows in the simplex.
4. **CORE 4/4: same next token, different future (15 min).** Compare histories
   `01` and `10`, and state precisely why a next-token vector fails only when it
   is the model's sole recursively retained state.
5. **CORE SYNTHESIS: build the Mess3 recursion (10 min).** Read visible Mess3
   matrices, normalize rows to obtain image triangles, calculate
   $\eta^{(01)}$, and answer before the supplied length-two/fractal reveal.
6. **Optional design studio.** Tune a constrained noisy three-state cycle,
   predict how its update maps should change, and test the prediction with the
   supplied visualizer.

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
   parameter variant, compare baselines, and explain why a strong next-token-only
   baseline makes Mess3 alone inconclusive about information beyond the optimal
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

**Target time:** 55–65 minutes through the Hankel/PSR core; 15–25 additional
minutes for the Z1R bridge; 8–10 additional minutes for an instructor-led WFA
demo.  
**Student coding:** none; supplied cells check arithmetic and plot tables.  
**End state:** students can move from observable word probabilities to
predictive equivalence and finite predictive coordinates, and strong students
can interpret observable symbol operators.

1. **CORE 1/5: interrogate the observable table (8 min).** Recover two short word
   probabilities, check stationarity, and propose the simplest first-order
   continuation. A withheld length-four check is stretch.
2. **CORE 2/5: read small-block rank honestly (10 min).** Assemble the
   $\{\epsilon,0,1\}$ block, derive the generic identity
   $H_{\cdot,\epsilon}=H_{\cdot,0}+H_{\cdot,1}$, and use a nonzero minor to show
   only that this block—and therefore a lower bound on full rank—is two.
3. **CORE 3/5: normalize before comparing predictions (10 min).** Distinguish
   proportional joint rows from equal conditional predictive rows.
4. **CORE 4/5: predictive equivalence and its caveat (10 min).** State why finite
   suffix agreement is not full equivalence, then adopt the first-order
   continuation and identify its two recurrent predictive classes.
5. **CORE 5/5: choose predictions as coordinates (12 min).** Use
   $\{\epsilon,0\}$ as a PSR basis, derive representative length-two predictions
   and symbol-conditioned updates, and distinguish linear dimension from affine
   degrees of freedom.
6. **Core synthesis (8 min).** Derive the factorization
   $H_{h,t}=\Pr(h)\,\eta^{(h)}T^{(t)}\mathbf 1$ to connect observable predictive
   rows back to HMM beliefs. The adopted two-state realization gives the missing
   full-rank upper bound; together with the block lower bound it establishes
   exact full rank two for that continuation, not for an arbitrary completion of
   the finite table.
7. **Stretch: Z1R belief ↔ PSR bridge (15–25 min).** Show that tests
   $\{0,00\}$ recover Z1R's three-component belief and distinguish histories
   merged by one-step prediction.
8. **Optional WFA instructor demo (8–10 min).** Receive the basis blocks and
   solved operators, verify one word, and explain why a negative operator entry
   is not a negative transition probability. The derivation $BA_x=B_x$ is a
   stretch disclosure, not assigned matrix inversion.

**Natural cut points:** after CORE 3 (conceptual introduction), after the core
factorization (complete PSR core), after the Z1R bridge, or after the WFA demo.

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
