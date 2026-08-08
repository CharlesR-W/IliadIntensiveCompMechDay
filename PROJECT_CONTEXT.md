# Iliad Intensive Computational Mechanics Day

## Current direction

This project is a redesign of Xavier Poncini's Summer 2026 Iliad Intensive
computational-mechanics module. The source material is preserved in
`XavOriginal/`.

The intended revision is to:

- remove GHMMs and the extended discussion of minimal/non-unique HMM
  realizations;
- treat HMMs as one convenient realization of an observable stochastic process,
  rather than positing a privileged "true" latent HMM;
- make belief states and their geometry in the hidden-state simplex the
  pedagogical centre;
- use a small amount of Hankel-matrix / predictive-state-representation (PSR)
  material to connect observable word probabilities to finite-dimensional state
  and spectral reconstruction;
- integrate short pen-and-paper checks into the teaching rather than separating
  all exercises into hour-long coding blocks;
- retain an ARENA-like, hint-rich interactive companion if useful, but require
  approximately zero Python knowledge;
- replace or substantially constrain the long open-ended "design an HMM"
  workshop, which prior students found difficult.

Audience floor: mathematically strong bachelor's students; the room may also
include physics PhDs. Scope discipline matters more than maximal coverage.

## What is in `XavOriginal/`

### `[Internal] CompMech Module Updated.md`

This is the curriculum specification and teaching guide, not lecture notes. It
frames computational mechanics as predicting the internal representations of a
near-optimal predictor from both the data-generating process and predictor
architecture. The AI-safety motivation is that optimal prediction is a
scale-free instrumental goal, so predictive representations may remain relevant
as systems become more capable.

The declared mathematical sequence is:

1. HMMs and edge-labelled transition matrices.
2. Minimal HMM realizations and their possible non-uniqueness.
3. GHMMs as a relaxed class with a unique minimal realization up to similarity.
4. Belief states as minimal sufficient statistics for prediction.
5. The mixed-state presentation (MSP), especially for Zero-One-Random (Z1R).
6. Affine/linear maps from transformer activations to belief geometry.
7. An open-ended workshop designing interesting processes.

The original full-day schedule is:

- 10:00–10:30 overview and scope;
- 10:30–11:00 HMM lecture;
- 11:00–12:00 HMM coding exercises;
- 12:00–12:30 prediction lecture;
- 13:30–14:30 belief-state coding exercises;
- 14:30–15:30 transformer-belief-geometry reading and discussion;
- 16:00–17:45 process-design workshop and presentations;
- 17:45–18:00 feedback.

The document's own postmortem says that some participants did not know Python,
suggests language-agnostic or parallel paper/coding tracks, asks for more reading
time, and notes that designing processes from scratch was difficult without
examples. It also says the GHMM generalization can be skipped when short on time.

### `CompMechSlidesSummer26_Pt1.pdf` (19 pages)

This is the high-level "Overview & Scope" lecture. It:

- contrasts statistical mechanics (statistical properties predict behaviour)
  with computational mechanics (computational/predictive structure predicts
  behaviour);
- asks how much of the past a system stores, how it stores it, and how stored
  information affects behaviour;
- narrows the research question to convergent structures relevant to optimal
  prediction;
- motivates optimal prediction as an AI instrumental goal and as a way to reduce
  the search space for internal representations;
- locates the research programme along two axes: richer processes and larger or
  stronger neural predictors;
- shows a hierarchy IID ⊂ HMM ⊂ GHMM ⊂ probabilistic context-free grammars ⊂
  probabilistic Turing machines;
- previews the Shai et al. result as an affine map from transformer residual
  activations to HMM belief states;
- scopes the day to (G)HMM processes, transformers, belief geometry, and process
  design.

### `CompMechSlidesSummer26_Pt2.pdf` (20 pages)

This is almost entirely the material now proposed for removal. It:

- defines a Mealy/edge-emitting HMM with symbol-indexed matrices
  $T^{(x)}_{ij}=P(X=x,S'=j\mid S=i)$;
- derives the word probability
  $P(w)=\eta^{(\emptyset)}T^{(w_1)}\cdots T^{(w_L)}\mathbf 1$;
- works through biased-coin, Z1R, random-random-XOR, and redundant-state
  examples in both diagrams and matrices;
- defines a minimal HMM by number of hidden states;
- demonstrates equivalent realizations by stochastic factorizations;
- states that minimal HMM realizations need not be unique;
- defines GHMMs by relaxing entrywise non-negativity/stochasticity while
  preserving valid non-negative word probabilities and a normalization vector;
- shows similarity transformations preserve the generated process;
- states uniqueness of a minimal GHMM up to similarity and gives a block-walk
  example with signed matrices.

The ordinary HMM definition and word-probability calculation are useful and
load-bearing. The minimality, factorization, and GHMM material consumes most of
the deck but is not needed by the local notebooks.

The internal plan refers to a separate Part 3 prediction slide deck, but that
file is not present locally.

### Part 1 exercise and solution notebooks

`part1_sequence_probabilities_{exercises,solutions}_colab.ipynb` are
self-contained ARENA-style Colab notebooks. They encode an HMM as a tensor with

$$
T[x,i,j]=P(X=x,S'=j\mid S=i)
$$

and build:

1. validation of the transition tensor and initial distribution;
2. unnormalised forward propagation through a word;
3. $P(w)=\eta T^{(w)}\mathbf 1$;
4. $P(x\mid w)=P(wx)/P(w)$;
5. breadth-first enumeration of all reachable next-token distributions.

Each method has an explanation, estimated time, difficulty/importance markers,
hint, hidden/revealed solution, and tests. The demos plot next-token geometry
for Mess3 and custom processes. Much of each notebook is duplicated setup,
testing, and Plotly code, so apparent notebook length overstates the conceptual
content.

### Part 2 exercise and solution notebooks

`part2_belief_states_{exercises,solutions}_colab.ipynb` continue with:

1. the posterior belief
   $\eta^{(w)}=\eta T^{(w)}/(\eta T^{(w)}\mathbf 1)$;
2. recursive Bayesian update
   $\eta\mapsto\eta T^{(x)}/(\eta T^{(x)}\mathbf 1)$;
3. the linear map from belief to next-token probabilities,
   $P(x\mid\eta)=\eta T^{(x)}\mathbf 1$;
4. enumeration of reachable beliefs as a finite-depth mixed-state
   presentation.

The demos make the geometric payoff explicit: Mess3 beliefs form a fractal in a
triangle, Arch beliefs lie in a tetrahedron, and a custom Fern process produces
another fractal. This belief/simplex/update sequence is the strongest existing
spine for the revision.

### `Block 4_ Identifying belief geometry in transformers.md`

This is a 45-minute reading plus 15-minute group discussion.

- Main reading: Shai et al., *Transformers represent belief state geometry in
  their residual stream*. Students focus on the affine map from residual
  activations to HMM belief states, its MSE across training, and a shuffled
  control; explaining Figure 6D is the comprehension check.
- Extension: *Neural networks leverage nominally quantum and post-quantum
  representations*. This depends directly on the GHMM material and argues that
  several neural architectures can learn lower-dimensional GHMM belief
  representations.
- Discussion prompts ask what evidence is most/least convincing, what further
  experiments would help, and what alternative explanations remain.

The main reading survives the proposed redesign. The extension reading should
be removed or made optional because its motivation depends on GHMMs.

## Dependency map and redesign implications

- HMM matrices → word probabilities → forward vector → normalized belief →
  Bayesian update → belief simplex → transformer probing is one clean,
  self-contained chain.
- GHMM uniqueness/minimality is a side branch. Removing it does not break that
  chain.
- An HMM posterior belief is sufficient for predicting the future, but it is not
  automatically the minimal sufficient statistic of the observable process:
  distinct beliefs can induce the same distribution over every future. The
  minimal observable notion instead quotients histories by equality of their
  full future distributions (the causal-state idea).
- A process Hankel matrix can be introduced from the already-familiar word
  probabilities: rows are histories, columns are future tests, and entries are
  joint or conditional future probabilities.
- HMM beliefs factor the history-to-future map. This lets Hankel rank motivate a
  finite predictive state without declaring latent HMM states ontologically
  fundamental.
- A PSR should be presented as a coordinate system made of a small set of future
  predictions, not as a second large formalism. Its role is to contrast
  model-relative beliefs with observable predictive equivalence.
- Finite Hankel rank guarantees a finite-dimensional linear/predictive
  realization. Recovering an entrywise non-negative stochastic HMM requires
  additional realization/identifiability assumptions; spectral methods often
  recover an observable operator model related to an HMM state by an invertible
  linear transform rather than literal HMM parameters.

## Likely capstone direction

Prefer a tightly scaffolded "reverse-engineer the process" task to free HMM
design:

- fill a small prefix/suffix Hankel table from supplied word probabilities;
- identify proportional predictive rows, while distinguishing generic
  finite-block consistency dependencies from evidence about full Hankel rank;
- choose a small predictive basis and verify a few future probabilities;
- interpret supplied symbol-update operators in a short instructor demo, with
  the reconstruction derivation available as stretch;
- compare the resulting predictive coordinates with Bayesian beliefs and with
  what a transformer probe is claimed to recover.

An interactive companion can expose selectors, sliders, plots, answer boxes,
hints, and solutions while keeping all computation hidden. Students should not
need to edit Python; a paper worksheet should remain fully sufficient.

## Current proposed day plan

The concrete schedule and pedagogical proposal sent for collaborator feedback
on 2026-07-29 is recorded in [`DAY_PLAN.md`](DAY_PLAN.md). Its main commitments
are:

- use short lecture introductions followed by ARENA-like guided derivation
  notebooks;
- make almost all student work mathematical rather than programmatic;
- organize the day into roughly 90-minute blocks, revisiting each topic as
  introduction, derivation, and recap or deeper theory;
- treat PSR/Hankel/WFA material as modular, droppable buffer if earlier material
  needs more time;
- retain a more scaffolded version of the process-design activity rather than
  asking students to invent an HMM from scratch.

The critique-revised teaching draft is generated from
[`notebooks/build_notebooks.py`](notebooks/build_notebooks.py):

- [`NOTEBOOK_OUTLINES.md`](NOTEBOOK_OUTLINES.md) gives the compact, instructor-facing
  progression and cut points;
- [`notebooks/`](notebooks/) contains aligned exercise and worked-solution variants for
  HMM/MSP derivations, transformer belief-geometry discovery, and
  Hankel/PSR/WFA derivations;
- no core route requires student-written Python; one NumPy least-squares line
  and a three-parameter HMM design cell are optional stretch activities;
- central figures are embedded as static outputs and supplied plumbing cells are
  collapsed, so the exercise notebooks remain legible without running code;
- the transformer visual is prominently labelled as a teaching simulation, not
  paper data, and its pedagogical Mess3-family parameters are distinguished from
  the paper's parameters;
- the Hankel notebook explicitly treats the rank-two
  $\{\epsilon,0,1\}$ block as a generic binary-alphabet dependency and postpones
  the full-rank-two conclusion until after adopting and factorizing the
  first-order continuation;
- `python notebooks/build_notebooks.py --check` verifies that all six generated
  notebooks exactly match a fresh in-memory build.

## Prepared teaching artifacts

The redesigned day now has six aligned exercise/solution notebooks and six
spoiler-gated micro-decks. The new decks begin with the 10:30 HMM primer and do
not replace the existing 10:00–10:30 overview pending Ben's edits. Offline
presenter HTML, speaker notes, and matching 1600×900-canvas PDF handouts live in
`slides/dist/`; the derivational source of truth remains
`notebooks/build_notebooks.py`. The final WFA material is an optional 8–10-minute
instructor-led word check, not a reconstruction assignment.
