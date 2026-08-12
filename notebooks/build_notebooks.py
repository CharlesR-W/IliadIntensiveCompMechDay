"""Build the guided-derivation exercise and solution notebooks.

The generated notebooks are intentionally standalone Colab-compatible files.
Students do mathematics in Markdown answer cells; supplied Python only checks
arithmetic or produces figures. Re-run this script after editing notebook
content here.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import io
import json
import os
import textwrap
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent


def clean(source: str) -> str:
    return textwrap.dedent(source).strip() + "\n"


def cell_id(kind: str, source: str) -> str:
    return hashlib.sha1(f"{kind}:{source}".encode()).hexdigest()[:12]


def md(source: str, tags: list[str] | None = None) -> dict[str, Any]:
    source = clean(source)
    metadata: dict[str, Any] = {}
    if tags:
        metadata["tags"] = tags
    return {
        "cell_type": "markdown",
        "id": cell_id("md", source),
        "metadata": metadata,
        "source": source.splitlines(keepends=True),
    }


def code(source: str, tags: list[str] | None = None) -> dict[str, Any]:
    source = clean(source)
    actual_tags = ["provided-code"] if tags is None else tags
    metadata: dict[str, Any] = {"tags": actual_tags}
    if "provided-code" in actual_tags:
        metadata["collapsed"] = True
        metadata["jupyter"] = {"source_hidden": True}
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id("code", source),
        "metadata": metadata,
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def response(solution: str, show_solutions: bool) -> dict[str, Any]:
    if show_solutions:
        return md(
            "#### Worked solution\n\n"
            + textwrap.dedent(solution).strip(),
            ["solution"],
        )
    return md(
        """
        #### Your answer

        _Work on paper or edit this cell. Show enough intermediate reasoning to
        locate a disagreement._
        """,
        ["student-answer"],
    )


def seneca_epigraph() -> dict[str, Any]:
    return md(
        """
        > <span lang="la">*longum iter est per praecepta, breve et efficax per exempla. multum interest utrum non velit an nesciat. velle non discitur.*</span>
        >
        > — **Seneca the Younger**, adapted from *Moral Letters to Lucilius* 6.5, 90.46, and 81.13
        """,
        ["epigraph"],
    )


def plato_geometry_epigraph() -> dict[str, Any]:
    return md(
        r"""
        > <span lang="grc">*ἀεὶ ὁ θεὸς γεωμετρεῖ.*</span>
        >
        > “God always geometrizes.”
        >
        > — Saying attributed to **Plato**, reported by **Plutarch**,
        > [*Table Talk* 8.2 (*Moralia* 718B–C)](https://topostext.org/work/297#718B)
        """,
        ["epigraph"],
    )


def heraclitus_hidden_nature_epigraph() -> dict[str, Any]:
    return md(
        r"""
        > <span lang="grc">*φύσις κρύπτεσθαι φιλεῖ.*</span>
        >
        > “Nature loves to hide.”
        >
        > — **Heraclitus**, [fragment DK 22 B123](https://www.greek-language.gr/digitalResources/ancient_greek/anthology/literature/browse.html?text_id=123)
        """,
        ["epigraph"],
    )


def notebook(cells: list[dict[str, Any]], title: str) -> dict[str, Any]:
    for index, item in enumerate(cells):
        source = "".join(item["source"])
        item["id"] = cell_id(f"{item['cell_type']}-{index}", source)
    return {
        "cells": cells,
        "metadata": {
            "colab": {
                "name": title,
                "provenance": [],
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def notebook_1(show_solutions: bool) -> dict[str, Any]:
    variant = "Worked solutions" if show_solutions else "Exercises"
    cells: list[dict[str, Any]] = [
        md(
            f"""
            # 1 — Hidden Markov models and mixed-state presentations

            **{variant} · draft for the Iliad Intensive**

            A small hidden generator can create a surprisingly complicated
            prediction problem. This notebook is a self-contained introduction:
            it begins with the probabilistic definition of a hidden Markov model
            (HMM), develops the matrix notation from that definition, and ends
            with the geometry of its **mixed states**. You will derive the main
            results rather than implement an HMM library.

            By the end, you should be able to:

            - read an edge-emitting HMM diagram and its symbol-labelled matrices;
            - calculate word probabilities by multiplying matrices;
            - derive posterior beliefs and their recursive Bayesian update;
            - turn a belief into probabilities for any future word;
            - explain the mixed-state presentation (MSP) as dynamics in a
              probability simplex;
            - exhibit two beliefs that agree about the next token but disagree
              about the longer future.

            **Route through the notebook:** four numbered **CORE** problems plus
            the Mess3 **CORE SYNTHESIS** form the required path. Their timed
            exercises total 58 minutes, leaving about 12 minutes in a 70-minute
            notebook block for the self-contained primer, supplied visuals, and
            transitions. **OPTIONAL CONSOLIDATION** and the **DESIGN STUDIO** are
            time buffers: skip them without breaking the argument.

            **Working mode.** Most answers are mathematics. Code cells marked
            **visual/check** are supplied plumbing: run them, inspect the result,
            and return to the derivation. No Python knowledge is assumed.
            """
        ),
        seneca_epigraph(),
        md(
            r"""
            ## Hidden Markov models: what is hidden, given, and inferred

            A **hidden Markov model** has a hidden state $S_t$ and an observed
            symbol $X_t$. The state is Markovian: once $S_t$ is known, the
            distribution of the next step does not depend on earlier history.
            We use the **edge-emitting** convention, in which one transition
            both emits the next symbol and changes the hidden state:

            $$
            \Pr(X_{t+1}=x,S_{t+1}=S_j\mid S_t=S_i).
            $$

            Here is the division of labour.

            | Role | Objects |
            |---|---|
            | **Specified by the model** | hidden states, observation alphabet, and all transition/emission probabilities |
            | **Observed by us** | a word $w=x_1\cdots x_L$ of visible symbols |
            | **Calculated by us** | $\Pr(w)$, the posterior distribution of the current hidden state, and probabilities of future words |

            An initial state distribution must also be specified. In this
            notebook we calculate the stationary distribution $\pi$ and use it
            as the initial prior. The hidden state itself is never assumed to be
            observed.

            ### The running process: Zero–One–Random (Z1R)

            The diagram below is the complete probabilistic specification of
            our first HMM. Its visible alphabet is
            $\mathcal A=\{\text{'0'},\text{'1'}\}$. A label
            $\text{'0'}:p$ means “emit the token '0' while traversing this edge,
            with probability $p$.” The code that draws the diagram is supplied
            and collapsed by default; the diagram, not the plotting code, is the
            mathematical object to read.
            """
        ),
        code(
            r"""
            # VISUAL / CHECK — run this cell; there is nothing to edit.
            import itertools
            import numpy as np
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle, FancyArrowPatch

            np.set_printoptions(precision=4, suppress=True)

            def draw_z1r():
                pos = {
                    r"$S_0$": np.array([0.0, 0.0]),
                    r"$S_1$": np.array([2.4, 0.0]),
                    r"$S_R$": np.array([1.2, 1.9]),
                }
                fig, ax = plt.subplots(figsize=(7.2, 4.6))
                for name, xy in pos.items():
                    ax.add_patch(Circle(xy, 0.28, facecolor="#eef3ff",
                                        edgecolor="#27324a", lw=2, zorder=3))
                    ax.text(*xy, name, ha="center", va="center", fontsize=13,
                            zorder=4)

                def arrow(a, b, label, rad=0.0, offset=(0, 0)):
                    patch = FancyArrowPatch(
                        pos[a], pos[b], arrowstyle="-|>", mutation_scale=16,
                        shrinkA=24, shrinkB=24, lw=1.8, color="#3f4d68",
                        connectionstyle=f"arc3,rad={rad}",
                    )
                    ax.add_patch(patch)
                    mid = (pos[a] + pos[b]) / 2 + np.array(offset)
                    ax.text(*mid, label, ha="center", va="center", fontsize=11,
                            bbox=dict(boxstyle="round,pad=.2", fc="white",
                                      ec="none", alpha=.9))

                arrow(r"$S_0$", r"$S_1$", "'0' : 1", offset=(0, -.18))
                arrow(r"$S_1$", r"$S_R$", "'1' : 1", offset=(.18, .10))
                arrow(r"$S_R$", r"$S_0$", r"'0' : $\frac{1}{2}$", rad=.17,
                      offset=(-.22, .12))
                arrow(r"$S_R$", r"$S_0$", r"'1' : $\frac{1}{2}$", rad=-.17,
                      offset=(.08, -.10))
                ax.set(xlim=(-.6, 3.0), ylim=(-.55, 2.45), aspect="equal")
                ax.axis("off")
                ax.set_title(
                    "HIDDEN GENERATOR — node = unseen phase; "
                    "edge label = emitted symbol : probability",
                    fontsize=11,
                )
                plt.show()

            draw_z1r()
            """
        ),
        md(
            r"""
            ### From the diagram to matrices

            Z1R repeatedly emits '0', then '1', then a fair random bit, and
            repeats. At time $t$, $S_t$ is the hidden phase **before** the next
            edge. Traversing that edge emits $X_{t+1}$ and arrives at
            $S_{t+1}$:

            - $S_0$: emit '0', then arrive at $S_1$;
            - $S_1$: emit '1', then arrive at $S_R$;
            - $S_R$: emit a fair random token, then arrive at $S_0$.

            We now **vectorize** the probabilities in the diagram. For each
            possible visible symbol $x$, define one symbol-labelled matrix

            $$
            T^{(x)}_{ij}
            =\Pr(X_{t+1}=x,S_{t+1}=S_j\mid S_t=S_i).
            $$

            Rows mean the phase **before** an edge and columns mean the phase
            **after** it. Thus $T^{(\text{'0'})}$ contains only edges that emit
            '0', and $T^{(\text{'1'})}$ contains only edges that emit '1'.

            A symbol matrix is not generally row-stochastic: it contains only
            the joint events that emit one symbol. After forgetting which symbol
            was emitted, we obtain the ordinary hidden-state transition matrix

            $$
            P=\sum_{x\in\mathcal A}T^{(x)}.
            $$

            It is $P$, not each $T^{(x)}$, whose rows sum to one. We use row
            vectors for state distributions, so one hidden-state step is
            $\mu_{t+1}=\mu_tP$.
            """
        ),
        md(
            r"""
            ## CORE 1/4 — Reconstruct the generator (10 minutes)

            1. Without looking at code, reconstruct the complete $3\times3$
               matrices $T^{(\text{'0'})}$ and $T^{(\text{'1'})}$ from the
               diagram.
            2. In terms of the matrices $T^{(x)}$, write the identity expressing
               that the probabilities of all possible emitted-symbol/destination
               pairs sum to one from each source state. Explain why this does
               **not** require either symbol matrix to have row sums equal to
               one.

               $$
               \sum_{x\in\mathcal A}\sum_j T^{(x)}_{ij}=1
               \quad\text{for each source state }i.
               $$

            3. Consider the hidden-state transition matrix
               $P=T^{(\text{'0'})}+T^{(\text{'1'})}$. Explain what information
               is discarded by this sum, then verify that the uniform row
               vector $\pi=(1/3,1/3,1/3)$ is stationary: $\pi P=\pi$.
            4. Use the diagram or your matrices to calculate
               $\Pr(X_1=\text{'0'})$ under $\pi$, keeping the two hidden-source
               contributions visible.

            <details><summary>Hint 1</summary>
            Each arrow contributes to exactly one entry of exactly one symbol
            matrix.
            </details>

            <details><summary>Hint 2</summary>
            Stationarity is a statement about the state-transition matrix
            $P$, after marginalizing (forgetting) which symbol was emitted.
            </details>
            """
        ),
        response(
            r"""
            Reading each labelled edge gives

            $$
            T^{(\text{'0'})}=
            \begin{pmatrix}
            0&1&0\\
            0&0&0\\
            1/2&0&0
            \end{pmatrix},
            \qquad
            T^{(\text{'1'})}=
            \begin{pmatrix}
            0&0&0\\
            0&0&1\\
            1/2&0&0
            \end{pmatrix}.
            $$

            Summing over every mutually exclusive emitted-symbol/destination
            pair exhausts the possible next edges, which proves the
            normalization identity. For instance, the zero second row of
            $T^{(\text{'0'})}$ is harmless because all of that row's mass sits in
            $T^{(\text{'1'})}$.

            Adding the matrices marginalizes out the emitted token and gives
            $P=\left(\begin{smallmatrix}0&1&0\\0&0&1\\1&0&0\end{smallmatrix}\right)$.
            It cyclically permutes the three coordinates, so
            $\pi P=\pi$. Finally,

               $$
               \Pr(X_1=\text{'0'})
               =\pi T^{(\text{'0'})}\mathbf 1
               =\frac13+\frac16=\frac12.
               $$
               The two contributions are “we were in $S_0$” and “we were in
               $S_R$ and its fair bit was zero”.
            """,
            show_solutions,
        ),
        code(
            r"""
            # POST-ANSWER CHECK / SETUP — matrices are revealed only after CORE 1.
            T0 = np.array([
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
            ])
            T1 = np.array([
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.5, 0.0, 0.0],
            ])
            T = np.stack([T0, T1])
            P_hidden = T0 + T1
            eta_empty = np.ones(3) / 3

            print("T^('0') =\n", T0)
            print("T^('1') =\n", T1)
            print("P = T^('0') + T^('1') =\n", P_hidden)
            print("row sums after adding symbols:", T.sum(axis=(0, 2)))
            """
        ),
        md(
            r"""
            ### Notation card — keep this visible

            | Symbol | Meaning | Shape |
            |---|---|---|
            | $T^{(x)}$ | model-given joint transition/emission matrix for token $x$ | $3\times3$ |
            | $P=\sum_xT^{(x)}$ | hidden-state transition after the token is forgotten | $3\times3$ |
            | $\pi=\eta^{(\epsilon)}$ | stationary initial belief, calculated from $\pi P=\pi$ | $1\times3$ row |
            | $\mathbf 1=(1,1,1)^\top$ | sums over the unknown final hidden state | $3\times1$ column |
            | $T^{(w)}=T^{(x_1)}\cdots T^{(x_L)}$ | all hidden paths that emit $w=x_1\cdots x_L$ | $3\times3$ |
            | $\alpha^{(w)}_j=\Pr(w,S_L=S_j)$ | unnormalized forward row | $1\times3$ |
            | $\eta^{(w)}_j=\Pr(S_L=S_j\mid w)$ | normalized posterior belief | $1\times3$ |

            The superscript $(w)$ labels an observed word; it is not a matrix
            power. From here on, $\epsilon$ denotes the empty history.
            """
        ),
        md(
            r"""
            ## CORE 2/4 — From hidden paths to word probabilities (12 minutes)

            Let $w=x_1x_2\ldots x_L$ be an observed word and write
            $T^{(w)}=T^{(x_1)}T^{(x_2)}\cdots T^{(x_L)}$.

            1. For a two-symbol word $x_1x_2$, begin with a sum over the unknown
               hidden states $s_0,s_1,s_2$. Regroup the factors until you can
               recognize matrix multiplication.
            2. Generalize the calculation to a word
               $w=x_1\ldots x_L$ and show that the forward row obeys

               $$
               \alpha^{(wx)}=\alpha^{(w)}T^{(x)},\qquad
               \Pr(w)=\alpha^{(w)}\mathbf 1
               =\pi T^{(w)}\mathbf 1.
               $$

            3. Use either hidden paths or the matrix formula to calculate
               $\Pr(\text{'010'})$ for Z1R. Use the stationary prior $\pi$, or
               choose another prior and state it explicitly.
            4. Explain what information $\alpha^{(w)}$ retains that the scalar
               $\Pr(w)$ discards.

            <details><summary>Hint 1</summary>
            For fixed $s_0,s_1,s_2$, the path contribution is
            $\eta_{s_0}T^{(x_1)}_{s_0s_1}T^{(x_2)}_{s_1s_2}$.
            </details>

            <details><summary>Hint 2</summary>
            The final multiplication by $\mathbf 1$ sums over the unknown final
            hidden state.
            </details>
            """
        ),
        response(
            r"""
            For two symbols,

            $$
            \begin{aligned}
            \Pr(x_1x_2)
            &=\sum_{s_0,s_1,s_2}
              \eta_{s_0}^{(\emptyset)}
              T^{(x_1)}_{s_0s_1}T^{(x_2)}_{s_1s_2}\\
            &=\eta^{(\emptyset)}T^{(x_1)}T^{(x_2)}\mathbf 1.
            \end{aligned}
            $$

            Repeating the same marginalization at each hidden step gives
            $\alpha^{(w)}=\pi T^{(w)}$ and
            $\alpha^{(wx)}=\alpha^{(w)}T^{(x)}$. Summing its destination-state
            coordinate gives $\Pr(w)=\alpha^{(w)}\mathbf 1$.

            For a general prior $\rho=(\rho_0,\rho_1,\rho_R)$,

            $$
            \rho T^{(\text{'0'})}T^{(\text{'1'})}T^{(\text{'0'})}
            =(\rho_0/2,0,0).
            $$

            Thus $\Pr(\text{'010'})=\rho_0/2$. With the stationary prior
            $\rho=\pi$, this is $1/6$: the process must begin in $S_0$, emit
            the certain prefix '01', and then emit the fair '0' from $S_R$.

            The scalar word probability forgets where a compatible hidden path
            ended. The forward row retains the joint mass assigned to each
            possible final phase, which is exactly what the next update needs.
            """,
            show_solutions,
        ),
        code(
            r"""
            # CHECK — compare the matrix calculation with your arithmetic.
            def forward(word, initial=eta_empty, transition_tensor=T):
                vector = np.asarray(initial, dtype=float)
                for symbol in word:
                    vector = vector @ transition_tensor[int(symbol)]
                return vector

            def word_probability(word, initial=eta_empty, transition_tensor=T):
                return float(forward(word, initial, transition_tensor).sum())

            for word in ["0", "1", "00", "01", "10", "11"]:
                print(f"Pr('{word}') = {word_probability(word):.6f}")
            """
        ),
        md(
            r"""
            ## CORE 3/4 — Bayes as a recursive geometric map (16 minutes)

            After observing $w$, define the **belief state**

            $$
            \eta^{(w)}_j=\Pr(S_{\lvert w\rvert}=S_j\mid X_{1:\lvert w\rvert}=w).
            $$

            In computational mechanics this posterior distribution over hidden
            generator states is called a **mixed state**. “Mixed” means that the
            observer may assign probability to several hidden states; a belief
            concentrated on one state is a **pure state**.

            The **mixed-state presentation (MSP)** keeps every reachable mixed
            state and its weighted Bayesian edges. From belief $\eta$, token $x$
            has probability

            $$
            p_x(\eta)=\eta T^{(x)}\mathbf 1.
            $$

            Write $F_x(\eta)$ for the updated belief after observing $x$.
            When $p_x(\eta)>0$, the MSP contains the edge
            $\eta\xrightarrow{x:p_x(\eta)}F_x(\eta)$. If $p_x(\eta)=0$, that
            observation is impossible from $\eta$, so there is no such edge and
            the normalized update $F_x(\eta)$ is undefined rather than $0/0$.

            1. Use $\alpha^{(w)}$ to derive

               $$
               \eta^{(w)}
               =\frac{\alpha^{(w)}}{\alpha^{(w)}\mathbf1}
               =\frac{\pi T^{(w)}}{\pi T^{(w)}\mathbf1}.
               $$

            2. Now suppose the current belief is an arbitrary row $\eta$ and
               the next observed symbol is $x$. Derive the recursive map

               $$
               F_x(\eta)=\frac{\eta T^{(x)}}
               {\eta T^{(x)}\mathbf1},
               $$

               and interpret its denominator probabilistically.
            3. Calculate $\eta^{(\text{'0'})}$, $\eta^{(\text{'01'})}$, and
               $\eta^{(\text{'10'})}$ by
               composing these maps. Check explicitly that the order matters:
               $F_{\text{'1'}}(F_{\text{'0'}}(\pi))\ne
               F_{\text{'0'}}(F_{\text{'1'}}(\pi))$. For
               $\eta^{(\text{'01'})}$, report all three coordinates explicitly
               as $\Pr(S_2=s\mid\text{'01'})$.
            4. Where do all three-component beliefs live geometrically? Why is
               $F_x$ generally rational rather than linear, even though its
               numerator is linear?

            <details><summary>Hint 1</summary>
            Bayes' rule divides the joint row
            $(\Pr(w,S_L=S_j))_j$ by its total mass.
            </details>

            <details><summary>Hint 2</summary>
            Treat the current posterior as the next step's prior. Compare
            histories '01' and '10', rather than recomputing '01' twice.
            </details>
            """
        ),
        response(
            r"""
            Bayes' rule normalizes the forward row:

            $$
            \eta^{(w)}
            =\frac{\alpha^{(w)}}{\alpha^{(w)}\mathbf1}
            =\frac{\pi T^{(w)}}{\pi T^{(w)}\mathbf1}.
            $$

            Reusing a posterior as the next prior makes the unnormalized
            destination row $\eta T^{(x)}$. Its mass is
            $\eta T^{(x)}\mathbf1=\Pr(X_{\mathrm{next}}=x\mid\eta)$, so dividing
            by it gives $F_x$.

            For '0' and the two orderings,

            $$
            \eta^{(0)}
            =\frac{(1/6,1/3,0)}{1/2}
            =(1/3,2/3,0),\qquad
            \eta^{(01)}=(0,0,1),
            $$

            while

            $$
            \eta^{(1)}=(1/3,0,2/3),\qquad
            \eta^{(10)}=(1/2,1/2,0).
            $$

            In particular,
            $\Pr(S_2=S_0\mid\text{'01'})=0$,
            $\Pr(S_2=S_1\mid\text{'01'})=0$, and
            $\Pr(S_2=S_R\mid\text{'01'})=1$. Thus the two update orders land at
            different points. Every belief
            is nonnegative and sums to one, so beliefs live in the triangular
            simplex $\Delta^2$. Division by the belief-dependent observation
            probability makes $F_x$ rational rather than linear.

            """,
            show_solutions,
        ),
        code(
            r"""
            # VISUAL / CHECK — reachable Z1R beliefs in the hidden-state simplex.
            simplex_vertices = np.array([
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, np.sqrt(3) / 2],
            ])

            def belief(word, initial=eta_empty, transition_tensor=T):
                unnormalized = forward(word, initial, transition_tensor)
                mass = unnormalized.sum()
                if mass <= 1e-14:
                    raise ValueError(f"Impossible history: {word}")
                return unnormalized / mass

            def reachable_beliefs(alphabet, max_depth, initial, tensor):
                result = {"": np.asarray(initial, dtype=float)}
                for depth in range(1, max_depth + 1):
                    for symbols in itertools.product(alphabet, repeat=depth):
                        word = "".join(map(str, symbols))
                        if word_probability(word, initial, tensor) > 1e-14:
                            result[word] = belief(word, initial, tensor)
                return result

            z1r_beliefs = reachable_beliefs([0, 1], 5, eta_empty, T)
            belief_groups = {}
            for word, b in z1r_beliefs.items():
                key = tuple(np.round(b, 10))
                belief_groups.setdefault(key, {"belief": b, "histories": []})
                belief_groups[key]["histories"].append(word)

            # Regression checks for the weighted-edge definition of the MSP.
            assert np.isclose(np.array([1.0, 0.0, 0.0]) @ T1 @ np.ones(3), 0)
            for group in belief_groups.values():
                b = group["belief"]
                edge_probabilities = np.array([
                    b @ symbol_matrix @ np.ones(3)
                    for symbol_matrix in (T0, T1)
                ])
                assert np.isclose(edge_probabilities.sum(), 1)

            fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2))
            triangle = np.vstack([simplex_vertices, simplex_vertices[0]])
            for ax in axes:
                ax.plot(triangle[:, 0], triangle[:, 1],
                        color="#27324a", lw=1.6)
                for label, xy in zip([r"$S_0$", r"$S_1$", r"$S_R$"],
                                     simplex_vertices):
                    ax.text(*(xy + np.array([0.0, -.055])), label,
                            ha="center", va="top", fontsize=11)
                ax.set_aspect("equal")
                ax.axis("off")

            # Left: the state set, uncluttered.
            for group in belief_groups.values():
                b = group["belief"]
                xy = b @ simplex_vertices
                representative = min(
                    group["histories"], key=lambda word: (len(word), word)
                )
                label = (
                    r"$\epsilon$" if representative == ""
                    else f"'{representative}'"
                )
                axes[0].scatter(*xy, s=55, color=b, edgecolor="black",
                                linewidth=.4, zorder=3)
                axes[0].annotate(label, xy, xytext=(4, 4),
                                 textcoords="offset points", fontsize=9)
            axes[0].set_title(
                "Distinct Z1R mixed states\n"
                "RGB color = mass on $(S_0,S_1,S_R)$",
                fontsize=11,
            )

            # Right: only two update paths, so the arrows remain legible.
            for group in belief_groups.values():
                b = group["belief"]
                axes[1].scatter(*(b @ simplex_vertices), s=25, color=b,
                                alpha=.28, linewidth=0, zorder=1)

            symbol_colors = {"0": "#2474a6", "1": "#d6633c"}
            for path in [["", "0", "01"], ["", "1", "10"]]:
                for source_word, target_word in zip(path[:-1], path[1:]):
                    start = belief(source_word) @ simplex_vertices
                    destination = belief(target_word) @ simplex_vertices
                    symbol = target_word[-1]
                    axes[1].add_patch(FancyArrowPatch(
                        start, destination, arrowstyle="-|>",
                        mutation_scale=14, shrinkA=8, shrinkB=8, lw=2.2,
                        color=symbol_colors[symbol],
                        connectionstyle=f"arc3,rad={-.08 if symbol == '0' else .08}",
                        zorder=3,
                    ))
                    midpoint = (start + destination) / 2
                    axes[1].text(*midpoint, f"'{symbol}'", fontsize=11,
                                 color=symbol_colors[symbol],
                                 bbox=dict(boxstyle="round,pad=.12",
                                           fc="white", ec="none", alpha=.9))
                for word in path:
                    xy = belief(word) @ simplex_vertices
                    axes[1].scatter(*xy, s=58, color=belief(word),
                                    edgecolor="black", linewidth=.4, zorder=4)
                    axes[1].annotate(
                    r"$\epsilon$" if word == "" else f"'{word}'",
                        xy, xytext=(4, 4), textcoords="offset points",
                        fontsize=9,
                    )
            axes[1].set_title(
                "Two Bayesian update paths\n"
                "blue = observe '0'; orange = observe '1'",
                fontsize=11,
            )
            fig.suptitle(
                "Z1R posterior beliefs in the probability simplex",
                fontsize=15, fontweight="bold",
            )
            fig.text(
                .5, .015,
                "A point is a posterior over hidden phase, not a physical state "
                "and not merely the last observed token.",
                ha="center", fontsize=10,
            )
            plt.tight_layout(rect=(0, .05, 1, .92))
            plt.show()

            print(f"{len(z1r_beliefs)} reachable histories collapse to "
                  f"{len(belief_groups)} distinct belief states.")
            """
        ),
        md(
            r"""
            The original HMM describes how the hidden world generates data. Its
            **mixed-state presentation** describes the corresponding inference
            process: the states are reachable posterior beliefs, while each
            possible token supplies both an edge probability $p_x(\eta)$ and a
            destination $F_x(\eta)$.

            The plot labels states by histories for convenience, but the state
            itself is the posterior vector. Two histories that yield the same
            posterior land at the same point. Z1R's nearly deterministic cycle
            makes histories merge onto only seven beliefs; a finite-state HMM
            can nevertheless have infinitely many reachable mixed states.
            """
        ),
        md(
            r"""
            ## CORE 4/4 — Same next token, different future (12 minutes)

            From a belief $\eta$, the probability of a future word
            $u=u_1\ldots u_k$ is

            $$
            \Pr(u\mid\eta)=\eta T^{(u)}\mathbf 1.
            $$

            1. Derive the next-token map
               $\Pr(X_{\mathrm{next}}=x\mid\eta)=\eta T^{(x)}\mathbf 1$.
               Then, for a history $h$ and a whole new future word $u$, show
               that, when $\Pr(u\mid h)>0$, the destination-state row before
               normalization is

               $$
               \eta^{(h)}T^{(u)}
               =\Pr(u\mid h)\eta^{(hu)}.
               $$

               Summing this row should recover
               $\Pr(u\mid h)=\eta^{(h)}T^{(u)}\mathbf1$.
            2. Using the beliefs $\eta^{(\text{'01'})}$ and
               $\eta^{(\text{'10'})}$ from CORE 3, show that both give the
               next-token distribution $(1/2,1/2)$.
            3. For each belief, calculate the distribution over
               '00', '01', '10', and '11'. As a separate three-step check,
               calculate the probability that the **next three symbols** are
               '010' after the history '01'.
            4. Imagine compressing every history $h$ to only its next-token
               vector

               $$
               q(h)=\big(\Pr(\text{'0'}\mid h),
               \Pr(\text{'1'}\mid h)\big),
               $$

               and then updating recursively by one fixed rule
               $q(hx)=G(q(h),x)$. The histories '01' and '10' have the same
               $q$. Show that after both histories next emit '1', their
               following-token predictions differ. Why is that impossible for
               such a rule $G$?
            5. State the limited conclusion carefully. Why does this rule out
               next-token probabilities as a sufficient recurrent state, but
               **not** prove that a full-context model literally stores the HMM
               posterior $\eta$?

            <details><summary>Hint 1</summary>
            You already found $\eta^{(\text{'01'})}$. For '10', propagate one token at a
            time and normalize.
            </details>

            <details><summary>Hint 2</summary>
            Under $\eta^{(\text{'01'})}$ the hidden state is certainly $S_R$;
            under $\eta^{(\text{'10'})}$ it is an equal mixture of $S_0$ and
            $S_1$.
            </details>
            """
        ),
        response(
            r"""
            Summing the unnormalized destination distribution gives
            $\Pr(x\mid\eta)=\eta T^{(x)}\mathbf 1$. More generally, when
            $\Pr(u\mid h)>0$, retaining the final hidden-state coordinate gives

            $$
            \Pr(u,S_{|h|+|u|}=S_j\mid h)
            =[\eta^{(h)}T^{(u)}]_j
            =\Pr(u\mid h)\eta_j^{(hu)}.
            $$

            Summing over $j$ gives
            $\Pr(u\mid h)=\eta^{(h)}T^{(u)}\mathbf 1$. If
            $\Pr(u\mid h)=0$, the unnormalized row is the zero row and the
            posterior $\eta^{(hu)}$ is undefined; one must not interpret the
            displayed factorization as zero times an undefined posterior.

            The two beliefs are

            $$
            \eta^{(\text{'01'})}=(0,0,1),\qquad
            \eta^{(\text{'10'})}=(1/2,1/2,0).
            $$

            The first is certainly at $S_R$; the second is equally likely to be
            at $S_0$ or $S_1$. Both emit '0' and '1' with probability $1/2$.
            Their two-step predictions differ:

            $$
            \begin{array}{c|cccc}
            &00&01&10&11\\\hline
            \eta^{(01)}&1/2&0&1/2&0\\
            \eta^{(10)}&0&1/2&1/4&1/4
            \end{array}
            $$

            Starting from $\eta^{(01)}=(0,0,1)$, the first future '0' moves
            from $S_R$ to $S_0$, from which the next symbol '1' is impossible.
            Therefore the probability that the next three symbols are '010'
            after history '01' is
            $\eta^{(01)}T^{(\text{'010'})}\mathbf1=0$.

            Condition once more on observing '1'. After '01', that token takes
            the process from $S_R$ to $S_0$, so the following token is certainly
            '0': $q(\text{'011'})=(1,0)$. After '10', only the $S_1$ component
            can emit '1', and it moves to $S_R$, so
            $q(\text{'101'})=(1/2,1/2)$. A recursive rule $G$ would receive the
            same input state $(1/2,1/2)$ and the same new token '1' in both cases,
            yet would have to return two different outputs. No such $G$ exists.

            This is an insufficiency result about one particular recurrent state
            compression. A full-context model can revisit the whole prefix, and
            another recurrent model may encode the needed distinction in
            coordinates other than the HMM posterior. The posterior $\eta$ is a
            sufficient predictive state for this known HMM, but the calculation
            does not prove it is the unique possible representation.
            """,
            show_solutions,
        ),
        code(
            r"""
            # VISUAL / CHECK — compare the two future distributions.
            eta_01 = belief("01")
            eta_10 = belief("10")
            future_words = ["00", "01", "10", "11"]

            def future_distribution(current_belief, words, tensor=T):
                return np.array([
                    word_probability(w, current_belief, tensor) for w in words
                ])

            next_01 = future_distribution(eta_01, ["0", "1"])
            next_10 = future_distribution(eta_10, ["0", "1"])
            two_01 = future_distribution(eta_01, future_words)
            two_10 = future_distribution(eta_10, future_words)

            assert np.allclose(next_01, next_10)
            assert not np.allclose(two_01, two_10)

            print("belief after '01':", eta_01)
            print("belief after '10':", eta_10)
            print("next-token distributions:", next_01, next_10)

            x = np.arange(len(future_words))
            fig, ax = plt.subplots(figsize=(7.2, 3.8))
            ax.bar(x - .18, two_01, .36, label="after '01'")
            ax.bar(x + .18, two_10, .36, label="after '10'")
            ax.set(xticks=x, xticklabels=[f"'{w}'" for w in future_words],
                   ylim=(0, .58),
                   ylabel="conditional probability",
                   title="Same one-step prediction; different two-step future")
            ax.legend()
            plt.show()
            """
        ),
        md(
            r"""
            **Model-relative caveat.** A belief over the chosen HMM's hidden
            states is sufficient for prediction relative to that realization,
            but it need not be the minimal observable predictive state.
            Distinct beliefs can, in principle, assign the same probability to
            **every** future word;
            an observable predictive representation would merge such beliefs.
            Later notebooks return to that distinction.
            """
        ),
        md(
            r"""
            ## CORE SYNTHESIS — Build the Mess3 recursion (8 minutes)

            Z1R has only seven reachable mixed states. That finiteness is a
            special property of Z1R, not a consequence of having three hidden
            states. We now apply exactly the same inference machinery to Mess3,
            another three-state HMM whose mixed states have a far richer
            geometry. There is no new formalism in this section.

            Mess3 has the visible alphabet
            $\{\text{'0'},\text{'1'},\text{'2'}\}$. Assume we are given its
            three symbol-labelled matrices and a uniform initial belief:

            $$
            T^{(\text{'0'})}=
            \begin{pmatrix}
            .14&.06&.06\\ .03&.28&.06\\ .03&.06&.28
            \end{pmatrix},\qquad
            T^{(\text{'1'})}=
            \begin{pmatrix}
            .28&.03&.06\\ .06&.14&.06\\ .06&.03&.28
            \end{pmatrix},
            $$

            $$
            T^{(\text{'2'})}=
            \begin{pmatrix}
            .28&.06&.03\\ .06&.28&.03\\ .06&.06&.14
            \end{pmatrix},\qquad
            \eta^{(\epsilon)}=(1/3,1/3,1/3).
            $$

            A point $(\eta_0,\eta_1,\eta_2)$ in the belief triangle means that
            the observer assigns probability $\eta_i$ to hidden state $S_i$.
            Its three corners are the **certainty beliefs** $(1,0,0)$,
            $(0,1,0)$, and $(0,0,1)$. For a fixed observed token $x$, update
            each corner with

            $$
            F_x(\eta)=\frac{\eta T^{(x)}}{\eta T^{(x)}\mathbf1}.
            $$

            At a certainty belief this simply normalizes one row of $T^{(x)}$.
            For a certainty belief $e_i$, write
            $r_i=e_iT^{(x)}\mathbf1$ for the probability of token $x$ and
            $v_i=F_x(e_i)$ for the updated corner. For an uncertain belief
            $\eta$,

            $$
            F_x(\eta)=\sum_i\beta_i v_i,
            \qquad
            \beta_i=\frac{\eta_i r_i}{\sum_k \eta_k r_k}.
            $$

            Thus the update uses **likelihood-reweighted** coefficients, not
            generally the original coordinates $\eta_i$. All Mess3 row masses
            are positive, so the possible coefficients $\beta$ fill the simplex
            and the three corner images bound exactly the region of posteriors
            after observing $x$. The first visual shows those three regions; it
            does not reveal any length-two calculations.
            """
        ),
        code(
            r"""
            # CORE VISUAL 1 — one-symbol image triangles only. Run; do not edit.
            from matplotlib.patches import Polygon

            mess3_T = np.array([
                [[.14,.06,.06], [.03,.28,.06], [.03,.06,.28]],
                [[.28,.03,.06], [.06,.14,.06], [.06,.03,.28]],
                [[.28,.06,.03], [.06,.28,.03], [.06,.06,.14]],
            ], dtype=float)
            mess3_initial = np.ones(3) / 3
            map_colors = ["#2474a6", "#d6633c", "#6b5aa6"]
            label_offsets = [np.array([.055, .015]),
                             np.array([-.055, .015]),
                             np.array([0.0, -.045])]

            # Regression check: Bayesian image weights are likelihood-reweighted.
            rows = mess3_T[0]
            row_masses = rows.sum(axis=1)
            corner_images = rows / row_masses[:, None]
            beta = mess3_initial * row_masses / (mess3_initial @ row_masses)
            direct_update = (
                mess3_initial @ rows / (mess3_initial @ rows).sum()
            )
            assert np.allclose(beta @ corner_images, direct_update)
            assert not np.allclose(mess3_initial @ corner_images, direct_update)

            fig, ax = plt.subplots(figsize=(6.7, 5.4))
            ax.plot(triangle[:, 0], triangle[:, 1], color="#27324a", lw=1.3)
            for label, xy in zip([r"$S_0$", r"$S_1$", r"$S_2$"],
                                 simplex_vertices):
                ax.text(*(xy + np.array([0.0, -.055])), label,
                        ha="center", va="top", fontsize=9)

            for symbol in range(3):
                mapped_beliefs = mess3_T[symbol] / (
                    mess3_T[symbol].sum(axis=1, keepdims=True)
                )
                mapped_vertices = mapped_beliefs @ simplex_vertices
                ax.add_patch(Polygon(
                    mapped_vertices, closed=True,
                    facecolor=map_colors[symbol], edgecolor=map_colors[symbol],
                    alpha=.20, lw=2,
                ))
                center = mapped_vertices.mean(axis=0) + label_offsets[symbol]
                ax.text(
                    *center, f"after '{symbol}'",
                    ha="center", va="center", fontsize=10,
                    color=map_colors[symbol], fontweight="bold",
                )

            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(
                "Possible posterior regions after one observed token\n"
                "Each region is bounded by three updated certainty beliefs",
                fontsize=12,
            )
            plt.show()
            """
        ),
        md(
            r"""
            Now do one composition before the reveal.

            1. Normalize the three rows of $T^{(\text{'0'})}$. These are the
               updated beliefs obtained by observing '0' when starting with
               certainty in $S_0$, $S_1$, or $S_2$. Match them to the corners
               of the blue region.
            2. Starting from the uniform belief, calculate
               $\eta^{(\text{'0'})}$ and then
               $\eta^{(\text{'01'})}=F_{\text{'1'}}
               (\eta^{(\text{'0'})})$.
            3. On the one-token visual, identify the region that must contain
               $\eta^{(\text{'01'})}$. Explain why the **last** observed token,
               rather than the first, determines that outer region.
            4. Predict qualitatively what repeated compositions will do.
            5. Does having three hidden generator states constrain the observer
               to only three predictive states?

            <details><summary>Hint</summary>
            For a history '01', apply the '0' update first and the '1' update
            second. Every point produced by the second update lies in the region
            labelled “after '1'.”
            </details>
            """
        ),
        response(
            r"""
            Normalizing the rows of $T^{(\text{'0'})}$ gives

            $$
            (7,3,3)/13,\quad (3,28,6)/37,\quad (3,6,28)/37.
            $$

            From the uniform initial belief,

            $$
            \eta^{(\text{'0'})}=(1/5,2/5,2/5).
            $$

            Propagating through $T^{(\text{'1'})}$ gives an unnormalized row
            $(.104,.074,.148)$, hence

            $$
            \eta^{(\text{'01'})}=(52,37,74)/163
            \approx(.319,.227,.454).
            $$

            Because the '1' update is applied last, this point must lie inside
            the orange region labelled “after '1'.” More generally, histories
            sharing their last token occupy the same outer region; the earlier
            token selects a smaller region within it. Repetition nests these
            images and produces infinitely many mixed states whose closure can
            be fractal. Three hidden generator states fix the two-dimensional
            belief triangle, not the number of reachable posterior beliefs.
            """,
            show_solutions,
        ),
        code(
            r"""
            # CORE VISUAL 2 — reveal the recursive layer and the final geometry.
            mess3_beliefs = reachable_beliefs(
                [0, 1, 2], 7, mess3_initial, mess3_T
            )
            B = np.array(list(mess3_beliefs.values()))
            XY = B @ simplex_vertices

            fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.0))
            for ax in axes:
                ax.plot(triangle[:, 0], triangle[:, 1],
                        color="#27324a", lw=1.3)
                for label, xy in zip([r"$S_0$", r"$S_1$", r"$S_2$"],
                                     simplex_vertices):
                    ax.text(*(xy + np.array([0.0, -.055])), label,
                            ha="center", va="top", fontsize=9)
                ax.set_aspect("equal")
                ax.axis("off")

            depth2_words = list(itertools.product(range(3), repeat=2))
            depth2_B = np.array([
                belief("".join(map(str, word)), mess3_initial, mess3_T)
                for word in depth2_words
            ])
            depth2_XY = depth2_B @ simplex_vertices
            axes[0].scatter(
                depth2_XY[:, 0], depth2_XY[:, 1],
                c=np.clip(depth2_B, 0, 1), s=68,
                edgecolor="black", linewidth=.35,
            )
            for word, xy in zip(depth2_words, depth2_XY):
                axes[0].annotate(
                    "'" + "".join(map(str, word)) + "'",
                    xy, xytext=(4, 3),
                    textcoords="offset points", fontsize=8,
                )
            axes[0].set_title(
                "Posteriors after every two-token history\n"
                "The last token selects the outer region",
                fontsize=11,
            )

            axes[1].scatter(
                XY[:, 0], XY[:, 1], c=np.clip(B, 0, 1),
                s=6, alpha=.72, linewidth=0,
            )
            axes[1].set_title(
                f"Posteriors through length seven ({len(B):,} histories)\n"
                "RGB color = hidden-state belief",
                fontsize=11,
            )
            fig.suptitle(
                "Mess3 posterior beliefs under repeated Bayesian updates",
                fontsize=15, fontweight="bold",
            )
            fig.text(
                .5, .015,
                "Every dot is a posterior belief. Three hidden generator states "
                "do not mean three predictive states.",
                ha="center", fontsize=10,
            )
            fig.subplots_adjust(
                left=.04, right=.98, bottom=.12, top=.84, wspace=.12
            )
            plt.show()
            """
        ),
        md(
            r"""
            ## Optional consolidation — the full pipeline (5 minutes)

            Complete this chain in words, then identify which arrows are linear,
            which require normalization, and which change the interpretation
            rather than the underlying calculation:

            $$
            \text{HMM edges}
            \longrightarrow \alpha^{(w)}
            \longrightarrow \Pr(w)
            \longrightarrow \eta^{(w)}
            \longrightarrow F_x
            \longrightarrow \text{belief geometry}
            \longrightarrow \Pr(u\mid\eta).
            $$
            """
        ),
        response(
            r"""
            Symbol-labelled matrices encode the edges. Their products sum
            hidden paths; retaining the final state gives
            $\alpha^{(w)}=\pi T^{(w)}$, while summing it gives
            $\Pr(w)=\alpha^{(w)}\mathbf1$. Normalizing produces the posterior
            $\eta^{(w)}$, and treating that posterior as the next prior gives the
            weighted MSP edge $\eta\xrightarrow{x:p_x(\eta)}F_x(\eta)$.
            Iterating those edges produces the geometry in the simplex, and
            $\eta T^{(u)}\mathbf1$ gives any future-word probability.

            Matrix propagation and marginalization are linear. Bayesian updates
            are generally nonlinear because their normalizer depends on the
            current belief. The simplex is a geometric view of the posterior
            vectors, not a new probabilistic object.
            """,
            show_solutions,
        ),
        md(
            r"""
            ### Exit ticket

            In two sentences: why can prediction be geometrically more
            complicated than generation, even when the data-generating HMM has
            only three hidden states?
            """
        ),
        md(
            r"""
            ## Optional design studio (12–15 minutes) — tune rather than start from nothing

            The supplied family has a cyclic '0' skeleton:

            $$
            S_0\xrightarrow{\text{'0'}:a}S_1,\quad
            S_1\xrightarrow{\text{'0'}:b}S_2,\quad
            S_2\xrightarrow{\text{'0'}:c}S_0,
            $$

            with a '1' self-loop carrying the remaining probability at each
            state.

            Choose one brief and its starting preset:

            | Brief | Start from $(a,b,c)$ | Operational target |
            |---|---|---|
            | **fast synchronization** | $(.95,.55,.08)$ | median posterior entropy below $0.35$ nats by depth 8 |
            | **persistent ambiguity** | $(.62,.52,.42)$ | median entropy above $0.70$ nats and no single coordinate above $.95$ |
            | **branching geometry** | $(.88,.50,.15)$ | the centroids after final token '0' versus '1' are far apart |

            1. Predict which parameters make an observed '0' most diagnostic.
            2. Run the matching preset and compare the printed metrics with its
               target.
            3. Change **one parameter**, state the direction in which each metric
               should move, and rerun.
            4. Report one failed prediction or one tradeoff between the targets.

            The metrics are only handles for reasoning, not canonical measures
            of geometric complexity.
            """
        ),
        code(
            r"""
            # OPTIONAL DESIGN CELL — pick a preset, then edit one number.
            a, b, c = 0.88, 0.50, 0.15  # branching-geometry preset
            """,
            ["student-parameters"],
        ),
        code(
            r"""
            # PROVIDED VISUALIZER — run after choosing a, b, and c.

            def noisy_cycle(a, b, c):
                assert all(0 <= z <= 1 for z in (a, b, c))
                symbol_0 = np.array([
                    [0, a, 0],
                    [0, 0, b],
                    [c, 0, 0],
                ], dtype=float)
                symbol_1 = np.diag([1-a, 1-b, 1-c])
                return np.stack([symbol_0, symbol_1])

            custom_T = noisy_cycle(a, b, c)
            state_T = custom_T.sum(axis=0)
            evals, evecs = np.linalg.eig(state_T.T)
            custom_initial = np.real(evecs[:, np.argmin(np.abs(evals - 1))])
            custom_initial = custom_initial / custom_initial.sum()

            custom_beliefs = reachable_beliefs(
                [0, 1], 8, custom_initial, custom_T
            )
            C = np.array(list(custom_beliefs.values()))
            Cxy = C @ simplex_vertices

            fig, ax = plt.subplots(figsize=(6.5, 5.7))
            ax.plot(triangle[:, 0], triangle[:, 1], color="#27324a", lw=1.3)
            ax.scatter(Cxy[:, 0], Cxy[:, 1], c=np.clip(C, 0, 1), s=11,
                       alpha=.68, linewidth=0)
            ax.set_title(
                "DESIGN STUDIO — reachable belief geometry\n"
                f"a={a:.2f}, b={b:.2f}, c={c:.2f}; RGB = posterior belief"
            )
            ax.set_aspect("equal")
            ax.axis("off")
            plt.show()

            safe_C = np.clip(C, 1e-15, 1)
            entropies = -(safe_C * np.log(safe_C)).sum(axis=1)
            histories_custom = list(custom_beliefs)
            last_zero = np.array([
                custom_beliefs[w] for w in histories_custom if w.endswith("0")
            ])
            last_one = np.array([
                custom_beliefs[w] for w in histories_custom if w.endswith("1")
            ])
            centroid_gap = np.linalg.norm(
                last_zero.mean(axis=0) - last_one.mean(axis=0)
            )
            print("stationary initial belief:", custom_initial)
            print("median posterior entropy (nats):", np.median(entropies))
            print("largest posterior coordinate:", C.max())
            print("last-symbol centroid separation:", centroid_gap)
            """
        ),
    ]
    return notebook(cells, f"01 HMMs and MSPs — {variant}")


def notebook_2(show_solutions: bool) -> dict[str, Any]:
    variant = "Worked solutions" if show_solutions else "Exercises"
    theta_line = (
        "theta = np.linalg.pinv(A_tiny_aug) @ B_tiny"
        if show_solutions
        else "theta = None  # replace using: np.linalg.pinv(____) @ ____"
    )
    cells: list[dict[str, Any]] = [
        md(
            f"""
            # 2 — A guided discovery of belief geometry in transformers

            **{variant} · draft for the Iliad Intensive**

            This notebook is a **guided discovery**. At each stage, commit to an
            answer before opening the worked solution or reveal. The point is
            to reconstruct the questions that make the experiment in Shai et
            al., *Transformers Represent Belief State Geometry in their
            Residual Stream*, feel necessary rather than merely receive its
            procedure.

            You will:

            - show why a **recursively updated** predictor needs more than
              today's next-token probabilities;
            - separate that theorem from the empirical hypothesis that a
              full-context transformer learns belief-like coordinates;
            - choose what to measure and invent a diagnostic experiment;
            - turn the proposed diagnostic into a shape-checked estimator;
            - predict what cross-validation, label shuffling, and training-time
              comparisons rule out;
            - distinguish belief geometry from next-token geometry.

            **Evidence labels matter.** The supplied plots below use a
            transparent **teaching simulation, not transformer activations and
            not paper data**. Reported empirical results are clearly separated
            and linked to the full paper.

            **Core route (60–70 minutes).** Follow the five numbered CORE tasks,
            run the Mess3 simulation, and stop at the **CORE SYNTHESIS**. Each
            CORE asks for one primary artifact; bullets marked **instructor
            handoff** are checks to discuss, not extra student deliverables.
            Everything after the synthesis is optional and can be skipped
            without breaking the argument.

            | Act | Question | Primary artifact |
            |---|---|---|
            | I | What must survive a one-step prediction collision? | one implication chain |
            | II | What measurement would test the transformer hypothesis? | one activation–target row plus map |
            | III | How do we fit the chosen map without losing track of shapes? | one estimator specification |
            | IV | Which comparisons make a pretty projection evidential? | one control table |
            | V | How do we rule out “it is only the next-token vector”? | one matched-prediction diagnostic |
            """
        ),
        plato_geometry_epigraph(),
        md(
            r"""
            ## ACT I — From a prediction collision to a representation question

            Notebook 1 ended with two histories that agree about the next token
            but disagree later. We will use that collision as a lever. First
            solve a deliberately restricted problem about a recursively updated
            predictor. Then decide which part of the result is a theorem and
            which part becomes a hypothesis about transformers.
            """
        ),
        code(
            r"""
            # PROVIDED SETUP — local access to the paper's source figures.
            from pathlib import Path
            from urllib.request import urlretrieve

            import matplotlib.pyplot as plt

            SHAI_FIGURE_FILES = {
                1: ("mess3_overview.png", "shai_et_al_2024_figure_1.png"),
                3: ("z1rmsp.png", "shai_et_al_2024_figure_3.png"),
                4: ("procedure.png", "shai_et_al_2024_figure_4.png"),
                5: ("main_results.png", "shai_et_al_2024_figure_5.png"),
                6: ("Fig6.png", "shai_et_al_2024_figure_6.png"),
                7: ("rrxor.png", "shai_et_al_2024_figure_7.png"),
            }

            def load_shai_figure(number):
                remote_name, local_name = SHAI_FIGURE_FILES[number]
                candidates = [
                    Path("notebooks/assets") / local_name,
                    Path("assets") / local_name,
                    Path("../notebooks/assets") / local_name,
                ]
                figure_path = next(
                    (path for path in candidates if path.exists()), None
                )
                if figure_path is None:
                    figure_path = Path("/tmp") / local_name
                    if not figure_path.exists():
                        urlretrieve(
                            "https://arxiv.org/html/2405.15943/"
                            f"extracted/6176469/{remote_name}",
                            figure_path,
                        )
                return plt.imread(figure_path)

            def show_shai_figure(number, title, note, figsize=(15.5, 7.0)):
                source_figure = load_shai_figure(number)
                fig, ax = plt.subplots(figsize=figsize)
                ax.imshow(source_figure)
                ax.axis("off")
                ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
                fig.text(
                    .5, .012,
                    f"Source: Shai et al. (2024), Figure {number} · {note}",
                    ha="center", fontsize=9,
                )
                plt.tight_layout(rect=(0, .035, 1, 1))
                plt.show()

            """
        ),
    ]

    # Define the completed paper apparatus here, but reveal it only after
    # students have invented the measurement in CORE 2.
    test_bed_cells = [
        code(
            r"""
            # REVEAL VISUAL — the paper-specific model and probe quantities.
            import numpy as np
            import matplotlib.pyplot as plt
            from matplotlib.patches import (
                FancyBboxPatch, FancyArrowPatch, Rectangle
            )

            fig, ax = plt.subplots(figsize=(15.2, 7.2))
            ax.set(xlim=(0, 15.2), ylim=(0, 7.2))
            ax.axis("off")

            def box(x, y, w, h, text, color="#eef3ff", edge="#27324a",
                    fontsize=9.5, linewidth=1.6):
                ax.add_patch(FancyBboxPatch(
                    (x, y), w, h, boxstyle="round,pad=.04,rounding_size=.08",
                    fc=color, ec=edge, lw=linewidth,
                ))
                ax.text(x+w/2, y+h/2, text, ha="center", va="center",
                        fontsize=fontsize)

            def arrow(x1, y1, x2, y2, label="", color="#47546b",
                      style="-|"):
                ax.add_patch(FancyArrowPatch(
                    (x1, y1), (x2, y2), arrowstyle=style+">",
                    mutation_scale=15, lw=1.6, color=color,
                ))
                if label:
                    ax.text((x1+x2)/2, (y1+y2)/2-.16, label,
                            ha="center", fontsize=8.5, color=color)

            # Data path.
            box(.15, 4.55, 1.45, 1.0,
                r"prefix $h=x_1\cdots x_L$"+"\n"+r"$L\leq10$ tokens")
            box(1.95, 4.55, 1.55, 1.0,
                "token + position\nembeddings")
            arrow(1.60, 5.05, 1.95, 5.05)

            # Four transformer blocks, matching the paper appendix.
            shell = FancyBboxPatch(
                (3.85, 3.55), 6.25, 2.85,
                boxstyle="round,pad=.08,rounding_size=.12",
                fc="#f7f8fc", ec="#27324a", lw=2.0,
            )
            ax.add_patch(shell)
            ax.text(6.98, 6.67,
                    r"4-layer causal transformer · $d_{\rm model}=64$",
                    ha="center", fontsize=11.5, fontweight="bold")
            arrow(3.50, 5.05, 3.95, 5.05)

            layer_outputs = []
            for layer in range(4):
                x = 4.10 + 1.48*layer
                box(x, 4.20, 1.18, 1.62,
                    f"layer {layer+1}\ncausal attention\n"
                    f"1 head · $d_h=8$\n+\nReLU MLP\n$d_{{mlp}}=256$",
                    color="#e8f1ff", fontsize=7.8, linewidth=1.2)
                ax.add_patch(FancyArrowPatch(
                    (x+.16, 5.96), (x+1.02, 5.96),
                    connectionstyle="arc3,rad=-.34", arrowstyle="-|>",
                    mutation_scale=12, lw=1.25, color="#6d7890",
                ))
                ax.text(x+.59, 6.12, "residual", ha="center",
                        fontsize=7.4, color="#59657d")
                layer_outputs.append(x+1.18)
                if layer < 3:
                    arrow(x+1.18, 5.01, x+1.48, 5.01)

            ax.text(6.98, 3.63,
                    "Each block processes every prefix position; taps below "
                    "refer to the last position only.",
                    ha="center", fontsize=8.2, color="#59657d")

            # Current-token output path.
            box(10.48, 4.62, 1.48, .86,
                r"final residual"+"\n"+r"$a_4(h)\in\mathbb{R}^{64}$",
                color="#dff3ec", edge="#28765c")
            box(12.35, 4.62, 1.15, .86,
                "final\nLayerNorm", color="#eef3ff")
            box(13.88, 4.62, 1.15, .86,
                "unembedding\n+ softmax", color="#eef3ff")
            arrow(10.02, 5.05, 10.48, 5.05)
            arrow(11.96, 5.05, 12.35, 5.05)
            arrow(13.50, 5.05, 13.88, 5.05)
            ax.text(14.45, 4.30,
                    r"$p(x_{L+1}\mid h)$", ha="center", fontsize=10.5)

            # Read-only diagnostic branch.
            box(10.48, 2.45, 1.48, .86,
                "affine probe\n"+r"$\widehat b=Wa_4+c$",
                color="#fff2d8", edge="#a26712")
            box(12.45, 2.45, 1.52, .86,
                r"$\widehat b(h)\in\mathbb{R}^{3}$"+"\nrecovered belief",
                color="#fff2d8", edge="#a26712")
            arrow(11.22, 4.62, 11.22, 3.31, "read only",
                  color="#a26712")
            arrow(11.96, 2.88, 12.45, 2.88, color="#a26712")
            ax.text(11.22, 2.20,
                    r"$W\in\mathbb{R}^{3\times64},\ c\in\mathbb{R}^{3}$",
                    ha="center", fontsize=8.5, color="#7b4b0b")

            # Exact label comes from the HMM, never from the forward pass.
            box(.55, .58, 3.05, 1.28,
                "exact HMM label\n"+
                r"$b(h)=\Pr(S_L\mid h)\in\Delta^2$"+"\n"+
                r"$\Pr(u\mid h)=b(h)T^{(u)}\mathbf{1}$",
                color="#e8f6e8", edge="#3d7b45", fontsize=9.2)
            arrow(3.60, 1.22, 10.48, 2.63,
                  r"fit $W,c$ by training MSE", color="#3d7b45")
            ax.text(4.18, .48,
                    "Ground truth supervises the diagnostic probe; held-out "
                    "pairs evaluate the fitted map. Neither is shown to the "
                    "transformer.",
                    ha="left", fontsize=8.5, color="#2f6336")

            # Multi-layer quantity used for RRXOR.
            ax.add_patch(Rectangle((5.02, 2.20), 3.70, .75,
                                   fc="#f4eafb", ec="#76509a", lw=1.4))
            ax.text(6.87, 2.57,
                    r"RRXOR: concat $[a_1;\ldots;a_4]\in\mathbb{R}^{256}$",
                    ha="center", va="center", fontsize=8.8,
                    color="#5c3b7b")
            for output_x in layer_outputs:
                ax.plot([output_x, output_x], [3.55, 2.95],
                        color="#76509a", lw=1.0, ls="--")

            ax.set_title(
                "THE ACTUAL TEST BED — where every quantity in the probe lives",
                fontsize=15.5, fontweight="bold", pad=10,
            )
            fig.text(
                .5, .015,
                "Paper architecture: context 10, four layers, one attention "
                "head per layer, d_model=64, d_head=8, d_mlp=256.",
                ha="center", fontsize=9,
            )
            plt.tight_layout(rect=(0, .04, 1, .96))
            plt.show()
            """
        ),
        code(
            r"""
            # SOURCE VISUAL — Shai et al. (2024), Figure 4.
            show_shai_figure(
                4,
                "THE PAPER'S MEASUREMENT PIPELINE — activations to belief simplex",
                "A: residual-stream taps · B: activation cloud · "
                "C: supervised affine projection · D: decoded geometry",
                figsize=(16.0, 5.3),
            )
            """
        ),
        md(
            r"""
            The source schematic abstracts away the exact layer sizes shown in
            the first diagram, but it makes the geometric operation vivid: color
            each high-dimensional activation by the independently calculated
            belief for the same prefix, then fit one affine projection that
            preserves those labels. See [Shai et al. (2024), Figure
            4](https://arxiv.org/html/2405.15943#S2.F4).

            ### Object and notation card

            - $h=x_1\cdots x_L$: the entire token prefix presented to the model.
            - $a_\ell(h)\in\mathbb R^d$: the residual-stream vector at the
              **last token position**, after layer $\ell$. When one layer is
              fixed, we abbreviate this to $a(h)$;
            - in the paper's main Mess3 analysis, the probe reads the final
              residual **before** the final LayerNorm; LayerNorm and the
              **unembedding** then map that residual to current next-token
              logits;
            - $b(h)\equiv\eta^{(h)}\in\Delta^{m-1}$: Notebook 1's exact
              posterior over the $m$ hidden states of the chosen generator,
              renamed to match the probe notation here;
            - a **probe** is a separate diagnostic map fit after training. It is
              not part of the transformer's forward pass.

            A transformer with the full prefix can recompute information from
            earlier tokens at every position. Therefore the preceding recursive
            argument does not force it to maintain one state. The HMM belief $b(h)$
            is a generator-relative sufficient coordinate: it determines future
            probabilities for the chosen HMM, but it need not be a minimal
            observable predictive state if two beliefs induce the same full
            future law. Notebook 3 will make that model-free distinction
            explicit. Whether the transformer learns coordinates affinely
            related to this chosen belief is an empirical question.
            """
        ),
    ]

    cells.extend([
        md(
            r"""
            ## The discovery puzzle

            A transformer is trained only to minimize next-token cross-entropy.
            That sounds like a one-step job. Why should it create anything as
            rich as a full belief state?

            To isolate the pressure created by repeated prediction, CORE 1 first
            studies a predictor constrained to carry one recursively updated
            state. **This is not yet a theorem about transformers.** A
            full-context transformer can attend back to the prefix and recompute
            information instead of updating one state by the rule below. The
            recursive argument will motivate a measurement; only data can tell
            us whether transformer activations realize the suggested geometry.

            Begin one rung above Notebook 1. Take its Z1R result as given; do
            **not** recompute the beliefs or the two-step table:

            $$
            \eta^{(01)}=(0,0,1),\qquad
            \eta^{(10)}=(1/2,1/2,0).
            $$

            Both predict the next token as $(1/2,1/2)$. Yet if the common next
            token is `1`, their following predictions split:

            $$
            q(\text{'011'})=(1,0),\qquad
            q(\text{'101'})=(1/2,1/2).
            $$

            The puzzle is no longer to verify this collision. It is to ask what
            a predictor must carry **through** the collision if the same system
            is to keep predicting correctly at the next position, and the next.
            """
        ),
        code(
            r"""
            # DISCOVERY VISUAL — a one-step collision that later has to split.
            import matplotlib.pyplot as plt
            from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

            fig, ax = plt.subplots(figsize=(11.8, 4.5))
            ax.set(xlim=(0, 11.8), ylim=(0, 4.5))
            ax.axis("off")

            def state_box(x, y, text, color, edge):
                ax.add_patch(FancyBboxPatch(
                    (x, y), 2.35, .92,
                    boxstyle="round,pad=.04,rounding_size=.08",
                    fc=color, ec=edge, lw=1.7,
                ))
                ax.text(x+1.175, y+.46, text, ha="center", va="center",
                        fontsize=10.5)

            def step(x1, y1, x2, y2, label):
                ax.add_patch(FancyArrowPatch(
                    (x1, y1), (x2, y2), arrowstyle="-|>",
                    mutation_scale=16, lw=1.7, color="#47546b",
                ))
                ax.text((x1+x2)/2, (y1+y2)/2+.16, label,
                        ha="center", fontsize=9.5, color="#47546b")

            state_box(.35, 2.75,
                      "history '01'\n"+r"$q=(1/2,1/2)$",
                      "#eef3ff", "#355e91")
            state_box(.35, .70,
                      "history '10'\n"+r"$q=(1/2,1/2)$",
                      "#eef3ff", "#355e91")
            state_box(4.15, 1.73,
                      "same proposed state\n"+r"$z=q=(1/2,1/2)$",
                      "#fff2d8", "#a26712")
            state_box(8.65, 2.75,
                      "after '011'\n"+r"$q=(1,0)$",
                      "#e8f6e8", "#3d7b45")
            state_box(8.65, .70,
                      "after '101'\n"+r"$q=(1/2,1/2)$",
                      "#fbe8e6", "#a64b42")

            step(2.70, 3.21, 4.15, 2.42, "compress")
            step(2.70, 1.16, 4.15, 2.00, "compress")
            step(6.50, 2.35, 8.65, 3.13, "observe the same token '1'")
            step(6.50, 2.03, 8.65, 1.16, "observe the same token '1'")

            ax.text(5.33, .18,
                    "Can one deterministic update rule take the same state and "
                    "same token to two different predictions?",
                    ha="center", fontsize=10.5, fontweight="bold")
            ax.set_title(
                "THE COLLISION — correct now is not enough to stay correct",
                fontsize=15, fontweight="bold", pad=10,
            )
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            r"""
            ## CORE 1/5 — What must a recursive predictor remember? (12 minutes)

            Suppose a predictor compresses a history to a state $z(h)$. From
            that state it reads out today's next-token distribution and, after
            observing token $x$, updates for tomorrow:

            $$
            q(h)=D(z(h)),\qquad z(hx)=U(z(h),x).
            $$

            Work from the collision diagram rather than repeating Notebook 1.

            1. Try the tempting shortcut $z(h)=q(h)$. The two blue boxes then
               give the update rule $U$ exactly the same two inputs. What must a
               deterministic $U$ return? Why do the two boxes on the right make
               that impossible?
            2. Now let $z$ be arbitrary. If $z(h)=z(h')$, apply the same update
               repeatedly to show that the model must make the same later
               next-token prediction after $hu$ and $h'u$ for **every** possible
               continuation $u$. Use that result to complete one chain:

               > next-token loss at every prefix
               > $\Rightarrow$ correct again after every continuation
               > $\Rightarrow$ ______ must survive in $z(h)$
               > $\Rightarrow$ histories may merge only when ______.

               State the contrapositive in words.
            3. Notebook 1 supplied one update-ready state for a known HMM:
               $b(h)=\Pr(S_{|h|}\mid h)$. Without re-deriving Bayes' rule, write
               the two formulas showing (i) how $b$ updates after $x$, whenever
               the observed symbol $x$ has positive conditional probability,
               and (ii) how it answers a query about any future word $u$.
            4. Sort these statements into **forced by the recursive setup**,
               **natural empirical hypothesis**, or **not implied**:

               - future-distinct histories cannot share the same $z$;
               - the HMM belief is a sufficient choice of $z$;
               - a full-context transformer must store the exact HMM belief in
                 the coordinates chosen by the HMM designer.

               **Primary artifact:** your completed implication chain plus this
               one-sentence bridge to an experiment: “Because predictive
               training needs ______, a last-position activation $a_\ell(h)$
               may contain ______.”

               **Instructor handoff:** compare the three labels and identify
               exactly where the theorem ends.

            <details><summary>Hint</summary>
            Apply the same update $U(\cdot,u_1)$ to equal states, then repeat for
            $u_2,u_3,\ldots$. The readout $D$ cannot split states that remain
            equal.
            </details>
            """
        ),
        response(
            r"""
            With $z=q$, both histories supply the pair
            $((1/2,1/2),\text{'1'})$ to $U$. A deterministic function must
            return the same updated state, and $D$ must then return the same
            prediction. But correctness requires $(1,0)$ after '011' and
            $(1/2,1/2)$ after '101'. The current next-token vector therefore
            cannot be the whole update-ready state.

            More generally, if $z(h)=z(h')$, then for the first symbol $u_1$,

            $$
            z(hu_1)=U(z(h),u_1)=U(z(h'),u_1)=z(h'u_1).
            $$

            Repeating the argument gives $z(hu)=z(h'u)$ for every continuation
            $u$, so $D$ gives the same later next-token distributions. In
            contrapositive form: if some continuation eventually requires
            different predictions, an update-only state must distinguish the
            histories **before** that continuation arrives.

            The completed chain is: loss at every prefix requires the predictor
            to remain correct after every continuation; therefore every
            distinction that can change a later prediction must survive in
            $z(h)$; histories may merge only when they induce the same complete
            future law. Storing $q(h)$ answers one question. An update-ready
            predictive state must preserve enough information to answer the
            whole branching family of later questions.

            For the known HMM, the belief does this because, whenever the
            observed symbol $x$ has positive conditional probability,

            $$
            b(hx)=\frac{b(h)T^{(x)}}{b(h)T^{(x)}\mathbf1},\qquad
            \Pr(u\mid h)=b(h)T^{(u)}\mathbf1.
            $$

            Future-distinct histories being separated is forced by the stated
            recursive setup. The HMM belief is a sufficient choice, so it is a
            natural object to look for. Exact HMM coordinates in a full-context
            transformer are not forced: attention can revisit the prefix, and
            an equally sufficient state can use different or nonlinear
            coordinates. A suitable bridge is: “Because predictive training
            needs future-relevant distinctions to remain available, a
            last-position activation may contain a compact predictive state
            affinely related to $b(h)$.” That final step is the empirical bet we
            will test, not a theorem smuggled in from the objective.
            """,
            show_solutions,
        ),
        code(
            r"""
            # SYNTHESIS VISUAL — the chain just derived, with the logical hinge marked.
            fig, ax = plt.subplots(figsize=(14.2, 4.9))
            ax.set(xlim=(0, 14.2), ylim=(0, 4.9))
            ax.axis("off")

            chain = [
                ("next-token loss\nat every prefix", "#eef3ff", "#355e91"),
                ("must stay correct\nafter continuations", "#eef3ff", "#355e91"),
                ("preserve every distinction\nthat changes a later prediction",
                 "#fff2d8", "#a26712"),
                ("update-ready\npredictive state", "#e8f6e8", "#3d7b45"),
                ("for a known HMM:\nbelief $b(h)$ is sufficient",
                 "#e8f6e8", "#3d7b45"),
            ]
            xs = [.15, 2.95, 5.75, 8.75, 11.45]
            widths = [2.15, 2.15, 2.45, 2.05, 2.55]

            for index, ((label, face, edge), x, width) in enumerate(
                zip(chain, xs, widths)
            ):
                ax.add_patch(FancyBboxPatch(
                    (x, 2.05), width, 1.15,
                    boxstyle="round,pad=.05,rounding_size=.08",
                    fc=face, ec=edge, lw=1.7,
                ))
                ax.text(x+width/2, 2.625, label, ha="center", va="center",
                        fontsize=9.5)
                if index < len(chain)-1:
                    ax.add_patch(FancyArrowPatch(
                        (x+width, 2.625), (xs[index+1], 2.625),
                        arrowstyle="-|>", mutation_scale=15, lw=1.5,
                        color="#47546b",
                    ))

            ax.plot([.15, 10.80], [1.48, 1.48], color="#355e91", lw=2.2)
            ax.text(5.47, 1.13,
                    "logical pressure for any recursively updated predictor",
                    ha="center", fontsize=9.5, color="#355e91")
            ax.plot([11.45, 14.0], [1.48, 1.48], color="#8c5a13", lw=2.2)
            ax.text(12.72, 1.13,
                    "generator-relative coordinate",
                    ha="center", fontsize=9.5, color="#8c5a13")
            ax.text(12.72, .48,
                    "Does a transformer choose this geometry?  That is now an experiment.",
                    ha="center", fontsize=10.2, fontweight="bold")
            ax.set_title(
                "ONE-STEP OBJECTIVE, MULTI-STEP REPRESENTATIONAL BURDEN",
                fontsize=15, fontweight="bold", pad=10,
            )
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            r"""
            ## ACT II — Turn the hypothesis into a measurement

            CORE 1 established a necessity result only for the stipulated
            recursive predictor. For a transformer, the claim that belief-like
            information appears in a particular activation is an empirical bet.
            The next task is to decide what observation would make that bet
            testable—without yet seeing the paper's answer.

            ## CORE 2/5 — Invent the experiment before seeing the paper's version (8 minutes)

            You have a known HMM, a transformer trained only on its emitted
            strings, and permission to record activations. The transformer was
            never shown hidden states or beliefs. Design the smallest experiment
            that could reveal whether it nevertheless discovered the HMM's
            belief geometry.

            1. Make one row of the proposed dataset. What observed history is
               fed to the transformer? Which activation do you record? Which
               exact quantity can you calculate independently from the HMM?
            2. Consider three targets: the most likely hidden state, the current
               next-token vector $q(h)$, and the full belief $b(h)$. Which one
               actually tests the geometric hypothesis motivated by CORE 1?
               What information do the other two discard?
            3. Draw the diagnostic arrow and annotate it with the simplest map
               that can undo rotation, scaling, and translation without
               arbitrarily memorizing every point. Write its equation with $W$
               and $c$, including shapes for $a(h)\in\mathbb R^d$ and an
               $m$-state HMM.
            4. Suppose the map succeeds on new histories. Finish the sentence
               as cautiously as possible: “This would show ______; it would not
               yet show ______.”

            **Primary artifact:** one labelled dataset row and one diagnostic
            arrow. The target choice, map family, direction, and inference
            should all be visible on that sketch.

            <details><summary>Hint</summary>
            Pair two views of the **same history**. The exact label comes from
            Bayesian filtering in the known generator, not from the model.
            </details>
            """
        ),
        response(
            r"""
            One dataset row pairs the last-position residual activation at a
            chosen layer,

            $$
            a_\ell(h_i)\in\mathbb R^d,
            $$

            with the exact filtered HMM belief for that same observed prefix,

            $$
            b(h_i)=\Pr(S_{|h_i|}\mid h_i)\in\Delta^{m-1}.
            $$

            The most-likely-state label collapses a continuous simplex into
            decision regions. The current $q(h)$ retains probabilities but can
            merge states that require different later predictions, as CORE 1
            showed. The full belief retains an update-ready coordinate for the
            chosen generator, so it is the diagnostic target.

            The arrow must run from activation to belief: if a simple map reads
            $b(h)$ from $a_\ell(h)$, the information is decodable from the
            transformer's representation. An affine map is the first useful
            test because it can undo a rotated, rescaled, and translated copy of
            the simplex while imposing one shared global correspondence. With
            column-vector notation,

            $$
            b(h)\approx Wa(h)+c,\qquad
            W\in\mathbb R^{m\times d},\quad c\in\mathbb R^m.
            $$

            Success on genuinely held-out histories would show affine
            **decodability** of the chosen HMM belief geometry from those
            activations. It would not yet show that the transformer uses those
            coordinates causally, that they are the unique predictive
            coordinates, or that every HMM realization of the same process
            would yield the same target.
            """,
            show_solutions,
        ),
        md(
            r"""
            ### Reveal — compare your experiment with the paper's

            You have now chosen the two objects and the direction of the map.
            Only now uncover the completed apparatus below. Trace each arrow
            and identify where your design agrees with it, where the exact HMM
            supplies supervision, and which quantities the transformer itself
            never receives.
            """
        ),
        md(
            r"""
            ### Reveal 1 — the geometric target

            The paper makes the full posterior belief the target. The first
            image below is the particular fractal arrangement of exact Mess3
            beliefs inside the three-state simplex; the second is the HMM that
            generates the training strings. Each point is
            $b(h)=\Pr(S_{|h|}\mid h)$ for one observed prefix, and its RGB color
            is that same probability vector.

            The paper's Mess3 parameters differ from the pedagogical
            Mess3-family parameters in Notebook 1 and in the later simulation.
            """
        ),
        code(
            r"""
            # SOURCE VISUAL — Shai et al. (2024), Figure 5, panels B and A.
            paper_figure = load_shai_figure(5)

            fig, ax = plt.subplots(figsize=(8.4, 6.2))
            ax.imshow(paper_figure[:, 610:1328])
            ax.axis("off")
            ax.set_title(
                "THE TARGET — exact Mess3 belief states in the simplex",
                fontsize=14, fontweight="bold", pad=10,
            )
            fig.text(
                .5, .015,
                "Source: Shai et al. (2024), Figure 5B · "
                "color and position both encode the exact belief b(h)",
                ha="center", fontsize=9,
            )
            plt.tight_layout(rect=(0, .04, 1, 1))
            plt.show()

            fig, ax = plt.subplots(figsize=(6.2, 6.2))
            ax.imshow(paper_figure[:, :610])
            ax.axis("off")
            ax.set_title(
                "THE GENERATOR — the Mess3 edge-emitting HMM",
                fontsize=14, fontweight="bold", pad=10,
            )
            fig.text(
                .5, .015,
                "Source: Shai et al. (2024), Figure 5A · "
                "hidden states S1, S2, S3; emitted tokens A, B, C",
                ha="center", fontsize=9,
            )
            plt.tight_layout(rect=(0, .04, 1, 1))
            plt.show()
            """
        ),
        md(
            r"""
            Figure 3 supplies the conceptual bridge that justifies this target:
            generator $\rightarrow$ mixed-state updater $\rightarrow$ simplex
            $\rightarrow$ reachable belief geometry. Locate the beliefs labelled
            `01` and `10` in panel D. They remain distinct even though their
            current next-token distributions agree.
            """
        ),
        code(
            r"""
            # SOURCE VISUAL — Shai et al. (2024), Figure 3.
            show_shai_figure(
                3,
                "FROM GENERATION TO PREDICTION — the paper's Z1R construction",
                "A: generator · B: mixed-state presentation · "
                "C: simplex coordinates · D: belief geometry",
                figsize=(16.0, 4.8),
            )
            """
        ),
        md(
            r"""
            Source and full context: [Shai et al. (2024), Figure
            3](https://arxiv.org/html/2405.15943#S2.F3) and [Figure
            5](https://arxiv.org/html/2405.15943#S3.F5).

            ### Reveal 2 — the paper's apparatus

            Now compare your row and arrow with the actual diagnostic pipeline.
            The first diagram makes every quantity explicit; the source figure
            then shows the same operation geometrically.
            """
        ),
        *test_bed_cells,
        md(
            r"""
            ## ACT III — Make the proposed map operational

            The experiment is now specified conceptually. The only remaining
            question is whether its algebra is unambiguous enough that someone
            else could fit exactly the same map.

            ## CORE 3/5 — Specify the affine belief probe (10 minutes)

            Stack the activation row vectors into
            $A\in\mathbb R^{n\times d}$ and beliefs into
            $B\in\mathbb R^{n\times m}$. Add a column of ones:

            $$
            \widetilde A=
            \begin{pmatrix}
            --- & a(h_1)^\top & --- & 1\\
            &\vdots&&\vdots\\
            --- & a(h_n)^\top & --- & 1
            \end{pmatrix}
            \in\mathbb R^{n\times(d+1)}.
            $$

            1. Combine $W$ and $c$ into one parameter matrix $\Theta$. Give the
               shapes of $\widetilde A$, $\Theta$, $B$, and
               $\widehat B=\widetilde A\Theta$; check that they compose.
            2. Write the least-squares objective using the Frobenius norm, and
               explain what one summand measures geometrically.
            3. The minimum-norm solution is
               $\Theta^\star=\widetilde A^+B$. Explain why the pseudoinverse is
               preferable to assuming $\widetilde A^\top\widetilde A$ is
               invertible.
            4. A random history split can place near-duplicate prefixes on both
               sides. Propose a held-out split that tests extrapolation across
               a meaningful history family.

            **Primary artifact:** one shape-checked line containing
            $\widehat B$, the least-squares objective, and $\Theta^\star$, plus
            one sentence naming your held-out unit.

            <details><summary>Hint 1</summary>
            With examples stored as rows, the prediction is
            $\widehat B=\widetilde A\Theta$.
            </details>

            """
        ),
        response(
            r"""
            Store $W^\top$ in the first $d$ rows of $\Theta$ and $c^\top$ in its
            final row. The complete shape check is

            $$
            \widetilde A\in\mathbb R^{n\times(d+1)},\qquad
            \Theta\in\mathbb R^{(d+1)\times m},\qquad
            B,\widehat B\in\mathbb R^{n\times m}.
            $$

            The inner dimensions agree, so

            $$
            \widehat B=\widetilde A\Theta,\qquad
            \Theta^\star=\arg\min_\Theta
            \|\widetilde A\Theta-B\|_F^2.
            $$

            One squared-error summand is the squared Euclidean displacement
            between a predicted and exact belief point. The minimum-norm
            least-squares solution is

            $$
            \Theta^\star=\widetilde A^+B.
            $$

            Unlike $(\widetilde A^\top\widetilde A)^{-1}\widetilde A^\top B$,
            this expression remains defined when features are redundant or the
            Gram matrix is singular. A stronger split can hold out whole
            prefix subtrees (all histories beginning with selected two-symbol
            prefixes), trajectories, or context lengths, rather than randomly
            scattering neighboring histories across train and test.
            """,
            show_solutions,
        ),
        md(
            r"""
            ## ACT IV — Try to defeat your own probe

            A successful affine fit is easy to admire and easy to overread.
            Before seeing a plot, decide what comparisons would make failure
            possible and what alternative explanation each comparison targets.

            ## Predict the controls before seeing the picture

            A beautiful projected fractal is memorable, but the evidential work
            is done by comparisons.
            """
        ),
        md(
            r"""
            ### Control card — the evaluation vocabulary

            | Term | Meaning here |
            |---|---|
            | **fit / training set** | histories whose activation–belief pairs choose the probe parameters $\Theta$ |
            | **evaluation / held-out set** | histories not used to choose $\Theta$; these test whether the same map generalizes |
            | **mean squared error (MSE)** | average squared coordinate error, $\frac{1}{nm}\sum_{i,j}(\widehat B_{ij}-B_{ij})^2$; lower is better |
            | **baseline** | a competing predictor using less or different information |
            | **negative control** | a deliberately broken relationship, such as shuffled activation–belief pairings, that should fail |
            | **checkpoint** | the same model architecture saved at a particular stage of training |

            A probe with tiny fit error but large held-out error has memorized or
            interpolated the fit set. If shuffled pairings also achieved low
            held-out MSE, low error for the real pairing would not be convincing.
            None of these observations alone shows that the model **uses** the
            decoded direction: decodability is correlational; causal use requires
            a targeted intervention and a predicted downstream effect.
            """
        ),
        md(
            r"""
            ## CORE 4/5 — What does each comparison rule out? (10 minutes)

            Complete one row for each comparison. Predict the result under a
            genuine activation-to-belief relationship and name the main
            alternative it addresses.

            | Comparison | Predicted result | Alternative addressed |
            |---|---|---|
            | fit on some complete two-symbol prefix subtrees; evaluate on different subtrees | | |
            | randomly permute belief labels across activations before fitting | | |
            | repeat from initialization through trained checkpoints | | |
            | compare with mean-belief, token-plus-length, and exact-next-token baselines | | |

            **Primary artifact:** the completed control table.

            **Instructor handoff:** do any of these show that the transformer
            *uses* the decoded direction? If not, what kind of intervention
            would?

            <details><summary>Hint</summary>
            Separate “generalizes to unseen examples”, “not an arbitrary
            high-to-low-dimensional projection”, “emerges with learning”, and
            “contains more than optimal next-token probabilities”.
            </details>
            """
        ),
        response(
            r"""
            1. Low subtree-held-out MSE shows that the map generalizes across a
               structured change in prefixes, not merely between interleaved,
               neighboring histories.
            2. After shuffling, the target fractal still exists as a set, but its
               pointwise correspondence with activations is broken. Failure
               rules out the story that an arbitrary linear projection can be
               chosen to paint any desired geometry.
            3. Improvement through training links decodability to learned
               predictive competence rather than initialization or architecture
               alone.
            4. Beating the mean rules out trivial concentration near the
               stationary belief. Beating token/length features rules out
               obvious prefix covariates. Beating a next-token-only map asks
               whether the probe contains distinctions beyond the optimal
               next-token distribution. The shuffled-label negative control in
               item 2 has a different role from these baselines.

            These are correlational and geometric controls. They do not show
            causal use. A causal test could perturb an activation specifically
            along a decoded belief direction, predict the resulting Bayesian
            change in later-token probabilities or later activations, and compare
            it with matched orthogonal and random-direction interventions.
            """,
            show_solutions,
        ),
        md(
            r"""
            ## CORE CHECKPOINT — Mess3 success exposes a confound (10 minutes total)

            The next cell constructs exact beliefs for the **pedagogical
            Mess3-family parameter variant used in Notebook 1**
            ($x=.15,a=.20$), embeds them into a 24-dimensional synthetic
            “activation” with a random affine map, and adds nuisance variation.
            The paper uses a different Mess3 parameterization
            ($x=.05,a=.85$). Across four mock checkpoints, the
            signal-to-noise ratio rises. The probe trains on all histories in
            three two-symbol prefix subtrees and tests on the other six; it is
            also compared with three baselines and a shuffled-label negative
            control.

            **This is a teaching simulation, not transformer activations, not
            paper data, and not the paper's generator parameters.** Its purpose
            is to make the regression and controls inspectable without asking
            you to train a model.

            Spend about 3 minutes predicting, run the supplied cell, then use
            the remaining time to explain the result. Before running, sketch:

            - how held-out MSE and recovered geometry should change from the
              earliest to the latest checkpoint;
            - whether Mess3's exact next-token probabilities determine its
              belief, and how you could decide from the emission map.
            """
        ),
        code(
            r"""
            # TEACHING SIMULATION — supplied analysis and visualization.
            import itertools
            import matplotlib.pyplot as plt

            simplex_vertices = np.array([
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, np.sqrt(3) / 2],
            ])

            def mess3(x=0.15, a=0.2):
                b = (1 - a) / 2
                y = 1 - 2*x
                ay, bx, by, ax = a*y, b*x, b*y, a*x
                return np.array([
                    [[ay,bx,bx], [ax,by,bx], [ax,bx,by]],
                    [[by,ax,bx], [bx,ay,bx], [bx,ax,by]],
                    [[by,bx,ax], [bx,by,ax], [bx,bx,ay]],
                ])

            def exact_belief(word, tensor, initial):
                v = initial.copy()
                for x in word:
                    v = v @ tensor[x]
                return v / v.sum()

            mess_T = mess3()
            initial = np.ones(3) / 3
            histories = [
                word
                for depth in range(2, 7)
                for word in itertools.product(range(3), repeat=depth)
            ]
            B = np.array([exact_belief(w, mess_T, initial) for w in histories])

            rng = np.random.default_rng(12)
            n, d = len(B), 24
            embedding = rng.normal(size=(3, d))
            offset = rng.normal(scale=.4, size=d)
            nuisance = rng.normal(size=(n, d))

            # A structured split: whole first-two-symbol subtrees are held out.
            train_subtrees = {(0, 0), (1, 1), (2, 2)}
            train = np.array([
                i for i, word in enumerate(histories)
                if tuple(word[:2]) in train_subtrees
            ])
            test = np.array([
                i for i, word in enumerate(histories)
                if tuple(word[:2]) not in train_subtrees
            ])

            def augment(A):
                return np.column_stack([A, np.ones(len(A))])

            def fit_probe(A, targets):
                return np.linalg.pinv(augment(A)) @ targets

            def apply_probe(A, theta):
                return augment(A) @ theta

            levels = [
                ("initialization", 0.00),
                ("early", 0.05),
                ("middle", 0.18),
                ("late", 1.20),
            ]
            predictions = []
            mses = []
            activations = []
            for name, signal_strength in levels:
                A = offset + signal_strength*(B @ embedding) + .12*nuisance
                theta_level = fit_probe(A[train], B[train])
                B_hat = apply_probe(A[test], theta_level)
                activations.append(A)
                predictions.append(B_hat)
                mses.append(np.mean((B_hat - B[test])**2))

            fig, axes = plt.subplots(1, len(levels) + 1, figsize=(16, 3.7))
            true_xy = B[test] @ simplex_vertices
            axes[0].scatter(true_xy[:, 0], true_xy[:, 1],
                            c=np.clip(B[test], 0, 1), s=7, alpha=.7)
            axes[0].set_title("GROUND TRUTH\nheld-out subtrees")
            for ax, (name, _), B_hat in zip(axes[1:], levels, predictions):
                recovered_xy = B_hat @ simplex_vertices
                ax.scatter(recovered_xy[:, 0], recovered_xy[:, 1],
                           c=np.clip(B[test], 0, 1), s=7, alpha=.65)
                ax.set_title(
                    f"{name.upper()}\nMSE={np.mean((B_hat-B[test])**2):.4f}"
                )
            for ax in axes:
                ax.set_aspect("equal")
                ax.axis("off")
            fig.suptitle(
                "BELIEF GEOMETRY RECOVERY — color = exact belief; "
                "position = probe prediction",
                fontsize=14, fontweight="bold",
            )
            fig.text(
                .5, .015,
                "Late recovery means one affine map extrapolates to prefix "
                "subtrees excluded from fitting.",
                ha="center", fontsize=9,
            )
            plt.tight_layout(rect=(0, .05, 1, .91))
            plt.show()

            final_A = activations[-1]

            # Baselines are fit on exactly the same histories.
            mean_hat = np.tile(B[train].mean(axis=0), (len(test), 1))

            history_features = np.zeros((n, 4))
            for i, word in enumerate(histories):
                history_features[i, word[-1]] = 1
                history_features[i, 3] = len(word) / 6
            covariate_theta = fit_probe(history_features[train], B[train])
            covariate_hat = apply_probe(
                history_features[test], covariate_theta
            )

            emission_map = np.stack([
                mess_T[symbol].sum(axis=1) for symbol in range(3)
            ], axis=1)
            next_token = B @ emission_map
            assert np.linalg.matrix_rank(emission_map) == 3
            next_theta = fit_probe(next_token[train], B[train])
            next_hat = apply_probe(next_token[test], next_theta)
            next_token_mse = np.mean((next_hat - B[test])**2)
            assert next_token_mse < 1e-12

            shuffled_B = B[rng.permutation(n)]
            shuffled_theta = fit_probe(final_A[train], shuffled_B[train])
            shuffled_hat = apply_probe(final_A[test], shuffled_theta)

            comparison_names = [
                "mean",
                "token +\nlength",
                "next-token\nonly",
                "activation\nprobe",
                "shuffled\nlabels",
            ]
            comparison_mses = [
                np.mean((mean_hat - B[test])**2),
                np.mean((covariate_hat - B[test])**2),
                next_token_mse,
                mses[-1],
                np.mean((shuffled_hat - B[test])**2),
            ]

            fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))
            axes[0].bar(
                np.arange(len(comparison_mses)), comparison_mses,
                color=["#9aa4b6", "#9aa4b6", "#d39a3a", "#3976a8", "#b55d5d"],
            )
            axes[0].set(
                xticks=np.arange(len(comparison_names)),
                xticklabels=comparison_names,
                ylabel="held-out belief MSE",
                title="BASELINES + NEGATIVE CONTROL",
            )
            axes[0].tick_params(axis="x", labelsize=8)
            axes[0].text(
                2, max(comparison_mses) * .035,
                "exact\n(<$10^{-12}$)", ha="center", va="bottom",
                fontsize=8, color="#8a5a12",
            )
            late_xy = predictions[-1] @ simplex_vertices
            axes[1].scatter(late_xy[:, 0], late_xy[:, 1],
                            c=np.clip(B[test], 0, 1), s=7, alpha=.65)
            axes[1].set_title("CORRECT PAIRING\nactivation → belief")

            shuffled_xy = shuffled_hat @ simplex_vertices
            axes[2].scatter(shuffled_xy[:, 0], shuffled_xy[:, 1],
                            c=np.clip(B[test], 0, 1), s=7, alpha=.65)
            axes[2].set_title("NEGATIVE CONTROL\nshuffled targets")
            for ax in axes[1:]:
                ax.set_aspect("equal")
                ax.axis("off")
            fig.suptitle(
                "CONTROLS — what predicts held-out belief geometry?",
                fontsize=14, fontweight="bold",
            )
            fig.text(
                .5, .015,
                "Here next-token probabilities are invertible coordinates for "
                "Mess3 belief; RRXOR is needed to test beyond the current "
                "next-token distribution.",
                ha="center", fontsize=9,
            )
            plt.tight_layout(rect=(0, .06, 1, .90))
            plt.show()

            print("Training subtrees:", sorted(train_subtrees))
            print("Held-out subtree fraction:", len(test) / n)
            print("Mess3 emission-map rank:", np.linalg.matrix_rank(emission_map))
            for name, value in zip(comparison_names, comparison_mses):
                print(name.replace("\n", " "), "MSE:", value)
            """
        ),
        md(
            r"""
            ### After running — use the result to choose the next experiment

            Compare your predictions with the plots.

            **Primary artifact:** the next-token-only baseline reaches numerical
            zero error. Use
               $p_{\mathrm{next}}=bE$ and the rank of $E$ to explain why, then
            finish this sentence: “Mess3 cannot distinguish belief geometry
            from ______, so the next experiment must ______.”

            **Instructor handoff:** why may recovered points fall outside the
            simplex, why use ground-truth colors, and why is a held-out prefix
            subtree stronger than a random point split?

            **OPTIONAL AUDIT:** what aspect of the simulation is deliberately
            unrealistic, and what would you inspect before trusting the same
            pipeline on real transformer activations?
            """
        ),
        response(
            r"""
            An unconstrained affine regression does not enforce nonnegativity or
            sum-to-one, so noise can place predictions outside the simplex.
            Ground-truth colors preserve point identity: nearby colors in the
            recovered plot mean that nearby true beliefs remain nearby after
            projection. Coloring by the predictions themselves could make a
            distorted result look smooth by construction.

            Holding out whole subtrees prevents closely related continuations
            from straddling the split and asks the affine map to extrapolate
            across a meaningful context family.

            Here the emission map $E$ has rank three, so it is invertible:
            $p_{\mathrm{next}}=bE$ implies
            $b=p_{\mathrm{next}}E^{-1}$. Exact next-token probabilities
            therefore reconstruct Mess3 belief up to numerical precision, even
            more accurately than the noisy synthetic activation probe. Mess3
            alone cannot establish information beyond the current next-token
            distribution. The central next test needs histories or a process
            matched on optimal next-token distributions but separated on
            longer futures.

            The simulation explicitly plants a linear belief signal and merely
            changes its signal-to-noise ratio; a real network is not handed this
            embedding. On real activations we would inspect held-out errors,
            split construction, layer and position choices, baselines, shuffled
            correspondences, dependence on probe regularization, and whether
            simpler covariates such as token, position, or model next-token probabilities
            explain the result.
            """,
            show_solutions,
        ),
        md(
            r"""
            ## ACT V — Design the comparison that Mess3 cannot make

            The probe worked, but the strongest simple baseline worked even
            better: for this Mess3 parameterization, the current next-token
            vector is already an invertible coordinate for the belief. That
            apparent success creates the final design problem. We need a process
            containing histories that agree now but differ later—the same logic
            that started the notebook, scaled into a diagnostic dataset.

            ## CORE 5/5 — Matched next-token histories (10 minutes)

            The paper also studies Random–Random–XOR (RRXOR), whose 36 HMM
            belief states include many pairs with the same next-token
            prediction but different later conditional probabilities.

            Z1R already supplied the logic: histories `01` and `10` have equal
            current next-token vectors, while test `00` separates their longer
            futures. A representation that separates such matched pairs cannot
            be explained by optimal next-token distributions alone. RRXOR scales
            that diagnostic idea to a richer finite belief set.

            Before reading the result:

            1. As computation approaches the final LayerNorm and unembedding
               for the current next token, where would you expect distinctions
               irrelevant to that distribution to become weaker?
            2. Why might concatenating activations from all layers reveal a
               representation not linearly decodable from any one layer?
            3. Design a distance comparison that separates “activation geometry
               tracks beliefs” from “activation geometry tracks only next-token
               probabilities”.

            **Primary artifact:** one sketch showing the compared pairwise
            distances and your predicted layer pattern. Label the observation
            that would favor each explanation.
            """
        ),
        response(
            r"""
            Predictive directions irrelevant to the current next-token
            distribution can be compressed as computation approaches the final
            unembedding, so final-layer decodability may be weaker than
            intermediate or distributed decodability.

            Different layers may each carry different linear views or fragments
            of the relevant state. Concatenation lets one linear map combine
            these coordinates, though this observation alone does not specify
            the transformer's causal algorithm.

            For pairs of histories, compute:

            - distance between their exact belief vectors;
            - distance between their decoded activation representations;
            - distance between their next-token distributions.

            If decoded-activation distances follow belief distances even among
            pairs whose next-token-probability distance is zero or small, optimal
            next-token distributions cannot explain the geometry. RRXOR was
            chosen to make this comparison diagnostic.
            """,
            show_solutions,
        ),
        code(
            r"""
            # EVIDENCE VISUAL — Shai et al. (2024), Figure 7.
            show_shai_figure(
                7,
                "RRXOR — belief geometry beyond the current next-token vector",
                "A: process · B: exact beliefs · C: decoded representation · "
                "D: distance tests · E: layerwise probe error",
                figsize=(15.5, 9.7),
            )
            """
        ),
        md(
            r"""
            ### Reveal — reported RRXOR result

            RRXOR belief geometry is poorly decoded from the final layer and
            from individual layers, but is recovered from concatenated
            activations across layers. Pairwise distances in the recovered
            representation track ground-truth belief distances ($R^2=0.95$);
            their relation to distances between ground-truth optimal next-token
            probability distributions is much weaker ($R^2=0.31$).

            This is the core contrast that Mess3 alone cannot supply: the
            decoded representation separates chosen-HMM beliefs even where
            current next-token probabilities are matched; for the diagnostic
            pairs, those beliefs imply different later conditionals.

            **Source boundary:** the paper reports Mess3 cross-validation and
            shuffled-label tests in Figure 6, but it does not report an
            RRXOR-specific held-out split for Figure 7. The RRXOR numbers above
            therefore establish a reported fitted affine correspondence, not
            verified out-of-sample generalization. Repeating the analysis on
            held-out RRXOR histories is a necessary next control, especially for
            the 256-dimensional concatenated representation.

            Read the evidence in [Shai et al. (2024), Figure
            7](https://arxiv.org/html/2405.15943#S3.F7): panel C shows the
            recovered clusters, panel D compares competing explanations of
            their pairwise distances, and panel E shows why concatenating all
            layers matters for RRXOR.
            """
        ),
        md(
            r"""
            ## CORE SYNTHESIS — the discovery chain

            | What failed or remained ambiguous? | What question did that force next? | What we learned |
            |---|---|---|
            | the current next-token vector cannot update through the Z1R collision | what richer coordinate is sufficient? | an HMM belief is one update-ready choice |
            | sufficiency does not imply a transformer uses those coordinates | what observation would test the hypothesis? | pair each activation with the exact belief and fit a simple diagnostic map |
            | a fitted projection can overfit or track trivial covariates | which comparisons can make it fail? | held-out data, shuffled labels, checkpoints, and baselines carry the evidential burden |
            | Mess3 belief is reconstructible from its current next-token vector | how can we separate the two explanations? | use matched-next-token histories that differ in later futures |
            | RRXOR shows the reported belief-distance pattern, but without a reported RRXOR holdout | what evidence is still missing? | out-of-sample validation and causal intervention remain open |

            Complete both clauses before leaving the core route:

            > The affine probe is interesting because ________.
            >
            > The affine probe is not yet decisive because ________.
            """
        ),
        md(
            r"""
            # OPTIONAL EXTENSIONS

            The guided discovery is complete. The remaining sections provide
            source-paper evidence, an algebra/code lab, and an experimental
            design studio. Choose only what serves the available time.
            """
        ),
        md(
            r"""
            ## OPTIONAL PAPER CHECKPOINT — Read the complete evidence sequence (8 minutes)

            The three figures below are the paper's main Mess3 result, its
            controls, and its one-page visual synthesis. They appear only now so
            that the empirical answer did not pre-empt CORE 5.

            For Mess3, the authors calculate each history's exact belief, record
            the 64-dimensional final residual stream before the final LayerNorm
            and unembedding, and fit an affine activation-to-belief map. Figure 5
            puts the HMM, exact geometry, and recovered geometry side by side.
            Figure 6 asks whether the recovery emerges through training,
            generalizes to held-out pairs, and survives the shuffled-label
            negative control. Figure 1 compresses the whole argument into the
            paper's visual abstract.
            """
        ),
        code(
            r"""
            # EVIDENCE VISUALS — Shai et al. (2024), Figures 5, 6, and 1.
            show_shai_figure(
                5,
                "MESS3 MAIN RESULT — exact and decoded belief geometry",
                "A: generator · B: ground-truth MSP geometry · "
                "C: affine projection of final residual activations",
                figsize=(16.0, 5.7),
            )
            show_shai_figure(
                6,
                "MESS3 CONTROLS — emergence, generalization, and label shuffle",
                "A: training checkpoints · B: 20/80 cross-validation · "
                "C: shuffled correspondence · D: probe MSE",
                figsize=(13.8, 8.2),
            )
            show_shai_figure(
                1,
                "THE PAPER IN ONE FIGURE — prediction, representation, training",
                "top: generator implies belief geometry · "
                "bottom: residual geometry converges toward it",
                figsize=(10.5, 9.3),
            )
            """
        ),
        md(
            r"""
            Use the figures as an evidence ladder rather than as decoration:

            1. **Figure 5:** compare point identity as well as overall outline;
               the RGB color comes from the exact belief, while position in
               panel C comes from the affine probe.
            2. **Figure 6:** check the alternative explanation addressed by each
               panel. The paper's cross-validation uses random
               input–activation-pair splits, fitting on 20% and evaluating on
               80%, repeated independently 1,000 times—not the structured
               prefix-subtree split in our teaching simulation.
            3. **Figure 7:** contrast Mess3's final-layer result with RRXOR's
               distributed-across-layers result.

            These results support affine decodability and geometric
            correspondence. They do not, on their own, prove that the decoded
            coordinates are causally used. Full paper: [local NeurIPS
            PDF](../papers/shai_et_al_2024.pdf) · [online HTML with figure
            captions](https://arxiv.org/html/2405.15943).
            """
        ),
        md(
            r"""
            ## OPTIONAL ALGEBRA + CODE LAB

            These three short checks make the affine geometry concrete. They are
            deliberately placed after the core synthesis so that mechanics do
            not interrupt the experimental argument.

            <details><summary>Derive the normal equations</summary>
            Differentiating
            $\|\widetilde A\Theta-B\|_F^2$ gives
            $2\widetilde A^\top(\widetilde A\Theta-B)$. Setting it to zero
            yields
            $\widetilde A^\top\widetilde A\Theta=\widetilde A^\top B$.
            This derivation is optional; the geometry and experimental split
            carry more weight in the timed path.
            </details>

            ### Undo a tilted affine embedding (5 minutes)

            Suppose a three-state belief $b=(b_0,b_1,b_2)$, with
            $b_0+b_1+b_2=1$, is embedded in two activation coordinates:

            $$
            a_1=2b_0+b_1+1,\qquad
            a_2=b_0-b_1-1.
            $$

            Solve for $b_0,b_1,b_2$ as affine functions of $a_1,a_2$. This
            confirms that an apparently tilted and translated activation
            triangle can contain exactly the same belief geometry.
            """
        ),
        response(
            r"""
            Adding the activation equations cancels $b_1$ and the constants:

            $$
            b_0=\frac{a_1+a_2}{3}.
            $$

            The combination $a_1-2a_2-3$ cancels $b_0$, giving

            $$
            b_1=\frac{a_1-2a_2-3}{3},\qquad
            b_2=1-b_0-b_1.
            $$

            The offset is essential: a purely linear map through the origin
            need not undo the translation by $(1,-1)$.
            """,
            show_solutions,
        ),
        md(
            r"""
            ### Translate the formula into one line (2 minutes)

            The cell supplies exact activation–belief pairs. Replace `None` with
            **one NumPy expression** corresponding to your pseudoinverse formula.
            This is a quick transcription check, not a programming exercise.

            <details><summary>API hint</summary>
            `np.linalg.pinv(M)` computes $M^+$, and matrix multiplication uses
            `@`.
            </details>
            """
        ),
        code(
            """
            # PROVIDED SETUP — run this collapsed cell first.
            import numpy as np
            np.set_printoptions(precision=4, suppress=True)

            B_tiny = np.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1/3, 1/3, 1/3],
            ])
            A_tiny = np.column_stack([
                2*B_tiny[:, 0] + B_tiny[:, 1] + 1,
                B_tiny[:, 0] - B_tiny[:, 1] - 1,
            ])
            A_tiny_aug = np.column_stack([A_tiny, np.ones(len(A_tiny))])
            """
        ),
        code(
            f"""
            # ONE-LINE STUDENT EXERCISE
            {theta_line}
            """,
            ["student-code"],
        ),
        code(
            """
            # PROVIDED CHECK — run after the one-line exercise.

            if theta is None:
                print("Replace `None` with the pseudoinverse expression, then rerun.")
            else:
                B_tiny_hat = A_tiny_aug @ theta
                print("Theta shape:", theta.shape)
                print("maximum absolute reconstruction error:",
                      np.max(np.abs(B_tiny_hat - B_tiny)))
                print("reconstructed beliefs:\\n", B_tiny_hat)
            """
        ),
        md(
            r"""
            ## STRETCH DESIGN STUDIO — choose one experiment card

            Choose **one** card rather than inventing every component at once:

            - **Matched-prediction process:** construct many history pairs with
              identical optimal next-token distributions and separated two- or
              three-step tests.
            - **Belief-direction intervention:** perturb an activation along a
              decoded belief direction and compare later logits with the HMM's
              quantitative prediction.
            - **Objective comparison:** train matched architectures under
              objectives that differ in how much longer-future information is
              useful.

            Fill one row:

            | Field | Your design |
            |---|---|
            | diagnostic process / contrast | |
            | activation source and target | |
            | negative control or baseline | |
            | numerical success criterion | |
            | outcome that weakens the hypothesis | |

            If time remains, exchange rows with another group and identify one
            confound their proposed control does not address.
            """
        ),
        response(
            r"""
            One example:

            - **Process:** use a small HMM with many history pairs matched on
              next-token probabilities but separated on two- and three-token
              tests.
            - **Measurements:** decode beliefs and next-token vectors from every
              layer and position using strictly held-out histories.
            - **Controls:** shuffled targets, matched random directions,
              position/token baselines, and a model trained on permuted data
              that preserves unigram statistics but destroys the long-range
              process.
            - **Intervention:** move an activation a controlled distance along a
              decoded belief difference while holding the decoded next-token
              vector approximately fixed; predict the later conditional logits
              from the HMM update.
            - **Criterion:** later logits and later decoded beliefs change in
              the predicted direction more than under matched controls.
            - **Falsifier:** high observational decodability but no selective,
              predicted downstream effect under intervention would weaken the
              claim that the affine belief coordinates are causally used.
            """,
            show_solutions,
        ),
    ])
    return notebook(cells, f"02 Transformer belief geometry — {variant}")


def notebook_3(show_solutions: bool) -> dict[str, Any]:
    variant = "Worked solutions" if show_solutions else "Exercises"
    cells: list[dict[str, Any]] = [
        md(
            f"""
            # 3 — Hankel matrices, predictive states, and WFA reconstruction

            **{variant} · draft for the Iliad Intensive**

            Notebook 1 represented predictive state as a posterior over the
            hidden states of a chosen HMM. Here we ask the model-free version of
            the same question: if all we know is the probability of every
            observable word, what information about the past is sufficient for
            predicting the future?

            The answer begins with a simple object—the list of future-word
            probabilities after a history. A **Hankel matrix** arranges all such
            lists into one table. Its row space exposes predictive dimension;
            selected columns become **predictive state representation (PSR)**
            coordinates; and symbol-labelled linear maps give a **weighted
            finite automaton (WFA)** realization.

            By the end, you should be able to:

            - define predictive equivalence without referring to hidden states;
            - construct and normalize a process Hankel matrix;
            - explain what finite-block rank can and cannot establish;
            - derive core future tests as coordinates for prediction;
            - recover the Hankel-rank bound implied by an HMM realization;
            - distinguish observable predictive coordinates from the hidden
              states of a chosen realization.

            **Route through the notebook:** four numbered **CORE** problems plus
            the **CORE SYNTHESIS** have a 55-minute task budget; allow 60–70
            minutes including orientation, supplied checks, and debrief. The
            Z1R bridge, WFA reconstruction, and control-theory bridge are
            optional extensions and may be skipped without breaking the core
            argument. The WFA extension introduces a new linear coordinate
            system, so it is best treated as an instructor-led extension unless
            the group is moving quickly. The final reading links are for groups
            with substantial time left.

            **Working mode.** Most answers are short derivations. Supplied code
            only checks arithmetic or draws a mathematical object; no Python
            knowledge is assumed.
            """
        ),
        heraclitus_hidden_nature_epigraph(),
        md(
            r"""
            ## From hidden-state beliefs to observable predictive state

            Suppose a process emits symbols from an alphabet $\mathcal A$. A
            **history** $h$ is a finite word already observed; a **test** $t$ is
            a possible finite word in the future. The empty word $\epsilon$ is
            both the history before anything has been seen and the test that
            asks for no additional symbols.

            If $\Pr(h)>0$, the history determines a complete predictive profile

            $$
            K_{h,\cdot}=\big(\Pr(t\mid h)\big)_{t\in\mathcal A^*}.
            $$

            This is the observable analogue of a belief over the hidden states
            of a chosen HMM. Such a belief answers future questions by
            $\eta T^{(t)}\mathbf1$; a predictive profile
            stores the answers themselves. Two histories are **predictively
            equivalent** when their profiles agree for every finite test:

            $$
            h\sim h'
            \quad\Longleftrightarrow\quad
            \Pr(t\mid h)=\Pr(t\mid h')
            \quad\text{for every }t\in\mathcal A^*.
            $$

            These equivalence classes are predictive states. They are defined
            by observable consequences, not by a claim about what hidden
            mechanism generated the data.

            ### Why introduce a second, unnormalized table?

            Conditional profiles are the conceptually right states, but their
            entries contain a different denominator for every history. Multiply
            the row by $\Pr(h)$ and the conditionals become joint word
            probabilities:

            $$
            H_{h,t}=\Pr(ht)=\Pr(h)K_{h,t}.
            $$

            The infinite matrix $H$ is the **process Hankel matrix**. Histories
            index its rows, tests index its columns, and concatenation $ht$
            selects an observable word probability. The Hankel matrix preserves
            the predictive information while exposing ordinary linear algebra.
            When $\Pr(h)>0$, divide row $h$ by its first entry
            $H_{h,\epsilon}=\Pr(h)$ to recover the conditional profile.
            """
        ),
        code(
            r"""
            # ORIENTATION VISUAL — the observable route through this notebook.
            import numpy as np
            import matplotlib.pyplot as plt
            from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

            fig, ax = plt.subplots(figsize=(12.4, 3.15))
            ax.set(xlim=(0, 12.4), ylim=(0, 3.15))
            ax.axis("off")

            labels = [
                ("histories +\nfuture tests", "observable questions"),
                ("predictive\nprofiles", r"$K_{h,t}=P(t\mid h)$"),
                ("Hankel\nrow space", r"$H_{h,t}=P(ht)$"),
                ("core tests\n/ PSR", "finite coordinates"),
                ("symbol\noperators", "optional WFA"),
            ]
            xs = [.15, 2.65, 5.15, 7.65, 10.15]
            for index, ((title, subtitle), x) in enumerate(zip(labels, xs)):
                color = "#eef3ff" if index < 4 else "#f3f0f8"
                edge = "#27324a" if index < 4 else "#6b5aa6"
                ax.add_patch(FancyBboxPatch(
                    (x, 1.15), 2.05, 1.02,
                    boxstyle="round,pad=.05,rounding_size=.08",
                    fc=color, ec=edge, lw=1.6,
                ))
                ax.text(x+1.025, 1.75, title, ha="center", va="center",
                        fontsize=10.5, fontweight="bold")
                ax.text(x+1.025, 1.34, subtitle, ha="center", va="center",
                        fontsize=8.5, color="#566179")
                if index < 4:
                    ax.add_patch(FancyArrowPatch(
                        (x+2.05, 1.66), (xs[index+1], 1.66),
                        arrowstyle="-|>", mutation_scale=15, lw=1.5,
                        color="#47546b",
                    ))
            ax.text(
                6.2, .55,
                "One question throughout: which distinctions among histories "
                "matter for every future prediction?",
                ha="center", fontsize=10.5,
            )
            ax.set_title(
                "FROM OBSERVATIONS TO PREDICTIVE STATE",
                fontsize=15, fontweight="bold",
            )
            plt.show()
            """
        ),
        md(
            r"""
            ### Questions to keep in view

            The core asks you to discover the answers rather than memorizing
            them in advance.

            | Question | Observable object to inspect | Where you will settle it |
            |---|---|---|
            | When are two histories the same predictive state? | conditional rows $K_{h,\cdot}$ | CORE 3 |
            | How many linear coordinates are needed? | row/column dependencies of $H$ | CORE 2 + synthesis |
            | Which observable coordinates can represent a state? | selected future-test columns | CORE 4 |
            | How should reading one more symbol update them? | the shift from $h$ to $hx$ | CORE 4 + optional WFA |

            A finite data table exposes only a finite block of $H$. Its rank is
            a **lower bound** on the full rank, never automatically an upper
            bound. In empirical work, singular values of large estimated blocks
            can suggest an effective dimension, and a truncated SVD can produce
            a low-rank **linear** approximation. It does not by itself return the
            closest stochastic HMM: nonnegativity, normalization, and
            consistency of all induced word probabilities are additional
            requirements.

            ### Notation card — keep this visible

            | Symbol | Meaning |
            |---|---|
            | $\mathcal A^*$ | all finite words over the alphabet, including $\epsilon$ |
            | $h,t$ | an observed history and a future test |
            | $\mathcal P,\mathcal T$ | finite sets of histories and tests used to select a Hankel block |
            | $K_{h,t}=\Pr(t\mid h)$ | conditional predictive table |
            | $H_{h,t}=\Pr(ht)$ | joint process Hankel matrix |
            | $\operatorname{rank}(H)$ | dimension of the full linear predictive span |
            | $\mathcal Q=\{q_1,\ldots,q_r\}$ | selected core tests whose columns span that space |

            The full matrices are conceptual infinite objects. Every numerical
            calculation below uses a labelled finite block, so rows and columns
            can always be traced back to concrete words.
            """
        ),
        md(
            r"""
            ## Running example: an observable binary source

            Imagine that an idealized experiment has supplied exact word
            probabilities for a stationary binary process. No hidden states or
            transition graph are given: $\Pr(w)$ is the primitive observable.
            We will use this small rational table to derive the formalism, then
            return to the HMM language of Notebook 1 only after the observable
            construction is complete.

            Two entries are concealed so that you check the probability laws
            once without spending the notebook on fraction bookkeeping.

            | length | observed word probabilities |
            |---|---|
            | 0 | $\Pr(\epsilon)=1$ |
            | 1 | $\Pr(0)=2/3,\quad\Pr(1)=1/3$ |
            | 2 | $\Pr(00)=1/2,\quad\Pr(01)=?,\quad\Pr(10)=1/6,\quad\Pr(11)=1/6$ |
            | 3 | $\Pr(000)=3/8,\ \Pr(001)=1/8,\ \Pr(010)=1/12,\ \Pr(011)=1/12$ |
            |   | $\Pr(100)=1/8,\ \Pr(101)=?,\ \Pr(110)=1/12,\ \Pr(111)=1/12$ |

            Every process obeys **right-extension consistency** because the next
            symbol must be either `0` or `1`:

            $$
            \Pr(u)=\Pr(u0)+\Pr(u1).
            $$

            Stationarity additionally permits the time origin to be shifted, so
            the same marginalization works on the left:

            $$
            \Pr(u)=\Pr(0u)+\Pr(1u).
            $$
            """
        ),
        md(
            r"""
            ## CORE 1/4 — Check the source before representing it (10 minutes)

            1. Use right-extension consistency to recover only the two missing
               entries $\Pr(01)$ and $\Pr(101)$.
            2. Check left-extension consistency in two nontrivial places:
               $\Pr(00)=\Pr(000)+\Pr(100)$ and
               $\Pr(10)=\Pr(010)+\Pr(110)$. What assumption makes these
               identities appropriate?
            3. Calculate the complete next-symbol distributions after histories
               `0` and `1`.
            4. Propose—but do not yet treat as proved—the simplest continuation
               rule suggested by these distributions. Under that hypothesis,
               predict the withheld value $\Pr(1010)$.

            <details><summary>Hint</summary>
            If the continuation is first-order,
            $\Pr(1010)=\Pr(101)\Pr(0\mid1)$.
            </details>
            """
        ),
        response(
            r"""
            Right-extension consistency gives

            $$
            \begin{aligned}
            \Pr(01)&=\Pr(0)-\Pr(00)=\frac16,\\
            \Pr(101)&=\Pr(10)-\Pr(100)=\frac1{24}.
            \end{aligned}
            $$

            The two left-extension checks are
            $3/8+1/8=1/2$ and $1/12+1/12=1/6$. They rely on stationarity:
            observing $u$ at a fixed location has the same distribution after
            shifting the time origin, so we may marginalize the preceding
            symbol.

            The next-symbol laws are

            $$
            \begin{aligned}
            \big(\Pr(0\mid0),\Pr(1\mid0)\big)&=(3/4,1/4),\\
            \big(\Pr(0\mid1),\Pr(1\mid1)\big)&=(1/2,1/2).
            \end{aligned}
            $$

            The simplest hypothesis is a first-order Markov continuation whose
            next-symbol law depends only on the final symbol. It predicts
            $\Pr(1010)=(1/24)(1/2)=1/48$. Agreement with one withheld value
            supports this continuation but does not logically identify it from
            a finite table.
            """,
            show_solutions,
        ),
        code(
            r"""
            # CHECK — the completed observable word table.
            from fractions import Fraction
            import numpy as np
            import matplotlib.pyplot as plt

            word_p = {
                "": Fraction(1, 1),
                "0": Fraction(2, 3), "1": Fraction(1, 3),
                "00": Fraction(1, 2), "01": Fraction(1, 6),
                "10": Fraction(1, 6), "11": Fraction(1, 6),
                "000": Fraction(3, 8), "001": Fraction(1, 8),
                "010": Fraction(1, 12), "011": Fraction(1, 12),
                "100": Fraction(1, 8), "101": Fraction(1, 24),
                "110": Fraction(1, 12), "111": Fraction(1, 12),
            }
            for length in range(4):
                layer = {w: p for w, p in word_p.items() if len(w) == length}
                print(f"length {length}:", layer,
                      "sum =", sum(layer.values(), Fraction(0)))
            observed_held_out = Fraction(1, 48)
            first_order_prediction = (
                word_p["101"] * word_p["10"] / word_p["1"]
            )
            print("HELD-OUT OBSERVATION — P(1010) =", observed_held_out)
            print("FIRST-ORDER PREDICTION           =", first_order_prediction)
            assert first_order_prediction == observed_held_out
            """
        ),
        md(
            r"""
            ## CORE 2/4 — Build a Hankel block and read its rank honestly (12 minutes)

            A **process Hankel matrix** has histories as rows, future tests as
            columns, and concatenated word probabilities as entries:

            $$
            H_{h,t}=\Pr(ht).
            $$

            We cannot write the infinite matrix, so begin with histories and
            tests $\mathcal P=\mathcal T=\{\epsilon,0,1\}$ in that order.

            1. Translate the definition entry by entry to fill the $3\times3$
               block $H$. Explain in words what $H_{0,1}$ and
               $H_{1,\epsilon}$ mean. Why is the first column not all ones?
            2. Use right-extension consistency to show that, for **any** binary
            process and every displayed history $h$,

               $$
               H_{h,\epsilon}=H_{h,0}+H_{h,1}.
               $$

               What upper bound does this generic column dependence put on the
               rank of this particular block?
            3. Show that the upper-left $2\times2$ minor is nonzero. Combine the
               lower and upper bounds to determine the block's rank.
            4. The block is a submatrix of the full infinite Hankel matrix.
               Which bound on the **full** rank follows? Why does no full-rank
               upper bound follow from a finite block?

            <details><summary>Hint 1</summary>
            Concatenate the row word and column word. Thus the entry in row `0`,
            column `1` is $\Pr(01)$, not $\Pr(1\mid0)$.
            </details>

            <details><summary>Hint 2</summary>
            The empty suffix gives $H_{h,\epsilon}=\Pr(h)$, while the next symbol
            must be either `0` or `1`.
            </details>
            """
        ),
        response(
            r"""
            The joint Hankel block is

            $$
            H=
            \begin{pmatrix}
            1&2/3&1/3\\
            2/3&1/2&1/6\\
            1/3&1/6&1/6
            \end{pmatrix}.
            $$

            In particular, $H_{0,1}=\Pr(01)=1/6$ is the joint
            probability of seeing history `0` followed by test `1`, whereas
            $H_{1,\epsilon}=\Pr(1)=1/3$ is the probability of history `1`;
            appending the empty test adds no symbols.

            Its first column contains prefix probabilities because appending the
            empty word changes nothing: $\Pr(h\epsilon)=\Pr(h)$. Moreover,

            $$
            \Pr(h)=\Pr(h0)+\Pr(h1),
            $$

            so column $\epsilon$ is the sum of columns `0` and `1`. This forces
            rank at most two for **every** binary-process block using these three
            suffixes; the dependence is bookkeeping, not evidence of a
            two-dimensional process. The upper-left minor has determinant

            $$
            \frac12-\frac49=\frac1{18}\ne0.
            $$

            The displayed submatrix therefore has rank two. As a submatrix, it
            proves that the full Hankel rank is **at least** two. A larger suffix
            or prefix set could reveal new independent directions, so the block
            supplies no upper bound on full rank. The Z1R extension later gives a
            concrete warning: its $\{\epsilon,0,1\}$ block also has rank two,
            while a block containing the longer test `00` has rank three.
            """,
            show_solutions,
        ),
        code(
            r"""
            # VISUAL / CHECK — the joint Hankel block and its singular values.
            labels = [r"$\epsilon$", "0", "1"]
            H = np.array([
                [1,   2/3, 1/3],
                [2/3, 1/2, 1/6],
                [1/3, 1/6, 1/6],
            ], dtype=float)

            fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.7))
            image = axes[0].imshow(H, cmap="Blues", vmin=0, vmax=1)
            for i in range(3):
                for j in range(3):
                    axes[0].text(j, i, f"{H[i,j]:.3f}", ha="center",
                                 va="center", color="#172033")
            axes[0].set(xticks=range(3), xticklabels=labels,
                        yticks=range(3), yticklabels=labels,
                        xlabel="future test $t$", ylabel="history $h$",
                        title=r"$H_{h,t}=P(ht)$")
            fig.colorbar(image, ax=axes[0], fraction=.046)

            singular_values = np.linalg.svd(H, compute_uv=False)
            axes[1].bar(range(1, 4), singular_values, color="#5876a8")
            axes[1].set(xticks=range(1, 4), xlabel="index",
                        ylabel="singular value",
                        title="Rank 2 here; full rank still open")
            for i, value in enumerate(singular_values, start=1):
                axes[1].text(i, value, f"{value:.3g}", ha="center",
                             va="bottom")
            fig.text(
                .5, .01,
                r"The zero singular value is forced by "
                r"$H_{\cdot,\epsilon}=H_{\cdot,0}+H_{\cdot,1}$; "
                "it is not yet a predictive-dimension result.",
                ha="center", fontsize=9,
            )
            plt.tight_layout(rect=(0, .07, 1, 1))
            plt.show()
            """
        ),
        md(
            r"""
            ## CORE 3/4 — From Hankel rows to predictive states (12 minutes)

            A joint row is weighted by how likely its history was. A predictive
            row instead contains conditional future probabilities:

            $$
            K_{h,t}=\Pr(t\mid h)=\frac{H_{h,t}}{\Pr(h)}
            \quad\text{when }\Pr(h)>0.
            $$

            1. Normalize the displayed rows for histories $\epsilon$, `0`, and
               `1`. Interpret the resulting row for history `0` as answers to
               three future questions.
            2. Form the additional joint row for history `10` and suffixes
               $\{\epsilon,0,1\}$ using the length-three table. Show both

               $$
               H_{10,\cdot}=\frac14H_{0,\cdot}
               \quad\text{and}\quad
               K_{10,\cdot}=K_{0,\cdot}.
               $$

               What distinction does normalization remove?
            3. The finite rows agree only on $\{\epsilon,0,1\}$. Why does that
               not yet prove $0\sim10$ under the all-tests definition?
            4. Now adopt the simplest first-order continuation proposed in CORE
               1. Show that every nonempty history ending in `0` is equivalent,
               and likewise for `1`. Identify the two recurrent predictive
               states and label their token-conditioned transitions. Is
               $\epsilon$ equivalent to either one, or is it a distinct
               transient predictive state?

            <details><summary>Hint</summary>
            $H_{10,\cdot}=(\Pr(10),\Pr(100),\Pr(101))$.
            </details>

            <details><summary>Hint 2</summary>
            Under a first-order continuation, the last symbol fixes the law of
            the next symbol. After that next symbol arrives, it becomes the new
            last symbol. Use induction on the length of the future test.
            </details>
            """
        ),
        response(
            r"""
            The three conditional rows are

            $$
            K_\epsilon=(1,2/3,1/3),\quad
            K_0=(1,3/4,1/4),\quad
            K_1=(1,1/2,1/2).
            $$

            For history `10`,

            $$
            H_{10,\cdot}=(1/6,1/8,1/24)
            =\frac14(2/3,1/2,1/6)
            =\frac14H_{0,\cdot}.
            $$

            Dividing by $\Pr(10)=1/6$ gives
            $K_{10}=(1,3/4,1/4)=K_0$.

            Proportional joint rows say that the two histories have the same
            conditional predictions but occurred with different marginal
            probabilities. Equal normalized rows directly compare predictions.

            A finite block tests only finitely many futures. Two conditional
            rows could agree here and differ on a longer test, so the displayed
            equality suggests but does not prove full predictive equivalence.

            We now add the first-order continuation hypothesis:

            $$
            \Pr(0\mid\text{last }0)=3/4,\qquad
            \Pr(0\mid\text{last }1)=1/2.
            $$

            Histories with the same last symbol have the same one-step law. Once
            the next symbol is observed, both histories again share a last
            symbol; induction on test length therefore makes all their
            finite-future laws agree. Histories ending in `0` form predictive
            state $C_0$ and those ending in `1` form $C_1$:

            $$
            C_0\xrightarrow{0:3/4}C_0,\quad
            C_0\xrightarrow{1:1/4}C_1,
            $$

            $$
            C_1\xrightarrow{0:1/2}C_0,\quad
            C_1\xrightarrow{1:1/2}C_1.
            $$

            The empty history carries the stationary mixture
            $2/3\,C_0+1/3\,C_1$. Because $K_\epsilon$ differs from both
            $K_0$ and $K_1$, it is a third predictive-equivalence class in this
            one-sided, finite-history description—but a transient initial class,
            not a third recurrent state.
            """,
            show_solutions,
        ),
        code(
            r"""
            # VISUAL — the predictive machine inferred from observable rows.
            from matplotlib.patches import Circle, FancyArrowPatch

            fig, ax = plt.subplots(figsize=(7.2, 3.5))
            pos = {"C0": np.array([0.0, 0.0]), "C1": np.array([3.0, 0.0])}
            for name, xy in pos.items():
                ax.add_patch(Circle(xy, .34, fc="#eef3ff", ec="#27324a",
                                    lw=2, zorder=3))
                ax.text(*xy, rf"$C_{name[-1]}$", ha="center", va="center",
                        fontsize=13, zorder=4)

            def edge(a, b, label, rad, label_xy):
                ax.add_patch(FancyArrowPatch(
                    pos[a], pos[b], arrowstyle="-|>", mutation_scale=16,
                    shrinkA=28, shrinkB=28, lw=1.7, color="#3f4d68",
                    connectionstyle=f"arc3,rad={rad}",
                ))
                ax.text(*label_xy, label, ha="center", va="center", fontsize=11,
                        bbox=dict(boxstyle="round,pad=.2", fc="white", ec="none"))

            edge("C0", "C1", r"1 : $\frac{1}{4}$", .10, (1.5, .27))
            edge("C1", "C0", r"0 : $\frac{1}{2}$", .10, (1.5, -.32))

            def loop(name, label, above=True):
                center = pos[name]
                y_offset = .24 if above else -.24
                start = center + np.array([-.20, y_offset])
                end = center + np.array([.20, y_offset])
                curvature = -1.55 if above else 1.55
                ax.add_patch(FancyArrowPatch(
                    start, end, arrowstyle="-|>", mutation_scale=16,
                    lw=1.7, color="#3f4d68",
                    connectionstyle=f"arc3,rad={curvature}",
                ))
                label_y = .92 if above else -.92
                ax.text(center[0], label_y, label, ha="center", va="center",
                        fontsize=11,
                        bbox=dict(boxstyle="round,pad=.2", fc="white", ec="none"))

            loop("C0", r"0 : $\frac{3}{4}$", above=True)
            loop("C1", r"1 : $\frac{1}{2}$", above=False)
            ax.set(xlim=(-1, 4), ylim=(-1.35, 1.35), aspect="equal")
            ax.axis("off")
            ax.set_title("Observable predictive-state machine")
            plt.show()
            """
        ),
        md(
            r"""
            ## From rank to a finite predictive state representation

            Suppose the **full** Hankel matrix has finite rank $r$. Then we can
            choose $r$ linearly independent columns, indexed by tests

            $$
            \mathcal Q=\{q_1,\ldots,q_r\}.
            $$

            Every other Hankel column is a linear combination of these columns.
            Therefore the $r$ numbers

            $$
            s(h)=\big(\Pr(hq_1),\ldots,\Pr(hq_r)\big)
            $$

            determine $\Pr(ht)$ for every future test $t$. This is the basic
            predictive-state-representation result: **independent future
            questions can serve as coordinates for every other future
            prediction.** The $q_i$ are called **core tests**.

            The state $s(h)$ above is unnormalized and includes the probability
            of reaching history $h$. If $\epsilon\in\mathcal Q$, its first
            coordinate is $\Pr(h)$. Dividing by it gives conditional test
            probabilities and restricts the state to an affine slice. Symbol
            updates are linear on $s(h)$ and generally rational on the
            normalized conditional state—the same normalization pattern seen
            for HMM beliefs in Notebook 1.

            **Guardrail:** Hankel rank is the dimension of a linear span, not
            generally the number of predictive-equivalence classes. Even a
            finite-rank process can have many—sometimes infinitely many—distinct
            conditional predictive rows inside that finite-dimensional space.
            """
        ),
        md(
            r"""
            ## CORE 4/4 — Choose core tests and derive their update (13 minutes)

            A **predictive state representation** uses the probabilities of a
            selected set of future experiments, or **core tests**, as state
            coordinates.

            For the adopted first-order source, choose
            $\mathcal Q=\{\epsilon,0\}$. Its unnormalized and normalized states
            are

            $$
            s(h)=\big(\Pr(h),\Pr(h0)\big),\qquad
            \bar s(h)=\frac{s(h)}{\Pr(h)}=(1,p),
            $$

            where $p=\Pr(0\mid h)$.

            1. Evaluate $s(h)$ and $\bar s(h)$ for
               $h\in\{\epsilon,0,1,10\}$. Which distinction is removed by
               normalization?
            2. Express the four length-two test probabilities
               $\Pr(00\mid h),\Pr(01\mid h),\Pr(10\mid h),\Pr(11\mid h)$ in
               terms of $p$, and verify that they sum to one. This demonstrates
               concretely how the core tests determine non-core tests.
            3. Starting from the identity (valid when $\Pr(hx)>0$)

               $$
               \Pr(t\mid hx)=\frac{\Pr(xt\mid h)}{\Pr(x\mid h)},
               $$

               derive the updated value $p'=\Pr(0\mid hx)$ after observing `0`
               and after observing `1`.
            4. Why does a rank-two **linear** state have only one free
               coordinate after normalization?

            <details><summary>Hint 1</summary>
            Condition on the first future symbol, then use the transition
            probabilities leaving $C_0$ or $C_1$.
            </details>

            <details><summary>Hint 2</summary>
            For the update after `0`, compute
            $p'=\Pr(00\mid h)/\Pr(0\mid h)$; use the analogous expression after
            `1`.
            </details>
            """
        ),
        response(
            r"""
            Reading the first two columns of the Hankel matrix gives

            $$
            \begin{array}{c|c|c}
            h&s(h)&\bar s(h)\\ \hline
            \epsilon&(1,2/3)&(1,2/3)\\
            0&(2/3,1/2)&(1,3/4)\\
            1&(1/3,1/6)&(1,1/2)\\
            10&(1/6,1/8)&(1,3/4).
            \end{array}
            $$

            The unnormalized rows distinguish how probable the histories were;
            normalization removes that marginal weight and retains only their
            predictions. Thus $p=3/4$ in $C_0$, $p=1/2$ in $C_1$, and $p=2/3$
            initially.

            Conditioning on the first future symbol gives

            $$
            \Pr(00\mid h)=\frac34p,\qquad
            \Pr(01\mid h)=\frac14p,
            $$

            $$
            \Pr(10\mid h)=\frac12(1-p),\qquad
            \Pr(11\mid h)=\frac12(1-p).
            $$

            Their sum is $p+(1-p)=1$, and each is a linear function of the core
            coordinate $p$.

            When $p=\Pr(0\mid h)>0$, the update identity gives

            $$
            \Pr(0\mid h0)
            =\frac{\Pr(00\mid h)}{\Pr(0\mid h)}
            =\frac{(3/4)p}{p}=\frac34,
            $$

            provided that the observed `0` has positive probability.

            When $1-p=\Pr(1\mid h)>0$, it similarly gives

            $$
            \Pr(0\mid h1)
            =\frac{\Pr(10\mid h)}{\Pr(1\mid h)}
            =\frac{(1/2)(1-p)}{1-p}=\frac12.
            $$

            The unnormalized linear state needs one coordinate for total mass
            and one for predictive variation. Conditional states fix the mass
            coordinate $\Pr(\epsilon\mid h)=1$, leaving a one-dimensional affine
            set.
            """,
            show_solutions,
        ),
        md(
            r"""
            ## CORE SYNTHESIS — Derive the HMM factorization (8 minutes)

            The finite table did not logically force a unique continuation to
            all word lengths. We now continue to adopt the first-order source
            reconstructed above. It has the two-state edge-emitting realization

            $$
            T^{(0)}=
            \begin{pmatrix}3/4&0\\1/2&0\end{pmatrix},
            \qquad
            T^{(1)}=
            \begin{pmatrix}0&1/4\\0&1/2\end{pmatrix},
            \qquad
            \pi=(2/3,1/3).
            $$

            Notebook 1 established the word formula
            $\Pr(w)=\pi T^{(w)}\mathbf1$.

            1. Apply the word formula to a concatenated history and test $ht$,
               then split the product at their boundary. Call the resulting
               $1\times2$ row the **history embedding** $\phi(h)$ and the
               $2\times1$ column the **test embedding** $\psi(t)$. Derive their
               formulas and explain operationally what each component means.
               Check the pairing once by computing $\phi(0)\psi(1)=\Pr(01)$.
            2. Stack the history rows into an infinite-by-two matrix $\Phi$ and
               the test columns into a two-by-infinite matrix $\Psi$. Derive
               $H=\Phi\Psi$.
               Why does this imply $\operatorname{rank}(H)\le2$?
            3. Combine this full-rank upper bound with CORE 2's finite-block
               lower bound. What is the exact full Hankel rank of the **adopted
               continuation**?
            4. Explain why neither the original finite table nor the number of
               hidden states in an arbitrary, possibly redundant HMM would have
               justified that exact conclusion on its own.
            5. What extra property would be unsafe to infer from finite Hankel
               rank alone: a finite-dimensional linear realization, or a
               same-sized nonnegative stochastic HMM realization?

            <details><summary>Hint</summary>
            A product through a two-dimensional inner index has rank at most
            two. Keep separate the lower bound supplied by a finite submatrix
            and the upper bound supplied by a realization valid for every word.
            </details>
            """
        ),
        response(
            r"""
            The word formula splits at the boundary between history and future:

            $$
            \Pr(ht)=
            \underbrace{\pi T^{(h)}}_{\phi(h)\in\mathbb R^{1\times2}}
            \underbrace{T^{(t)}\mathbf1}_{\psi(t)\in\mathbb R^{2\times1}}.
            $$

            The two entries of $\phi(h)$ are the unnormalized masses that the
            history leaves at the two boundary states. The two entries of
            $\psi(t)$ are the probabilities of the future test $t$ starting
            from each boundary state. For example,

            $$
            \phi(0)=\pi T^{(0)}=(2/3,0),\qquad
            \psi(1)=T^{(1)}\mathbf1=(1/4,1/2)^\top,
            $$

            so $\phi(0)\psi(1)=(2/3)(1/4)=1/6=\Pr(01)$.

            Collecting $\phi(h)$ for all histories as rows and $\psi(t)$ for all
            tests as columns gives $H=\Phi\Psi$ and factors the **adopted
            first-order process's full Hankel matrix** through a two-dimensional
            space, so its full rank is at most two. Separately, the displayed
            finite block is a submatrix with rank two, so the full rank is at
            least two. These two logically distinct bounds establish that the
            adopted continuation has full Hankel rank exactly two. The finite
            table alone did not supply the upper bound.

            The finite table alone supplied no upper bound because an
            unmeasured longer test could add a new direction. Conversely, the
            size of an arbitrary HMM supplies only an upper bound: redundant
            hidden states can represent the same predictive direction.

            More generally, finite Hankel rank guarantees a finite-dimensional
            linear realization. It does **not** by itself guarantee that a
            same-sized realization has nonnegative entries and stochastic HMM
            semantics; nonnegative realization requires additional conditions.
            """,
            show_solutions,
        ),
        md(
            r"""
            ## Core synthesis table

            Complete this comparison before leaving the core:

            | Object | Coordinates refer to | Update type | Canonical caveat |
            |---|---|---|---|
            | HMM belief | posterior over a chosen model's hidden states | normalized $\,\eta T^{(x)}\,$ | model-relative; may be redundant |
            | Conditional Hankel row | probabilities of all future tests | conditioning by $\Pr(x\mid h)$ | infinite object |
            | Finite PSR | probabilities of selected core tests | generally rational after normalization | tests must span the predictive space |

            In one sentence, state what remains invariant when we change among
            valid coordinate systems.
            """
        ),
        response(
            r"""
            What remains invariant is the observable stochastic process: every
            valid realization must assign the same probability to every finite
            word, even though its state coordinates, geometry under arbitrary
            metrics, and update matrices may look different.
            """,
            show_solutions,
        ),
        md(
            r"""
            ---

            # Optional extension A — Belief ↔ PSR bridge (15–25 minutes)

            Notebook 1 found two Z1R histories, `01` and `10`, whose beliefs
            predict the same next-symbol distribution but different longer
            futures. We can now state that result in observable coordinates.

            Return to fair Z1R and order its belief coordinates as $(a,b,c)$
            over $(S_0,S_1,S_R)$. Choose two tests:

            $$
            q_0=\Pr(0\mid h),\qquad q_{00}=\Pr(00\mid h).
            $$

            1. Use the Z1R transition rules from Notebook 1 to derive $q_0$ and
               $q_{00}$ as functions of $(a,b,c)$.
            2. Combine those equations with $a+b+c=1$ to recover $(a,b,c)$ from
               $(q_0,q_{00})$.
            3. Evaluate the coordinates after histories `01` and `10`. Which
               test separates the beliefs that one-step prediction merged?
            4. Derive the coordinate update after observing `0`, whenever that
               observation has positive probability, by conditioning extended
               tests. **Stretch:** derive it after `1`.
            5. Explain what this example does—and does not—show about the
               relationship between HMM beliefs and PSR coordinates.

            <details><summary>Hint 1</summary>
            From $S_0$, `0` is certain; from $S_1$, it is impossible; from
            $S_R$, it has probability $1/2$. Two zeros in a row can begin only
            from $S_R$.
            </details>

            <details><summary>Hint 2</summary>
            Use $a+b+c=1$ after solving for $a$ and $c$.
            </details>

            **More open design route:** instead of the algebra above, modify
            Z1R's random state or cyclic skeleton to create your own pair of
            histories with equal next-token distributions and different longer
            futures. Specify the smallest future-test set you found that
            separates them.
            """
        ),
        response(
            r"""
            From the three hidden phases,

            $$
            q_0=a+\frac12c,\qquad q_{00}=\frac12c.
            $$

            Therefore

            $$
            a=q_0-q_{00},\qquad
            b=1-q_0-q_{00},\qquad
            c=2q_{00}.
            $$

            After `01`, the belief is $(0,0,1)$, so
            $(q_0,q_{00})=(1/2,1/2)$. After `10`, the belief is
            $(1/2,1/2,0)$, so $(q_0,q_{00})=(1/2,0)$. The one-step test `0`
            merges them; the longer test `00` separates them.

            When $q_0>0$, observing `0` has positive probability and Bayes'
            rule in predictive coordinates gives

            $$
            F_0(q_0,q_{00})
            =\left(\frac{q_{00}}{q_0},0\right).
            $$

            If $q_0=0$, observing `0` is impossible from that state and the
            conditional update is undefined rather than a $0/0$ value.

            The stretch update is

            $$
            F_1(q_0,q_{00})
            =
            \left(
            \frac{1-q_0+q_{00}}{2(1-q_0)},
            \frac{1-q_0-q_{00}}{2(1-q_0)}
            \right),
            $$

            whenever the observed symbol has nonzero probability.

            For this minimal Z1R realization, observable PSR coordinates and
            HMM-belief coordinates are invertibly related. This does not make
            arbitrary HMM beliefs canonical: a redundant realization can have
            distinct hidden-state posteriors that induce the same distribution
            over every observable future.
            """,
            show_solutions,
        ),
        code(
            r"""
            # CHECK — a small Z1R Hankel basis has rank 3.
            # Prefix and suffix order: epsilon, 0, 00.
            H_z1r = np.array([
                [1,   1/2, 1/6],
                [1/2, 1/6, 0],
                [1/6, 0,   0],
            ], dtype=float)
            print(H_z1r)
            print("rank:", np.linalg.matrix_rank(H_z1r))
            print("determinant:", np.linalg.det(H_z1r))
            print(
                "Interpretation: normalization plus two independent predictive "
                "coordinates, consistent with (q0, q00)."
            )
            """
        ),
        md(
            r"""
            ---

            # Optional extension B — Reconstruct a WFA (20–25 minutes)

            Continue with the adopted first-order source, whose full Hankel rank
            the core established to be two.

            A rank-$r$ Hankel matrix admits an $r$-dimensional **weighted finite
            automaton (WFA)**, also called a linear or observable-operator
            realization. It consists of

            $$
            \alpha\in\mathbb R^{1\times r},\qquad
            \{A_x\in\mathbb R^{r\times r}:x\in\mathcal A\},\qquad
            \omega\in\mathbb R^{r\times1},
            $$

            and assigns a word $w=x_1\cdots x_L$ the weight

            $$
            f(w)=\alpha A_{x_1}\cdots A_{x_L}\omega.
            $$

            Writing $A_h$ and $A_t$ for the ordered products associated with a
            history and future test makes the two WFA embeddings visible:

            $$
            \phi_{\mathrm{WFA}}(h)=\alpha A_h,\qquad
            \psi_{\mathrm{WFA}}(t)=A_t\omega,\qquad
            H_{h,t}=\phi_{\mathrm{WFA}}(h)\psi_{\mathrm{WFA}}(t).
            $$

            Appending a symbol updates the past embedding on the right,
            $\phi_{\mathrm{WFA}}(hx)=\phi_{\mathrm{WFA}}(h)A_x$; prepending it
            updates the future embedding on the left,
            $\psi_{\mathrm{WFA}}(xt)=A_x\psi_{\mathrm{WFA}}(t)$.

            A general WFA computes a real-valued **rational series**: $f(w)$
            may be a score, a signed weight, or another real quantity. It does
            **not** have to be a probability. In this notebook we want an exact
            realization of a stochastic process, so our target is specifically
            the cylinder probability $f(w)=\Pr(w)$. That adds

            $$
            f(\epsilon)=1,\qquad f(w)\ge0,\qquad
            f(w)=\sum_{x\in\mathcal A}f(wx).
            $$

            Equivalently, the weights sum to one separately at every fixed word
            length. This is not a probability distribution over the set of all
            finite words. To model probabilities of **completed** finite strings
            instead, one needs termination semantics, usually an end-of-sequence
            symbol. In either use, the intermediate WFA coordinates are merely a
            linear basis and need not be probabilities.

            The Hankel factorization makes two embeddings explicit. A history
            $h$ is embedded as a row $\phi(h)$; a future test $t$ is
            embedded as a column $\psi(t)$; and their pairing returns the word
            weight:

            $$
            H_{h,t}=f(ht)=\phi(h)\psi(t).
            $$

            This is the observable counterpart of the HMM factors
            $\pi T^{(p)}$ and $T^{(q)}\mathbf1$ from the core synthesis. To
            construct the embeddings from observable probabilities, select
            basis histories $\mathcal P_B$ and core tests $\mathcal Q_B$ so the
            square Hankel block

            $$
            B_{p,q}=\Pr(pq)
            $$

            is invertible. For each symbol $x$, insert $x$ at the
            history/future boundary:

            $$
            (B_x)_{p,q}=\Pr(pxq).
            $$

            The following exercise constructs both embeddings before deriving
            the update equation.
            """
        ),
        md(
            r"""
            ## Guided reconstruction — derive the observable operators

            Use $\mathcal P_B=\mathcal Q_B=\{\epsilon,0\}$ in that order.

            1. Fill $B$, $B_0$, and $B_1$ directly from the word table. In each
               entry, write the concatenated word before substituting its
               probability. Verify that $\det B\ne0$.
            2. For any history $h$, construct its **past embedding**

               $$
               \phi(h)=H_{h,\mathcal Q_B}
               =\big(\Pr(hq)\big)_{q\in\mathcal Q_B}.
               $$

               For any future test $t$, construct its **future embedding**

               $$
               \psi(t)=B^{-1}H_{\mathcal P_B,t}.
               $$

               Check their dimensions, show that $\psi(q_j)=e_j$ for each core
               test, and explain why $\Pr(ht)=\phi(h)\psi(t)$ for every $h,t$
               when the selected columns span the full Hankel matrix. Compute
               the pairing once for $h=1,t=0$.
            3. Reading $x$ must send the history embedding $\phi(h)$ to
               $\phi(hx)$. Apply this to every basis history $h$ to derive

               $$
               BA_x=B_x,
               $$

               then solve the two $2\times2$ systems for $A_0$ and $A_1$.
               The supplied check after your answer will reveal the matrices.

            4. Explain why $\alpha=\phi(\epsilon)$ and
               $\omega=\psi(\epsilon)$. Compute them, then verify

               $$
               \Pr(0)=\alpha A_0\omega,\qquad
               \Pr(01)=\alpha A_0A_1\omega.
               $$

            5. $A_1$ contains negative entries. Why are these not negative
               transition probabilities? State the observable validity
               conditions that the induced word weights must nevertheless
               satisfy.

            <details><summary>Hint</summary>
            The four entries of $B_x$ correspond to
            $\Pr(x)$, $\Pr(x0)$, $\Pr(0x)$, and $\Pr(0x0)$. For validity, think
            back to the normalization and extension-consistency identities in
            CORE 1.
            </details>

            <details><summary>Embedding hint</summary>
            The vector $\psi(t)$ contains the coefficients expressing the
            entire column for $t$ as a linear combination of the selected core
            columns. Pairing those coefficients with row $h$ reconstructs its
            entry in that column.
            </details>
            """
        ),
        response(
            r"""
            Concatenating each basis history $h$ with basis test $t$ for $B$,
            or $h$, inserted symbol $x$, and $t$ for $B_x$, gives

            $$
            B=
            \begin{pmatrix}
            \Pr(\epsilon)&\Pr(0)\\
            \Pr(0)&\Pr(00)
            \end{pmatrix}
            =\begin{pmatrix}1&2/3\\2/3&1/2\end{pmatrix},
            \qquad
            B_0=
            \begin{pmatrix}
            \Pr(0)&\Pr(00)\\
            \Pr(00)&\Pr(000)
            \end{pmatrix}
            =\begin{pmatrix}2/3&1/2\\1/2&3/8\end{pmatrix}.
            $$

            $$
            B_1=
            \begin{pmatrix}
            \Pr(1)&\Pr(10)\\
            \Pr(01)&\Pr(010)
            \end{pmatrix}
            =\begin{pmatrix}1/3&1/6\\1/6&1/12\end{pmatrix}.
            $$

            Thus every entry uses the concatenation convention explicitly; for
            example, the lower-right entry of $B_1$ is
            $\Pr(0\,1\,0)=\Pr(010)=1/12$. The basis is valid because

            $$
            \det B=\frac12-\frac49=\frac1{18}\ne0.
            $$

            The past embedding is the $1\times2$ row
            $\phi(h)=(\Pr(h),\Pr(h0))$. The future embedding is the
            $2\times1$ coefficient column
            $\psi(t)=B^{-1}H_{\mathcal P_B,t}$. Because
            $H_{\mathcal P_B,q_j}$ is column $j$ of $B$, each selected core test
            has $\psi(q_j)=e_j$. More generally, rank two means the full Hankel
            column for $t$ is the same linear combination of the two core
            columns, so taking entry $h$ gives

            $$
            \Pr(ht)=H_{h,t}=\phi(h)\psi(t).
            $$

            For example,

            $$
            \phi(1)=\left(\frac13,\frac16\right),\qquad
            \psi(0)=e_2,
            $$

            and $\phi(1)\psi(0)=1/6=\Pr(10)$. Thus $\phi(h)$ says how
            the **past reaches** the predictive boundary, while $\psi(t)$ says
            how a **future test reads out** that boundary state.

            After reading symbol $x$, the history row $\phi(h)$ must become
            $(B_x)_{h,\cdot}$ for prefix $hx$. Requiring
            $B_{h,\cdot}A_x=(B_x)_{h,\cdot}$ for both basis prefixes and stacking
            the equations gives

            $$
            BA_x=B_x,\qquad A_x=B^{-1}B_x.
            $$

            Multiplying the blocks then gives

            $$
            A_0=B^{-1}B_0
            =
            \begin{pmatrix}
            0&0\\
            1&3/4
            \end{pmatrix},
            \qquad
            A_1=B^{-1}B_1
            =
            \begin{pmatrix}
            1&1/2\\
            -1&-1/2
            \end{pmatrix}.
            $$

            The initial row
            $\alpha=\phi(\epsilon)=B_{\epsilon,\cdot}=(1,2/3)$ is the
            past embedding before any symbol has been observed. The final
            column $\omega=\psi(\epsilon)=e_1=(1,0)^\top$ is the future
            embedding of the empty test, so it extracts the total mass of the
            current unnormalized state. Hence

            $$
            \alpha A_0\omega=2/3=\Pr(0),
            $$

            and

            $$
            \alpha A_0A_1\omega=1/6=\Pr(01).
            $$

            Also
            $\alpha A_1A_0A_1\omega=1/24=\Pr(101)$.

            The entries of $A_x$ update an observable linear coordinate system;
            they are not edge probabilities. Negative intermediate coordinates
            are allowed. The **induced word weights** must still satisfy

            $$
            f(\epsilon)=1,\qquad f(w)\ge0,\qquad
            \sum_{|w|=L}f(w)=1,\qquad
            f(w)=\sum_{x\in\mathcal A}f(wx)
            $$

            (and left-extension consistency when stationarity is claimed).
            Thus this construction is a WFA or observable-operator realization,
            not automatically an HMM with nonnegative stochastic transitions.
            """,
            show_solutions,
        ),
        code(
            r"""
            # CHECK — reconstructed WFA probabilities versus the source table.
            B_basis = np.array([[1, 2/3], [2/3, 1/2]], dtype=float)
            B0 = np.array([[2/3, 1/2], [1/2, 3/8]], dtype=float)
            B1 = np.array([[1/3, 1/6], [1/6, 1/12]], dtype=float)
            operators = [np.linalg.solve(B_basis, B0),
                         np.linalg.solve(B_basis, B1)]
            alpha = np.array([1.0, 2/3])
            omega = np.array([1.0, 0.0])

            def wfa_probability(word):
                state = alpha.copy()
                for symbol in word:
                    state = state @ operators[int(symbol)]
                return float(state @ omega)

            print("A0 =\n", operators[0])
            print("A1 =\n", operators[1])
            print()
            for word in ["", "0", "1", "00", "01", "101", "111"]:
                expected = float(word_p[word])
                actual = wfa_probability(word)
                print(f"{word or 'epsilon':>7}: WFA={actual:.6f}, "
                      f"table={expected:.6f}")
            """
        ),
        md(
            r"""
            ### Where the SVD enters with empirical data

            Our rational example had an exact rank-two block and an invertible
            $2\times2$ basis. With estimated probabilities, every singular value
            will usually be nonzero because of sampling noise. A spectral
            workflow instead:

            1. estimates a larger finite Hankel block from data;
            2. inspects its singular values and chooses a working dimension $r$;
            3. uses a rank-$r$ factorization or pseudoinverse to construct
               linear symbol operators;
            4. tests the reconstructed word probabilities on held-out strings
               and checks stochastic validity.

            Before opening the theorem card, decide which of these claims you
            expect to be true:

            1. Truncated SVD is the best rank-$r$ approximation of the displayed
               finite matrix.
            2. Its output is automatically a Hankel matrix for one all-length
               series.
            3. Some structured infinite-Hankel problems nevertheless do have an
               optimal rank-$r$ solution.
            4. Any such solution is the closest $r$-state HMM.

            Commit a one-line reason for each answer before continuing.
            """
        ),
        response(
            r"""
            Claim 1 is true only for the chosen finite block and an
            unconstrained rank bound. Claim 2 is false because ordinary SVD
            truncation need not preserve Hankel structure. Claim 3 is true in
            the compact scalar Hankel-operator setting covered by AAK. Claim 4
            is false because HMMs impose additional positivity and stochastic
            constraints.
            """,
            show_solutions,
        ),
        md(
            r"""
            ### Theorem card — optimal among what, in which norm?

            Let the singular values use one-based indexing,
            $\sigma_1\ge\sigma_2\ge\cdots$.

            **Exact rank/minimality (Carlyle–Paz/Fliess).** A real series
            $f:\mathcal A^*\to\mathbb R$ has a finite-dimensional WFA
            realization exactly when its full Hankel matrix has finite rank.
            That rank is the minimum number of states in an exact WFA
            realization. This is an exact-representation theorem, not an
            approximation theorem.

            **Finite block (Eckart–Young–Mirsky).** If a particular finite
            block is $M=U\Sigma V^\top$, then
            $M_r=U_r\Sigma_rV_r^\top$ minimizes

            $$
            \min_{\operatorname{rank}(X)\le r}\|M-X\|_2
            =\|M-M_r\|_2=\sigma_{r+1},
            $$

            $$
            \min_{\operatorname{rank}(X)\le r}\|M-X\|_F
            =\|M-M_r\|_F
            =\left(\sum_{j>r}\sigma_j^2\right)^{1/2}
            $$

            over all matrices $X$ of rank at most $r$ (and truncated SVD is
            optimal for every unitarily invariant norm). But an arbitrary
            low-rank matrix need not preserve Hankel shift structure, define one
            consistent all-length series, or satisfy stochastic constraints.

            **Infinite structured Hankel approximation (AAK).** For a compact
            scalar Hankel operator, there exists a rank-at-most-$r$
            **Hankel-structured** operator achieving the optimal operator-norm
            error $\sigma_{r+1}$. This is not generally the raw truncated SVD.
            The exact AAK-based WFA algorithm currently cited here assumes a
            bounded real **one-letter** WFA; its proof does not directly extend
            to a multi-letter alphabet such as our binary example. For general
            alphabets, truncating a singular-value automaton gives principled
            tail-singular-value bounds, not this general global-optimality
            claim.

            None of these theorems says “closest $r$-state HMM.” An HMM adds
            nonnegative symbol matrices, a stochastic normalization, a
            probability initial state, and a choice of process-level loss.

            Sources: [the exact WFA rank theorem](https://cs.nyu.edu/~mohri/pub/swa.pdf);
            [Eckart & Young (1936)](https://doi.org/10.1007/BF02288367);
            [Singular Value Automata and Approximate Minimization](https://arxiv.org/abs/1711.05994);
            [Optimal Approximate Minimization of One-Letter WFAs](https://arxiv.org/abs/2306.00135).
            """
        ),
        md(
            r"""
            ### SVD synthesis

            In two sentences, answer: **In what sense is the rank-$r$ truncation
            optimal, and in what sense is it not an optimal rank-$r$ HMM?**
            """
        ),
        response(
            r"""
            The raw truncated SVD is optimal among all rank-at-most-$r$ matrices
            for the chosen finite block in spectral and Frobenius norm; in the
            classical compact one-variable setting, a different AAK construction
            is optimal among rank-$r$ Hankel operators in operator norm. Neither
            optimization ranges over normalized nonnegative HMMs, and raw SVD
            truncation need not even preserve the all-length Hankel or stochastic
            constraints, so no closest-$r$-state-HMM conclusion follows.
            """,
            show_solutions,
        ),
        md(
            r"""
            ### Optional WFA recap

            Add the WFA to the core comparison: its state is a finite linear
            coordinate row, its symbol update is linear, and neither its
            coordinates nor its operators need be nonnegative. What remains
            invariant is still the induced probability of every finite word.
            """,
        ),
        md(
            r"""
            ---

            # Optional extension C — Control theory sees the same Hankel split (15–20 minutes)

            Control theory asks how an input applied in the past can affect an
            output observed in the future. For a strictly proper linear
            time-invariant (LTI) system, compare

            $$
            \begin{array}{lll}
            \text{discrete time:}&x_{k+1}=Ax_k+Bu_k,&y_k=Cx_k,\\
            \text{continuous time:}&\dot x(t)=Ax(t)+Bu(t),&y(t)=Cx(t).
            \end{array}
            $$

            Here $B$ injects inputs into state and $C$ reads state into outputs;
            these are unrelated to the basis block $B$ used in the WFA
            extension.

            1. Starting from zero state and a unit impulse, derive the discrete
               Markov parameters $g_k=CA^{k-1}B$ for $k\ge1$ and the continuous
               impulse response $g(t)=Ce^{At}B$ for $t>0$.
            2. Show that their past-to-future Hankel kernels factor as

               $$
               \mathcal H^{\mathrm d}_{i,j}=CA^{i+j}B
               =(CA^i)(A^jB),\qquad i,j\ge0,
               $$

               $$
               \mathcal H^{\mathrm c}(t,s)=Ce^{A(t+s)}B
               =(Ce^{At})(e^{As}B),\qquad t,s\ge0.
               $$

               In each factorization, label the map that embeds past inputs into
               a reachable boundary state and the map that embeds that state
               into future outputs. Match them to $\phi(h)$ and $\psi(t)$ in the
               WFA.
            3. Assume the system is stable. Use an index shift for the sums and
               the fundamental theorem of calculus for the integrals to derive
               the four matrix equations satisfied by

               $$
               \begin{aligned}
               P_{\mathrm d}&=\sum_{k\ge0}A^kBB^\top(A^\top)^k,&
               Q_{\mathrm d}&=\sum_{k\ge0}(A^\top)^kC^\top CA^k,\\
               P_{\mathrm c}&=\int_0^\infty e^{At}BB^\top e^{A^\top t}\,dt,&
               Q_{\mathrm c}&=\int_0^\infty e^{A^\top t}C^\top Ce^{At}\,dt.
               \end{aligned}
               $$

               Which pair uses Stein equations, and which uses Lyapunov
               equations?
            4. Suppose a change of coordinates makes
               $P=Q=\operatorname{diag}(\sigma_1,\ldots,\sigma_n)$ with
               $\sigma_1\ge\cdots\ge\sigma_n>0$. Explain in words why a
               direction with small $\sigma_i$ is a natural candidate to
               truncate. Predict which parts of this argument should be common
               to discrete and continuous time.

            **Closer WFA analogy.** A multi-symbol WFA is not an ordinary LTI
            system: the observed symbol selects $A_x$, so it is closer to a
            discrete switched linear system. The common structure here is the
            past → boundary state → future factorization, not an identification
            of inputs with probabilities.
            """
        ),
        response(
            r"""
            In discrete time, an impulse at $k=0$ produces
            $x_1=B$, $x_2=AB$, and in general
            $y_k=CA^{k-1}B$. In continuous time, the state transition matrix is
            $e^{At}$, so the impulse response is $Ce^{At}B$. Splitting elapsed
            time at the present gives the displayed factorizations. The
            reachability factors $A^jB$ and $e^{As}B$ summarize how a past input
            reaches the present state; the observability factors $CA^i$ and
            $Ce^{At}$ summarize how that state appears in future outputs. These
            play the same structural roles as the WFA's past row $\phi(h)$ and
            future column $\psi(t)$.

            Shifting the discrete sums by one term gives the Stein equations

            $$
            P_{\mathrm d}=AP_{\mathrm d}A^\top+BB^\top,\qquad
            Q_{\mathrm d}=A^\top Q_{\mathrm d}A+C^\top C.
            $$

            Differentiating the continuous integrands and using stability to
            make the boundary term at infinity vanish gives the Lyapunov
            equations

            $$
            AP_{\mathrm c}+P_{\mathrm c}A^\top+BB^\top=0,\qquad
            A^\top Q_{\mathrm c}+Q_{\mathrm c}A+C^\top C=0.
            $$

            A balanced coordinate with small $\sigma_i$ is simultaneously hard
            for past inputs to reach and hard for future outputs to observe.
            Both time settings therefore use the same logic—factor the Hankel
            map, balance reachability against observability, and truncate weak
            shared directions—while sums become integrals and Stein equations
            become Lyapunov equations.
            """,
            show_solutions,
        ),
        md(
            r"""
            ### Balanced truncation theorem card

            Assume a finite-dimensional minimal LTI realization and asymptotic
            stability: $\rho(A)<1$ in discrete time or all eigenvalues of $A$
            have negative real part in continuous time.

            - **Balancing.** A change of state coordinates can make the
              controllability and observability Gramians equal:
              $P=Q=\Sigma=\operatorname{diag}(\sigma_i)$. The $\sigma_i$ are the
              Hankel singular values—the singular values of the past-to-future
              Hankel operator.
            - **Balanced truncation.** Retaining the first $r$ balanced
              coordinates preserves asymptotic stability and, in both standard
              discrete- and continuous-time settings, obeys

              $$
              \|G-G_r\|_{\mathcal H_\infty}
              \le 2\sum_{j>r}\sigma_j.
              $$

              This strong guarantee does **not** say balanced truncation is the
              norm-optimal order-$r$ model.
            - **Optimal Hankel-norm approximation.** A separate construction
              characterized by Glover attains error
              $\sigma_{r+1}$ in Hankel norm. It is the control-theory relative
              of AAK structured approximation, not ordinary balanced
              truncation.

            When the relevant Gramians exist, the WFA analogue is a **singular
            value automaton**, whose forward and backward Gramians are both
            $\Sigma$. Truncating it gives a reduced WFA with
            tail-singular-value error bounds, but the LTI
            $\mathcal H_\infty$ theorem above does not automatically transfer to
            a general multi-letter WFA.

            Sources: [Moore on balanced realizations](https://doi.org/10.1109/TAC.1981.1102568);
            [Enns on the balanced-truncation error bound](https://doi.org/10.1109/CDC.1984.272286);
            [Glover on optimal Hankel-norm approximation](https://doi.org/10.1080/00207178408933239).
            """
        ),
        md(
            r"""
            ### If you have lots of time — two modern bridges

            - [*Sequences of Logits Reveal the Low Rank Structure of Language
              Models*](https://arxiv.org/abs/2510.24966) (Golowich, Liu, and
              Shetty, 2025) studies an extended matrix of **mean-centered
              logits** indexed by histories and future/next-token pairs, and
              relates exact finite-horizon low logit rank to time-varying ISANs.
              This is not the probability Hankel matrix
              $H_{h,t}=\Pr(ht)$, so do not identify the two ranks.
            - [*Input Switched Affine Networks: An RNN Architecture Designed for
              Interpretability*](https://proceedings.mlr.press/v70/foerster17a.html)
              (Foerster et al., 2017) studies the recurrence
              $z_{k+1}=A_{x_k}z_k+b_{x_k}$ with a softmax language-model
              readout. The symbol-selected affine update is close to WFA and
              switched-system dynamics, while the softmax output keeps it
              distinct from a linear WFA readout.

            **Reading question:** Which object is low rank in each paper—the
            probability Hankel matrix, a logit matrix, or a hidden-state
            transition—and what conclusion does that rank actually license?
            """
        ),
    ]
    return notebook(cells, f"03 Hankel PSR WFA — {variant}")


def execute_and_embed_outputs(data: dict[str, Any]) -> None:
    """Execute code cells and embed static text/PNG outputs for no-code readers."""

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/iliad-matplotlib")
    import matplotlib.pyplot as plt

    plt.ioff()
    namespace: dict[str, object] = {"__name__": "__notebook__"}
    execution_count = 0

    for index, item in enumerate(data["cells"]):
        if item["cell_type"] != "code":
            continue

        execution_count += 1
        outputs: list[dict[str, Any]] = []

        def capture_show(*_args: object, **_kwargs: object) -> None:
            for figure_number in list(plt.get_fignums()):
                figure = plt.figure(figure_number)
                buffer = io.BytesIO()
                figure.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
                encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
                outputs.append(
                    {
                        "data": {
                            "image/png": encoded,
                            "text/plain": "<embedded teaching figure>",
                        },
                        "metadata": {},
                        "output_type": "display_data",
                    }
                )
            plt.close("all")

        stdout = io.StringIO()
        previous_show = plt.show
        plt.show = capture_show  # type: ignore[assignment]
        try:
            source = "".join(item["source"])
            with contextlib.redirect_stdout(stdout):
                exec(
                    compile(source, f"embedded-notebook-cell-{index}", "exec"),
                    namespace,
                )
            if plt.get_fignums():
                capture_show()
        finally:
            plt.show = previous_show  # type: ignore[assignment]
            plt.close("all")

        text_output = stdout.getvalue()
        if text_output:
            outputs.append(
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": text_output,
                }
            )
        item["execution_count"] = execution_count
        item["outputs"] = outputs


def build_specs() -> list[tuple[str, dict[str, Any]]]:
    """Construct every exercise/solution notebook before touching disk."""

    return [
        ("01_hmms_msps_exercises.ipynb", notebook_1(False)),
        ("01_hmms_msps_solutions.ipynb", notebook_1(True)),
        (
            "02_transformer_belief_geometry_exercises.ipynb",
            notebook_2(False),
        ),
        (
            "02_transformer_belief_geometry_solutions.ipynb",
            notebook_2(True),
        ),
        ("03_hankel_psr_wfa_exercises.ipynb", notebook_3(False)),
        ("03_hankel_psr_wfa_solutions.ipynb", notebook_3(True)),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if generated notebooks differ from a fresh in-memory build",
    )
    args = parser.parse_args()

    specs = build_specs()

    # Execute every variant successfully before writing any of them. This avoids
    # leaving a mixed-generation set if a later notebook fails.
    for filename, data in specs:
        execute_and_embed_outputs(data)

    expected_names = sorted(filename for filename, _ in specs)
    actual_names = sorted(path.name for path in HERE.glob("*.ipynb"))
    if args.check:
        failures: list[str] = []
        if actual_names != expected_names:
            failures.append(
                f"notebook inventory differs: expected {expected_names}, "
                f"found {actual_names}"
            )
        for filename, expected in specs:
            path = HERE / filename
            if not path.exists():
                failures.append(f"missing {filename}")
                continue
            actual = json.loads(path.read_text(encoding="utf-8"))
            if actual != expected:
                failures.append(f"stale or locally edited: {filename}")
        if failures:
            for failure in failures:
                print(f"FAIL {failure}")
            raise SystemExit(1)
        print("PASS generated notebooks exactly match a fresh build")
        return

    serialized = {
        filename: json.dumps(data, indent=1) + "\n"
        for filename, data in specs
    }
    for filename, data in specs:
        path = HERE / filename
        path.write_text(serialized[filename], encoding="utf-8")
        figures = sum(
            1
            for item in data["cells"]
            for output in item.get("outputs", [])
            if "image/png" in output.get("data", {})
        )
        print(
            f"wrote {path.name}: {len(data['cells'])} cells, "
            f"{figures} embedded figures"
        )


if __name__ == "__main__":
    main()
