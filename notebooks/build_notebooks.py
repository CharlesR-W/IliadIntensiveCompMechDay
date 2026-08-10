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
            2. Derive the normalization identity

               $$
               \sum_{x\in\mathcal A}\sum_j T^{(x)}_{ij}=1
               \quad\text{for each source state }i.
               $$

               Explain why it does **not** require either symbol matrix to have
               row sums equal to one.
            3. Form the hidden-state transition matrix
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
            2. Generalize the result to show that the forward row obeys

               $$
               \alpha^{(wx)}=\alpha^{(w)}T^{(x)},\qquad
               \Pr(w)=\alpha^{(w)}\mathbf 1
               =\pi T^{(w)}\mathbf 1.
               $$

            3. Use either hidden paths or the formula to calculate
               $\Pr(\text{'01'})$, $\Pr(\text{'00'})$, and
               $\Pr(\text{'010'})$ for Z1R.
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

            For '01', the only nonzero stationary path begins in $S_0$, so
            $\Pr(\text{'01'})=1/3$. For '00', the process must begin in $S_R$,
            emit its random '0', and then emit the certain '0' from $S_0$, so
            $\Pr(\text{'00'})=(1/3)(1/2)=1/6$. After '01' the process is at
            $S_R$, which emits '0' with probability $1/2$, so
            $\Pr(\text{'010'})=1/6$.

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
               F_{\text{'0'}}(F_{\text{'1'}}(\pi))$.
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

            Thus the two update orders land at different points. Every belief
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
            2. Using the beliefs $\eta^{(\text{'01'})}$ and
               $\eta^{(\text{'10'})}$ from CORE 3, show that both give the
               next-token distribution $(1/2,1/2)$.
            3. For each belief, calculate the distribution over
               '00', '01', '10', and '11'.
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
            $\Pr(x\mid\eta)=\eta T^{(x)}\mathbf 1$. More generally, retaining the
            later hidden path gives $\Pr(u\mid\eta)=\eta T^{(u)}\mathbf 1$.

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
            **Model-relative caveat.** An HMM belief is a sufficient state for
            prediction relative to the chosen HMM realization, but it need not
            be the minimal observable predictive state. Distinct beliefs can,
            in principle, assign the same probability to **every** future word;
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

            A point $(q_0,q_1,q_2)$ in the belief triangle means that the
            observer assigns probability $q_i$ to hidden state $S_i$. Its three
            corners are the **certainty beliefs** $(1,0,0)$, $(0,1,0)$, and
            $(0,0,1)$. For a fixed observed token $x$, update each corner with

            $$
            F_x(q)=\frac{qT^{(x)}}{qT^{(x)}\mathbf1}.
            $$

            At a certainty belief this simply normalizes one row of $T^{(x)}$.
            For a certainty belief $e_i$, write
            $r_i=e_iT^{(x)}\mathbf1$ for the probability of token $x$ and
            $v_i=F_x(e_i)$ for the updated corner. For an uncertain belief $q$,

            $$
            F_x(q)=\sum_i\beta_i v_i,
            \qquad
            \beta_i=\frac{q_i r_i}{\sum_k q_k r_k}.
            $$

            Thus the update uses **likelihood-reweighted** coefficients, not
            generally the original coordinates $q_i$. All Mess3 row masses are
            positive, so the possible coefficients $\beta$ fill the simplex and
            the three corner images bound exactly the region of posteriors after
            observing $x$. The first visual shows those three regions; it does
            not reveal any length-two calculations.
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

            This notebook is a **guided discovery**: it reconstructs a
            plausible chain of questions that could lead from optimal prediction
            to the experiment in Shai et al., *Transformers Represent Belief
            State Geometry in their Residual Stream*.

            You will:

            - turn the “same next token, different future” result into a
              hypothesis about a predictor's internal state;
            - design a dataset pairing transformer activations with exact HMM
              beliefs;
            - derive an affine least-squares probe;
            - translate the derivation into one line of NumPy;
            - predict what cross-validation, label shuffling, and training-time
              comparisons rule out;
            - distinguish belief geometry from next-token geometry.

            **Evidence labels matter.** The interactive plots below use a
            transparent **teaching simulation, not transformer activations and
            not paper data**. Reported empirical results are clearly separated
            and linked to the full paper.

            Five numbered **CORE** tasks plus the Mess3 simulation contain 58
            minutes of explicitly timed work. Their timings include reading the
            prompt, pair discussion, and a brief instructor handoff, leaving
            roughly 10 minutes in a 70-minute notebook path for transitions.
            Prompts marked **OPTIONAL AUDIT** are outside that core budget. The
            fifth task is the
            matched-next-token comparison needed to distinguish belief geometry
            from optimal next-token-distribution geometry. The geometric algebra
            checkpoint, paper checkpoint, one-line transcription, and final
            experimental-design cards are marked **STRETCH** or **OPTIONAL**.
            """
        ),
        seneca_epigraph(),
        code(
            r"""
            # ORIENTATION VISUAL — the objects used in this notebook.
            import numpy as np
            import matplotlib.pyplot as plt
            from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

            fig, ax = plt.subplots(figsize=(12.2, 3.6))
            ax.set(xlim=(0, 12.2), ylim=(0, 3.6))
            ax.axis("off")

            def box(x, y, w, h, text, color="#eef3ff", edge="#27324a"):
                ax.add_patch(FancyBboxPatch(
                    (x, y), w, h, boxstyle="round,pad=.04,rounding_size=.08",
                    fc=color, ec=edge, lw=1.6,
                ))
                ax.text(x+w/2, y+h/2, text, ha="center", va="center",
                        fontsize=10)

            def arrow(x1, y1, x2, y2, label=""):
                ax.add_patch(FancyArrowPatch(
                    (x1, y1), (x2, y2), arrowstyle="-|>",
                    mutation_scale=15, lw=1.6, color="#47546b",
                ))
                if label:
                    ax.text((x1+x2)/2, (y1+y2)/2-.14, label,
                            ha="right", fontsize=9, color="#47546b")

            box(.2, 1.55, 1.65, .72, r"full prefix $h$")
            box(2.35, 1.38, 2.1, 1.05,
                "transformer\n(all prefix tokens)")
            box(5.05, 1.38, 2.25, 1.05,
                r"last-position residual"+"\n"+r"$a_\ell(h)\in\mathbb{R}^d$")
            box(8.0, 2.25, 1.7, .72, "final LayerNorm\n+ unembedding")
            box(10.25, 2.25, 1.7, .72,
                "next-token\nlogits")
            box(8.0, .45, 1.7, .72, "diagnostic\naffine probe",
                color="#fff2d8", edge="#a26712")
            box(10.25, .45, 1.7, .72,
                r"predicted belief"+"\n"+r"$\widehat b(h)$",
                color="#fff2d8", edge="#a26712")

            arrow(1.85, 1.91, 2.35, 1.91)
            arrow(4.45, 1.91, 5.05, 1.91)
            arrow(7.30, 2.05, 8.0, 2.55)
            arrow(9.70, 2.61, 10.25, 2.61)
            arrow(7.30, 1.75, 8.0, .82, "read only")
            arrow(9.70, .81, 10.25, .81)

            ax.text(
                6.15, .05,
                r"Ground truth for comparison: exact HMM belief "
                r"$b(h)=\Pr(S_{|h|}\mid h)$",
                ha="center", fontsize=10, color="#7b4b0b",
            )
            ax.set_title(
                "THE TEST — does a transformer encode belief geometry?",
                fontsize=15, fontweight="bold",
            )
            plt.show()
            """
        ),
        md(
            r"""
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
            earlier tokens at every position. Therefore the argument below does
            not force it to maintain one recursive state. The HMM belief $b(h)$
            is a generator-relative sufficient coordinate: it determines future
            probabilities for the chosen HMM, but it need not be a minimal
            observable predictive state if two beliefs induce the same full
            future law. Notebook 3 will make that model-free distinction
            explicit. Whether the transformer learns coordinates affinely
            related to this chosen belief is an empirical question.
            """
        ),
        md(
            r"""
            ## The discovery puzzle

            A transformer is trained only to minimize next-token cross-entropy.
            Why look for anything richer than its next-token probabilities?

            Recall the Z1R pair from Notebook 1:

            $$
            \eta^{(01)}=(0,0,1),\qquad
            \eta^{(10)}=(1/2,1/2,0).
            $$

            Both predict the next token as $(1/2,1/2)$, but their distributions
            over two-token futures differ. This proves that a current
            next-token vector is not, in general, a sufficient **recursive**
            state. A full-prefix transformer has other options: revisit earlier
            tokens, cache a richer state in activations, or use a different
            predictive coordinate system. Which option training discovers is
            precisely the puzzle.
            """
        ),
        md(
            r"""
            ## CORE 1/5 — What does the Z1R pair actually establish? (10 minutes)

            1. Suppose two histories $h$ and $h'$ have identical next-token
               distributions but different distributions over complete futures.
               Argue that there must be some finite continuation $y$ after which
               their next-token distributions differ.
            2. State the extra assumption needed before “therefore the model
               must distinguish them **now**” follows. Apply it to a recurrent
               predictor whose sole carried state is the next-token vector.
            3. List two ways a full-context transformer could remain optimal
               without storing our HMM posterior as a persistent vector.
            4. Turn the result into a cautious empirical hypothesis about
               $a_\ell(h)$ rather than a theorem about its coordinates.

            <details><summary>Hint 1</summary>
            If two probability distributions over infinite sequences differ,
            they differ on at least one finite cylinder event (a finite prefix).
            Consider the first symbol position at which the relevant conditional
            probabilities diverge.
            </details>

            <details><summary>Hint 2</summary>
            Separate “must retain enough information to distinguish predictive
            futures” from “must use our preferred coordinates for that
            information”.
            </details>
            """
        ),
        response(
            r"""
            If the full future distributions differ, there is a finite word
            $yx$ whose conditional probability differs after $h$ and $h'$.
            Choose a shortest such word. Their probabilities for the shorter
            prefix $y$ agree up to the point where the conditional probability
            of its final symbol $x$ first differs. Thus, after seeing $y$, an
            optimal next-token predictor must make different predictions. If
            $y$ is impossible under one history, the two predictors already
            disagree at or before the first symbol that makes it impossible.

            The extra assumption is that the representation being merged is the
            model's **only** route by which the earlier history can affect later
            predictions. A recurrent predictor that retains only its current
            next-token vector satisfies this assumption: equal vectors followed
            by equal inputs force identical later states, so it fails on one of
            the histories.

            A full-context transformer can instead attend back to distinct
            prefix tokens and recompute the distinction; it could also encode a
            sufficient predictive state in nonlinear or model-independent
            coordinates rather than the chosen HMM's posterior.

            A cautious hypothesis is therefore: predictive training may make
            some layer's last-position residual $a_\ell(h)$ contain a compact
            representation affinely related to the exact belief $b(h)$. That is
            a testable geometric claim, not a necessity theorem.
            """,
            show_solutions,
        ),
        md(
            r"""
            ## CORE 2/5 — Turn the idea into a testable hypothesis (8 minutes)

            Imagine that we:

            1. generate token histories $h_1,\ldots,h_n$ from a known HMM;
            2. feed each history to a trained transformer and record one
               residual-stream activation $a(h_i)\in\mathbb R^d$;
            3. calculate the exact HMM belief
               $b(h_i)\in\Delta^{m-1}\subset\mathbb R^m$.

            Answer before reading on:

            1. What should be the **source** and **target** of a map testing
               whether belief geometry is linearly represented?
            2. Write an affine hypothesis using a matrix $W$ and offset $c$,
               including their shapes.
            3. Why is predicting the belief vector a stronger geometric claim
               than classifying only the most likely hidden state?
            4. Name one low error result that would nevertheless be
               unconvincing without a control.

            <details><summary>Hint</summary>
            The phrase “linearly represented” is operationalized as affine
            decodability from activations, not as a claim that an individual
            neuron equals an individual belief coordinate.
            </details>
            """
        ),
        response(
            r"""
            The source is the $d$-dimensional residual activation; the target is
            the $m$-component ground-truth belief. With column-vector notation,

            $$
            b(h)\approx Wa(h)+c,\qquad
            W\in\mathbb R^{m\times d},\quad c\in\mathbb R^m.
            $$

            A classifier need preserve only decision regions. Reconstructing the
            full continuous belief vector asks it to preserve relative position,
            mixtures, and the geometry inside the simplex.

            Low training error alone is weak: a high-dimensional probe could
            interpolate a small dataset. Other concerns include accidentally
            fitting history length or token identity, choosing the projection
            after seeing the target picture, or obtaining an attractive
            projection even after breaking the activation–belief pairing.
            """,
            show_solutions,
        ),
        md(
            r"""
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

            <details><summary>Hint 1</summary>
            With examples stored as rows, the prediction is
            $\widehat B=\widetilde A\Theta$.
            </details>

            <details><summary>STRETCH — derive the normal equations</summary>
            Differentiating
            $\|\widetilde A\Theta-B\|_F^2$ gives
            $2\widetilde A^\top(\widetilde A\Theta-B)$. Setting it to zero
            yields
            $\widetilde A^\top\widetilde A\Theta=\widetilde A^\top B$.
            This derivation is optional; the geometry and experimental split
            carry more weight in the timed path.
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
            ### STRETCH — Undo a tilted affine embedding (5 minutes)

            Suppose a three-state belief $b=(b_0,b_1,b_2)$, with
            $b_0+b_1+b_2=1$, is embedded in two activation coordinates:

            $$
            a_1=2b_0+b_1+1,\qquad
            a_2=b_0-b_1-1.
            $$

            Before running code, solve for $b_0,b_1,b_2$ as affine functions of
            $a_1,a_2$. This confirms that an apparently tilted and translated
            activation triangle can contain exactly the same belief geometry.
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
            ## STRETCH — Translate the formula into one line (2 minutes)

            The cell supplies exact activation–belief pairs. Replace `None` with
            **one NumPy expression** corresponding to your pseudoinverse formula.
            This is a quick transcription check, not a programming exercise;
            skip it if the matrix formula is already clear.

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

            For each comparison, predict the result under a genuine
            activation-to-belief relationship and state the main alternative it
            addresses.

            1. Fit on complete two-symbol prefix subtrees and evaluate on other
               subtrees, so no held-out history has a near-duplicate ancestor in
               the training set.
            2. Randomly permute belief labels across activations before fitting,
               while preserving the set of target belief points.
            3. Repeat the analysis at checkpoints from initialization through
               the end of training.
            4. Compare the activation probe with three baselines: mean belief,
               last-token-plus-length, and exact next-token probabilities.
            Which of these controls provides causal evidence that the transformer
            **uses** the decoded beliefs? If none does, propose the kind of
            intervention that would.

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
            ## CORE VISUAL — Watch a belief signal become recoverable (10 minutes)

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

            Before running it, sketch:

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
            ### Interpret, do not merely admire

            Compare your predictions with the plots.

            1. Why may recovered points fall slightly outside the simplex, and
               why color them by **ground-truth** belief rather than recovered
               coordinates?
            2. Why is a held-out prefix-subtree split more informative than a
               random point split here?
            3. The next-token-only baseline reaches numerical zero error. Use
               $p_{\mathrm{next}}=bE$ and the rank of $E$ to explain why, then
               say what experiment should become central next.

            **OPTIONAL AUDIT.** What aspect of the simulation is deliberately
            unrealistic? What would you inspect before trusting the same
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
            ## OPTIONAL PAPER CHECKPOINT — What Shai et al. report (8 minutes)

            Full source: [Shai et al., *Transformers Represent Belief State
            Geometry in their Residual Stream* (NeurIPS 2024,
            arXiv:2405.15943)](https://arxiv.org/abs/2405.15943).

            For Mess3, the authors:

            - calculate the exact belief associated with each input history;
            - record 64-dimensional final residual-stream activations before
              the final LayerNorm and unembedding;
            - fit an affine map from activations to the three-state belief;
            - recover the fractal belief geometry;
            - show decreasing regression MSE over training;
            - obtain similar held-out performance under **random
              input–activation-pair splits**, fitting on 20% and evaluating on
              80%, repeated independently 1,000 times—not the structured
              prefix-subtree split used in the teaching simulation above;
            - find that shuffling activation–belief correspondences collapses
              the recovered structure toward the simplex center.

            See the paper's [method schematic and main
            result](https://arxiv.org/html/2405.15943#S2.SS3) and
            [Figure 6 controls](https://arxiv.org/html/2405.15943#S3.F6).

            These results support affine decodability and geometric
            correspondence. They do not, on their own, prove that the decoded
            coordinates are causally used.
            """
        ),
        md(
            r"""
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
        md(
            r"""
            ### Exit ticket

            Complete both clauses:

            > The affine probe is interesting because ________.  
            > The affine probe is not yet decisive because ________.
            """
        ),
    ]
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
            Z1R bridge and WFA reconstruction are optional extensions and may be
            skipped without breaking the core argument. The WFA extension does
            introduce a new linear coordinate system, so it is best treated as
            an instructor-led extension unless the group is moving quickly.

            **Working mode.** Most answers are short derivations. Supplied code
            only checks arithmetic or draws a mathematical object; no Python
            knowledge is assumed.
            """
        ),
        seneca_epigraph(),
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

            This is the observable analogue of an HMM belief. A belief answers
            future questions by $\eta T^{(t)}\mathbf1$; a predictive profile
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
            ### Main results at a glance

            | Question | Observable answer | Result to derive |
            |---|---|---|
            | When are two histories the same predictive state? | Compare their conditional rows $K_{h,\cdot}$ | equality on **all** future tests |
            | How many linear coordinates are needed? | Study the row/column space of $H$ | full Hankel rank $r$ gives an $r$-dimensional linear realization |
            | Which coordinates can we use? | Select $r$ independent test columns | their probabilities determine every other test probability linearly |
            | How does a symbol update the coordinates? | Shift from $h$ to $hx$ | linear before normalization, generally rational after conditioning |

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
            tests $\mathcal P=\mathcal S=\{\epsilon,0,1\}$ in that order.

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
            3. Starting from the identity

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

            The update identity now gives

            $$
            \Pr(0\mid h0)
            =\frac{\Pr(00\mid h)}{\Pr(0\mid h)}
            =\frac{(3/4)p}{p}=\frac34,
            $$

            and, whenever the observed symbol has positive probability,

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

            1. Apply it to a concatenated history and test $ht$, then split the
               product at their boundary to derive

               $$
               H_{h,t}=\Pr(ht)
               =\underbrace{\pi T^{(h)}}_{R_{h,\cdot}}
                \underbrace{T^{(t)}\mathbf 1}_{C_{\cdot,t}}.
               $$

            2. Stack the history rows into an infinite-by-two matrix $R$ and the
               future columns into a two-by-infinite matrix $C$. Why does
               $H=RC$ imply $\operatorname{rank}(H)\le2$?
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
            \Pr(ht)=\pi T^{(h)}T^{(t)}\mathbf1.
            $$

            Collecting $\pi T^{(h)}$ for all histories as rows and
            $T^{(t)}\mathbf1$ for all tests as columns factors the **adopted
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

            # Optional extension B — Reconstruct a WFA (15–20 minutes)

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

            For a stochastic process we require $f(w)=\Pr(w)$, but the
            intermediate coordinates are merely a linear basis. They need not
            be probabilities.

            To recover the operators from observable probabilities, select
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

            The following exercise derives the update equation rather than
            presenting “right extensions” as an unexplained recipe.
            """
        ),
        md(
            r"""
            ## Guided reconstruction — derive the observable operators

            Use $\mathcal P_B=\mathcal Q_B=\{\epsilon,0\}$ in that order.

            1. Fill $B$, $B_0$, and $B_1$ directly from the word table. In each
               entry, write the concatenated word before substituting its
               probability. Verify that $\det B\ne0$.
            2. The row $B_{p,\cdot}$ is the core-test coordinate row for basis
               history $p$. After observing $x$, it must become
               $(B_x)_{p,\cdot}$. Derive the stacked equation

               $$
               BA_x=B_x,
               $$

               then solve the two $2\times2$ systems—or verify by
               multiplication—that

               $$
               A_0=
               \begin{pmatrix}0&0\\1&3/4\end{pmatrix},\qquad
               A_1=
               \begin{pmatrix}1&1/2\\-1&-1/2\end{pmatrix}.
               $$

            3. Explain why the empty-history row is
               $\alpha=(1,2/3)$ and why $\omega=(1,0)^\top$ extracts the
               empty-test column. Verify

               $$
               \Pr(0)=\alpha A_0\omega,qquad
               \Pr(01)=\alpha A_0A_1\omega.
               $$

            4. $A_1$ contains negative entries. Why are these not negative
               transition probabilities? State the observable validity
               conditions that the induced word weights must nevertheless
               satisfy.

            <details><summary>Hint</summary>
            The four entries of $B_x$ correspond to
            $\Pr(x)$, $\Pr(x0)$, $\Pr(0x)$, and $\Pr(0x0)$. For validity, think
            back to the normalization and extension-consistency identities in
            CORE 1.
            </details>
            """
        ),
        response(
            r"""
            Concatenating each basis history $p$ with basis test $q$ for $B$,
            or $p$, inserted symbol $x$, and $q$ for $B_x$, gives

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

            The row $B_{p,\cdot}$ is the observable coordinate row for prefix
            $p$. After reading symbol $x$, it must become the row
            $(B_x)_{p,\cdot}$ for prefix $px$. Requiring
            $B_{p,\cdot}A_x=(B_x)_{p,\cdot}$ for both basis prefixes and stacking
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

            The initial row $\alpha=B_{\epsilon,\cdot}=(1,2/3)$ is the
            core-test state before any symbol has been observed. The column
            $\omega=(1,0)^\top$ selects the coordinate for test $\epsilon$, so
            it returns the total mass of the current unnormalized state. Hence

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
            f(w)\ge0,\qquad
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

            The truncated SVD is optimal for approximating that **particular
            finite matrix block** in Frobenius norm. It is not automatically the
            closest full stochastic process, and still less automatically the
            closest $r$-state HMM. This distinction is why WFA reconstruction
            and nonnegative HMM realization should not be conflated.
            """
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
