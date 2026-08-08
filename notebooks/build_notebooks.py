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
            prediction problem. In this notebook you will derive that claim,
            rather than implement an HMM library.

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
            the Mess3 **CORE SYNTHESIS** form the 60–70 minute path. **STRETCH**
            and **DESIGN STUDIO** boxes are modular: skip them without breaking
            the argument.

            **Working mode.** Most answers are mathematics. Code cells marked
            **visual/check** are supplied plumbing: run them, inspect the result,
            and return to the derivation. No Python knowledge is assumed.
            """
        ),
        seneca_epigraph(),
        md(
            r"""
            ## The running process: Zero–One–Random

            The process repeatedly emits `0`, then `1`, then a random bit, and
            repeats. At time $t$, $S_t$ is the hidden phase **before** the next
            edge. Traversing that edge emits $X_{t+1}$ and arrives at
            $S_{t+1}$. The three phases are:

            - $S_0$: emit `0`, then arrive at $S_1$;
            - $S_1$: emit `1`, then arrive at $S_R$;
            - $S_R$: emit a fair random bit, then arrive at $S_0$.

            An edge $S_i \xrightarrow{x:p} S_j$ means the **joint** event

            $$
            \Pr(X_{t+1}=x,S_{t+1}=S_j\mid S_t=S_i)=p.
            $$

            We use row vectors for distributions. You will reconstruct the
            symbol matrices from the picture before they are revealed.
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

                arrow(r"$S_0$", r"$S_1$", "0 : 1", offset=(0, -.18))
                arrow(r"$S_1$", r"$S_R$", "1 : 1", offset=(.18, .10))
                arrow(r"$S_R$", r"$S_0$", r"0 : $\frac{1}{2}$", rad=.17,
                      offset=(-.22, .12))
                arrow(r"$S_R$", r"$S_0$", r"1 : $\frac{1}{2}$", rad=-.17,
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
            ### Reading convention

            Rows mean the phase **before** an edge and columns mean the phase
            **after** it. For symbol $x\in\{0,1\}$, define

            $$
            T^{(x)}_{ij}
            =\Pr(X_{t+1}=x,S_{t+1}=S_j\mid S_t=S_i).
            $$

            A symbol matrix is not generally row-stochastic: it contains only
            the joint events that emit that symbol. Normalization applies only
            after summing over both possible symbols.
            """
        ),
        md(
            r"""
            ## CORE 1/4 — Reconstruct the generator (10 minutes)

            1. Without looking at code, reconstruct the complete $3\times3$
               matrices $T^{(0)}$ and $T^{(1)}$ from the diagram.
            2. Derive the normalization identity

               $$
               \sum_{x\in\{0,1\}}\sum_j T^{(x)}_{ij}=1
               \quad\text{for each source state }i.
               $$

               Explain why it does **not** require either symbol matrix to have
               row sums equal to one.
            3. Let $T=T^{(0)}+T^{(1)}$. Verify that the uniform row vector
               $\pi=(1/3,1/3,1/3)$ is stationary: $\pi T=\pi$.
            4. Use the diagram or your matrices to calculate
               $\Pr(X_1=0)$ under $\pi$, keeping the two hidden-source
               contributions visible.

            <details><summary>Hint 1</summary>
            Each arrow contributes to exactly one entry of exactly one symbol
            matrix.
            </details>

            <details><summary>Hint 2</summary>
            Stationarity is a statement about the state-transition matrix
            $T$, after forgetting which symbol was emitted.
            </details>
            """
        ),
        response(
            r"""
            Reading each labelled edge gives

            $$
            T^{(0)}=
            \begin{pmatrix}
            0&1&0\\
            0&0&0\\
            1/2&0&0
            \end{pmatrix},
            \qquad
            T^{(1)}=
            \begin{pmatrix}
            0&0&0\\
            0&0&1\\
            1/2&0&0
            \end{pmatrix}.
            $$

            Summing over every mutually exclusive emitted-symbol/destination
            pair exhausts the possible next edges, which proves the
            normalization identity. For instance, the zero second row of
            $T^{(0)}$ is harmless because all of that row's mass sits in
            $T^{(1)}$.

            Adding the matrices gives
            $T=\left(\begin{smallmatrix}0&1&0\\0&0&1\\1&0&0\end{smallmatrix}\right)$.
            It cyclically permutes the three coordinates, so
            $\pi T=\pi$. Finally,

               $$
               \Pr(X_1=0)
               =\pi T^{(0)}\mathbf 1
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
            eta_empty = np.ones(3) / 3

            print("T^(0) =\n", T0)
            print("T^(1) =\n", T1)
            print("row sums after adding symbols:", T.sum(axis=(0, 2)))
            """
        ),
        md(
            r"""
            ### Notation card — keep this visible

            | Symbol | Meaning | Shape |
            |---|---|---|
            | $\pi=\eta^{(\epsilon)}$ | stationary belief before any observed symbol | $1\times3$ row |
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
            ## CORE 2/4 — From hidden paths to word probabilities (14 minutes)

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
               $\Pr(01)$, $\Pr(00)$, and $\Pr(010)$ for Z1R.
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

            For `01`, the only nonzero stationary path begins in $S_0$, so
            $\Pr(01)=1/3$. For `00`, the process must begin in $S_R$, emit its
            random zero, and then emit the certain zero from $S_0$, so
            $\Pr(00)=(1/3)(1/2)=1/6$. After `01` the process is at $S_R$,
            which emits `0` with probability $1/2$, so
            $\Pr(010)=1/6$.

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
                print(f"P({word}) = {word_probability(word):.6f}")
            """
        ),
        md(
            r"""
            ## CORE 3/4 — Bayes as a recursive geometric map (20 minutes)

            After observing $w$, define the **belief state**

            $$
            \eta^{(w)}_j=\Pr(S_{\lvert w\rvert}=S_j\mid X_{1:\lvert w\rvert}=w).
            $$

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
            3. Calculate $\eta^{(0)}$, $\eta^{(01)}$, and $\eta^{(10)}$ by
               composing these maps. Check explicitly that the order matters:
               $F_1(F_0(\pi))\ne F_0(F_1(\pi))$.
            4. Where do all three-component beliefs live geometrically? Why is
               $F_x$ generally rational rather than linear, even though its
               numerator is linear?

            <details><summary>Hint 1</summary>
            Bayes' rule divides the joint row
            $(\Pr(w,S_L=S_j))_j$ by its total mass.
            </details>

            <details><summary>Hint 2</summary>
            Treat the current posterior as the next step's prior. Compare
            histories `01` and `10`, rather than recomputing `01` twice.
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

            For `0` and the two orderings,

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
                label = r"$\epsilon$" if representative == "" else representative
                axes[0].scatter(*xy, s=55, color=b, edgecolor="black",
                                linewidth=.4, zorder=3)
                axes[0].annotate(label, xy, xytext=(4, 4),
                                 textcoords="offset points", fontsize=9)
            axes[0].set_title(
                "STATE SET — one point per distinct posterior\n"
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
                    axes[1].text(*midpoint, symbol, fontsize=11,
                                 color=symbol_colors[symbol],
                                 bbox=dict(boxstyle="round,pad=.12",
                                           fc="white", ec="none", alpha=.9))
                for word in path:
                    xy = belief(word) @ simplex_vertices
                    axes[1].scatter(*xy, s=58, color=belief(word),
                                    edgecolor="black", linewidth=.4, zorder=4)
                    axes[1].annotate(
                        r"$\epsilon$" if word == "" else word,
                        xy, xytext=(4, 4), textcoords="offset points",
                        fontsize=9,
                    )
            axes[1].set_title(
                "BAYES DYNAMICS — highlighted compositions\n"
                "blue arrow = observe 0; orange arrow = observe 1",
                fontsize=11,
            )
            fig.suptitle(
                "BELIEF GEOMETRY — the observer moves inside a simplex",
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
            The **mixed-state presentation** is the inference process whose
            states are reachable beliefs and whose symbol-labelled transitions
            are the maps $F_x$. The original HMM describes how the hidden world
            generates data; the MSP describes how an observer updates uncertainty
            about that world.

            The plot labels states by histories for convenience, but the state
            itself is the posterior vector. Two histories that yield the same
            posterior land at the same point.
            """
        ),
        md(
            r"""
            ## CORE 4/4 — Same next token, different future (15 minutes)

            From a belief $\eta$, the probability of a future word
            $u=u_1\ldots u_k$ is

            $$
            \Pr(u\mid\eta)=\eta T^{(u)}\mathbf 1.
            $$

            1. Derive the next-token map
               $\Pr(X_{\mathrm{next}}=x\mid\eta)=\eta T^{(x)}\mathbf 1$.
            2. Calculate the two beliefs $\eta^{(01)}$ and $\eta^{(10)}$.
            3. Show that both give the next-token distribution $(1/2,1/2)$.
            4. For each belief, calculate the distribution over
               `00`, `01`, `10`, and `11`.
            5. Explain precisely what fails if the next-token vector is the
               **only recursively retained state**. Why does this not imply that
               a full-context model must literally store an HMM posterior?

            <details><summary>Hint 1</summary>
            You already found $\eta^{(01)}$. For `10`, propagate one symbol at a
            time and normalize.
            </details>

            <details><summary>Hint 2</summary>
            Under $\eta^{(01)}$ the hidden state is certainly $S_R$; under
            $\eta^{(10)}$ it is an equal mixture of $S_0$ and $S_1$.
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
            \eta^{(01)}=(0,0,1),\qquad
            \eta^{(10)}=(1/2,1/2,0).
            $$

            The first is certainly at $S_R$; the second is equally likely to be
            at $S_0$ or $S_1$. Both emit `0` and `1` with probability $1/2$.
            Their two-step predictions differ:

            $$
            \begin{array}{c|cccc}
            &00&01&10&11\\\hline
            \eta^{(01)}&1/2&0&1/2&0\\
            \eta^{(10)}&0&1/2&1/4&1/4
            \end{array}
            $$

            If the next-token vector were the predictor's sole recursively
            retained state, these histories would be merged. Feeding the same
            next symbol into the same state-update rule would then produce the
            same later prediction, contradicting the table.

            This is an insufficiency result about a particular state
            compression. A full-context model can revisit the whole prefix, and
            any model may encode the needed distinction in coordinates other
            than our HMM posterior. The result motivates a search for richer
            predictive information; it does not dictate how a transformer must
            represent it.
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

            print("belief after 01:", eta_01)
            print("belief after 10:", eta_10)
            print("next-token distributions:", next_01, next_10)

            x = np.arange(len(future_words))
            fig, ax = plt.subplots(figsize=(7.2, 3.8))
            ax.bar(x - .18, two_01, .36, label=r"after $01$")
            ax.bar(x + .18, two_10, .36, label=r"after $10$")
            ax.set(xticks=x, xticklabels=future_words, ylim=(0, .58),
                   ylabel="conditional probability",
                   title="Same one-step prediction; different two-step future")
            ax.legend()
            plt.show()
            """
        ),
        md(
            r"""
            ## CORE SYNTHESIS — Build the Mess3 recursion (10 minutes)

            Z1R has only a small set of recurrent beliefs. The same update rule
            can instead generate a fractal set. Mess3 still has only three
            **hidden generator states**, but it has three observation symbols and
            three contracting, renormalizing belief maps. For the parameters used
            here, its visible symbol matrices are

            $$
            T^{(0)}=
            \begin{pmatrix}
            .14&.06&.06\\ .03&.28&.06\\ .03&.06&.28
            \end{pmatrix},\quad
            T^{(1)}=
            \begin{pmatrix}
            .28&.03&.06\\ .06&.14&.06\\ .06&.03&.28
            \end{pmatrix},
            $$

            $$
            T^{(2)}=
            \begin{pmatrix}
            .28&.06&.03\\ .06&.28&.03\\ .06&.06&.14
            \end{pmatrix},\qquad
            \eta^{(\epsilon)}=(1/3,1/3,1/3).
            $$

            For a simplex vertex $e_i$ (certainty about hidden state $i$),

            $$
            F_x(e_i)=\frac{e_iT^{(x)}}{e_iT^{(x)}\mathbf1}
            $$

            is just the normalized $i$th row of $T^{(x)}$. The first supplied
            visual draws only the three image triangles $F_x(\Delta^2)$; it does
            **not** reveal the length-two answers.
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
                center = mapped_vertices.mean(axis=0)
                ax.text(
                    *center, rf"$F_{symbol}(\Delta^2)$",
                    ha="center", va="center", fontsize=10,
                    color=map_colors[symbol], fontweight="bold",
                )

            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(
                "ONE SYMBOL — normalized rows give each image triangle\n"
                "Length-two histories are still hidden",
                fontsize=12,
            )
            plt.show()
            """
        ),
        md(
            r"""
            Now do one composition before the reveal.

            1. Normalize the three rows of $T^{(0)}$ to calculate
               $F_0(e_0)$, $F_0(e_1)$, and $F_0(e_2)$. Match them to the
               vertices of the blue image triangle.
            2. Starting from the uniform belief, calculate $\eta^{(0)}$ and then
               $\eta^{(01)}=F_1(\eta^{(0)})$.
            3. On the one-symbol visual, identify the image triangle that must
               contain $\eta^{(01)}$. Explain why the **last** observed symbol,
               rather than the first, determines that outer triangle.
            4. Sketch the qualitative placement of the other eight length-two
               histories. Then predict what repeated compositions will do.
            5. Does having three hidden generator states constrain the observer
               to only three predictive states?

            <details><summary>Hint</summary>
            For a history `01`, apply $F_0$ first and $F_1$ second. Every point
            produced by the second map lies in $F_1(\Delta^2)$.
            </details>
            """
        ),
        response(
            r"""
            Normalizing the rows of $T^{(0)}$ gives

            $$
            F_0(e_0)=(7,3,3)/13,\quad
            F_0(e_1)=(3,28,6)/37,\quad
            F_0(e_2)=(3,6,28)/37.
            $$

            From the uniform initial belief,

            $$
            \eta^{(0)}=(1/5,2/5,2/5).
            $$

            Propagating through $T^{(1)}$ gives an unnormalized row
            $(.104,.074,.148)$, hence

            $$
            \eta^{(01)}=(52,37,74)/163
            \approx(.319,.227,.454).
            $$

            Because $F_1$ is applied last, this point must lie inside the orange
            triangle $F_1(\Delta^2)$. More generally, histories sharing their
            last symbol occupy the same outer image triangle; the earlier symbol
            selects a smaller image within it. Repetition nests these images and
            produces many, potentially fractal, observer states. Three hidden
            generator states fix the ambient simplex, not the number of reachable
            posterior beliefs.
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
                    "".join(map(str, word)), xy, xytext=(4, 3),
                    textcoords="offset points", fontsize=8,
                )
            axes[0].set_title(
                "TWO SYMBOLS — nine compositions\n"
                "last symbol selects the outer image",
                fontsize=11,
            )

            axes[1].scatter(
                XY[:, 0], XY[:, 1], c=np.clip(B, 0, 1),
                s=6, alpha=.72, linewidth=0,
            )
            axes[1].set_title(
                f"SEVEN SYMBOLS — {len(B):,} reachable posteriors\n"
                "RGB color = hidden-state belief",
                fontsize=11,
            )
            fig.suptitle(
                "BELIEF GEOMETRY — three Bayes maps recursively carve a fractal",
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
            ## Synthesis

            Complete the chain in words, not just symbols:

            $$
            \text{HMM edges}
            \longrightarrow \Pr(w)
            \longrightarrow \eta^{(w)}
            \longrightarrow F_x
            \longrightarrow \text{belief geometry}
            \longrightarrow \Pr(\text{future}\mid\eta).
            $$

            Which arrows are linear, which require normalization, and which
            describe a change of interpretation rather than a new calculation?
            """
        ),
        response(
            r"""
            Symbol-labelled matrices encode the edges. Their products sum hidden
            paths and give word probabilities. Keeping the final hidden state
            unsummed gives an unnormalized forward vector; dividing by its mass
            turns it into a posterior belief. Reusing that posterior as the next
            prior gives the normalized update maps $F_x$. Iterating those maps
            produces geometry in the simplex. Finally, multiplying a belief by
            future symbol matrices and summing gives any future probability.

            Matrix propagation and the belief-to-future map are linear before
            normalization. Bayesian updates are generally nonlinear because
            their denominators depend on the current belief. “Belief geometry”
            is the same collection of posteriors viewed geometrically, not a
            separate probabilistic object.
            """,
            show_solutions,
        ),
        md(
            r"""
            ## Optional design studio — tune rather than start from nothing

            The supplied family has a cyclic `0` skeleton:

            $$
            S_0\xrightarrow{0:a}S_1,\quad
            S_1\xrightarrow{0:b}S_2,\quad
            S_2\xrightarrow{0:c}S_0,
            $$

            with a `1` self-loop carrying the remaining probability at each
            state.

            Choose one brief and its starting preset:

            | Brief | Start from $(a,b,c)$ | Operational target |
            |---|---|---|
            | **fast synchronization** | $(.95,.55,.08)$ | median posterior entropy below $0.35$ nats by depth 8 |
            | **persistent ambiguity** | $(.62,.52,.42)$ | median entropy above $0.70$ nats and no single coordinate above $.95$ |
            | **branching geometry** | $(.88,.50,.15)$ | the centroids after final symbol `0` versus `1` are far apart |

            1. Predict which parameters make an observed `0` most diagnostic.
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
        md(
            r"""
            ### Exit ticket

            In two sentences: why can prediction be geometrically more
            complicated than generation, even when the data-generating HMM has
            only three hidden states?
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

            This notebook is a **guided discovery fiction**: it reconstructs a
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

            Five numbered **CORE** tasks plus the Mess3 simulation form a
            60–70 minute notebook path, leaving room in the 90-minute block for
            the introduction, transitions, and buffer. The fifth task is the
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
                    ax.text((x1+x2)/2, (y1+y2)/2+.18, label,
                            ha="center", fontsize=9, color="#47546b")

            box(.2, 1.55, 1.65, .72, r"full prefix $h$")
            box(2.35, 1.38, 2.1, 1.05,
                "transformer\n(all prefix tokens)")
            box(5.05, 1.38, 2.25, 1.05,
                r"last-position residual"+"\n"+r"$a_\ell(h)\in\mathbb{R}^d$")
            box(8.0, 2.25, 1.7, .72, "unembedding")
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
              **last token position**, after layer $\ell$.
            - the **unembedding** maps a final residual vector to current
              next-token logits;
            - $b(h)\in\Delta^{m-1}$: the exact posterior over the $m$ hidden
              states of the chosen generator;
            - a **probe** is a separate diagnostic map fit after training. It is
              not part of the transformer's forward pass.

            A transformer with the full prefix can recompute information from
            earlier tokens at every position. Therefore the argument below does
            not force it to maintain one recursive state. Belief is one compact
            sufficient representation of predictive history; whether the model
            learns an affinely related representation is an empirical question.
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
            final row, so $\Theta\in\mathbb R^{(d+1)\times m}$. Then

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
            4. Compare the activation probe with four baselines: mean belief,
               last-token-plus-length, exact next-token probabilities, and
               shuffled belief labels.
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
               whether the probe contains predictive distinctions beyond the
               optimal next-token distribution. Shuffling tests whether arbitrary
               high-dimensional projections can paint the target set.

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
            also compared with simple covariate and next-token baselines.

            **This is a teaching simulation, not transformer activations, not
            paper data, and not the paper's generator parameters.** Its purpose
            is to make the regression and controls inspectable without asking
            you to train a model.

            Before running it, sketch:

            - the held-out MSE curve you expect;
            - what the earliest and latest recovered geometry should look like;
            - what a shuffled-label projection should do;
            - whether Mess3's exact next-token probabilities might already
              reveal much of its belief.
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
            next_theta = fit_probe(next_token[train], B[train])
            next_hat = apply_probe(next_token[test], next_theta)

            shuffled_B = B[rng.permutation(n)]
            shuffled_theta = fit_probe(final_A[train], shuffled_B[train])
            shuffled_hat = apply_probe(final_A[test], shuffled_theta)

            baseline_names = [
                "mean",
                "token +\nlength",
                "next-token\nonly",
                "activation\nprobe",
                "shuffled\nlabels",
            ]
            baseline_mses = [
                np.mean((mean_hat - B[test])**2),
                np.mean((covariate_hat - B[test])**2),
                np.mean((next_hat - B[test])**2),
                mses[-1],
                np.mean((shuffled_hat - B[test])**2),
            ]

            fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))
            axes[0].bar(
                np.arange(len(baseline_mses)), baseline_mses,
                color=["#9aa4b6", "#9aa4b6", "#d39a3a", "#3976a8", "#b55d5d"],
            )
            axes[0].set(
                xticks=np.arange(len(baseline_names)),
                xticklabels=baseline_names,
                ylabel="held-out belief MSE",
                title="BASELINE COMPARISON",
            )
            axes[0].tick_params(axis="x", labelsize=8)
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
                "If next-token-only is already strong on Mess3, Mess3 alone "
                "cannot establish representation beyond the optimal "
                "next-token distribution.",
                ha="center", fontsize=9,
            )
            plt.tight_layout(rect=(0, .06, 1, .90))
            plt.show()

            print("Training subtrees:", sorted(train_subtrees))
            print("Held-out subtree fraction:", len(test) / n)
            for name, value in zip(baseline_names, baseline_mses):
                print(name.replace("\n", " "), "MSE:", value)
            """
        ),
        md(
            r"""
            ### Interpret, do not merely admire

            Compare your predictions with the plots.

            1. Why are the recovered points allowed to fall slightly outside the
               simplex?
            2. Why color recovered points by their **ground-truth** beliefs
               rather than by their recovered coordinates?
            3. Why is a held-out prefix-subtree split more informative than a
               random point split here?
            4. If the next-token-only baseline nearly matches the activation
               probe on Mess3, what experiment should become central next?
            5. What aspect of the simulation is deliberately unrealistic?
            6. What would you want to inspect before trusting the same pipeline
               on real transformer activations?
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
            across a meaningful context family. If next-token probabilities
            already decode Mess3 belief, the central test should use histories
            or a process deliberately matched on optimal next-token
            distributions but separated on longer futures.

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
            - record 64-dimensional final residual-stream activations;
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

            The paper also studies Random–Random–XOR (RRXOR), whose 36 belief
            states include many pairs with the same next-token prediction.

            Z1R already supplied the logic: histories `01` and `10` have equal
            current next-token vectors, while test `00` separates their longer
            futures. A representation that separates such matched pairs cannot
            be explained by optimal next-token distributions alone. RRXOR scales
            that diagnostic idea to a richer finite belief set.

            Before reading the result:

            1. If final-layer activations feed directly into an unembedding for
               the current next token, where would you expect distinctions
               irrelevant to the predicted next-token distribution to become
               weaker?
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
            representation separates predictive states even where current
            next-token probabilities are matched.
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

            An HMM belief is a posterior over the hidden states of a **chosen
            model**. Can we define predictive state directly from observable
            word probabilities instead?

            In the core path you will:

            - use consistency constraints to interrogate a table of word
              probabilities without mistaking bookkeeping for the main idea;
            - arrange those probabilities into a Hankel matrix;
            - distinguish joint rows from conditional predictive rows;
            - separate a generic finite-block dependency from evidence for a
              finite predictive dimension;
            - choose future tests as predictive-state-representation (PSR)
              coordinates;
            - connect the observable construction back to HMM beliefs without
              treating latent states as ground truth.

            Five numbered **CORE** tasks (50 minutes total) and an eight-minute
            HMM factorization form the complete 55–65 minute path. Anything
            marked **STRETCH**, including the Z1R **BELIEF ↔ PSR BRIDGE**, is
            modular. The final WFA section is an instructor demo: students
            interpret one surprising operator rather than spend the core doing
            determinant or matrix-inversion arithmetic.
            """
        ),
        seneca_epigraph(),
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
                ("word\nprobabilities", "what is observed"),
                ("Hankel\nrows", r"$H_{u,v}=P(uv)$"),
                ("rank + row\nrelations", "block evidence ≠ full rank"),
                ("core future\ntests / PSR", "predictive coordinates"),
                ("symbol\noperators", "optional WFA demo"),
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
                "Question carried through every box: "
                "how much predictive memory does the process require?",
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
            ## Start from observations, not a hidden model

            A stationary binary process has supplied the table below. No hidden
            states have been specified: $\Pr(w)$ is the primitive observable.
            Two entries are concealed so that you use the constraints once,
            without spending the section on fraction bookkeeping.

            | length | observed word probabilities |
            |---|---|
            | 0 | $\Pr(\epsilon)=1$ |
            | 1 | $\Pr(0)=2/3,\quad\Pr(1)=1/3$ |
            | 2 | $\Pr(00)=1/2,\quad\Pr(01)=?,\quad\Pr(10)=1/6,\quad\Pr(11)=1/6$ |
            | 3 | $\Pr(000)=3/8,\ \Pr(001)=1/8,\ \Pr(010)=1/12,\ \Pr(011)=1/12$ |
            |   | $\Pr(100)=1/8,\ \Pr(101)=?,\ \Pr(110)=1/12,\ \Pr(111)=1/12$ |

            For a stationary process, a word can be extended on either side:

            $$
            \Pr(u)=\Pr(u0)+\Pr(u1)
            =\Pr(0u)+\Pr(1u).
            $$
            """
        ),
        md(
            r"""
            ## CORE 1/5 — Interrogate the observable table (8 minutes)

            1. Use right-extension consistency to recover only the two missing
               entries $\Pr(01)$ and $\Pr(101)$.
            2. Check left-extension consistency in two nontrivial places:
               $\Pr(00)=\Pr(000)+\Pr(100)$ and
               $\Pr(10)=\Pr(010)+\Pr(110)$. What assumption makes these
               identities appropriate?
            3. Calculate $\Pr(0\mid0)$ and $\Pr(0\mid1)$. Propose—not
               prove—the simplest continuation rule suggested by them.
            4. **STRETCH — continuation check.** Under that hypothesis, predict
               the withheld value $\Pr(1010)$. Then reveal the check cell.

            <details><summary>Hint</summary>
            Under a first-order continuation,
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

            The one-step conditionals are

            $$
            \Pr(0\mid0)=\frac{1/2}{2/3}=\frac34,\qquad
            \Pr(0\mid1)=\frac{1/6}{1/3}=\frac12.
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
                "1010": Fraction(1, 48),
            }
            for length in range(4):
                layer = {w: p for w, p in word_p.items() if len(w) == length}
                print(f"length {length}:", layer,
                      "sum =", sum(layer.values(), Fraction(0)))
            print("WITHHELD CHECK — P(1010) =", word_p["1010"])
            """
        ),
        md(
            r"""
            ## CORE 2/5 — What can this Hankel block establish? (10 minutes)

            A **process Hankel matrix** has histories as rows, future tests as
            columns, and concatenated word probabilities as entries:

            $$
            H_{u,v}=\Pr(uv).
            $$

            Use the ordered prefix and suffix sets
            $\mathcal P=\mathcal S=\{\epsilon,0,1\}$.

            1. Fill the $3\times3$ block $H$. Explain why its first column is not
               all ones.
            2. Use right-extension consistency to show that, for **any** binary
               process and every displayed history $u$,

               $$
               H_{u,\epsilon}=H_{u,0}+H_{u,1}.
               $$

               What upper bound does this generic column dependence put on the
               rank of this particular block?
            3. Show that the upper-left $2\times2$ minor is nonzero. Combine the
               lower and upper bounds to determine the block's rank.
            4. A finite Hankel block is a submatrix of the full infinite Hankel
               matrix. Which bound on the **full** rank follows from this
               calculation? Why does no full-rank upper bound follow yet?

            <details><summary>Hint 1</summary>
            The entry in row `0`, column `1` is $\Pr(01)$, not
            $\Pr(1\mid0)$.
            </details>

            <details><summary>Hint 2</summary>
            The empty suffix gives $H_{u,\epsilon}=\Pr(u)$, while the next symbol
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

            Its first column contains prefix probabilities because appending the
            empty word changes nothing: $\Pr(u\epsilon)=\Pr(u)$. Moreover,

            $$
            \Pr(u)=\Pr(u0)+\Pr(u1),
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
                        xlabel="future test $v$", ylabel="history $u$",
                        title=r"$H_{u,v}=P(uv)$")
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
            ## CORE 3/5 — Normalize before comparing predictions (10 minutes)

            A joint row is weighted by how likely its history was. A predictive
            row instead contains conditional future probabilities:

            $$
            K_{u,v}=\Pr(v\mid u)=\frac{H_{u,v}}{\Pr(u)}
            \quad\text{when }\Pr(u)>0.
            $$

            1. Normalize the rows for histories $\epsilon$, `0`, and `1`.
            2. Form the additional joint row for history `10` and suffixes
               $\{\epsilon,0,1\}$ using the length-three table.
            3. Show both

               $$
               H_{10,\cdot}=\frac14H_{0,\cdot}
               \quad\text{and}\quad
               K_{10,\cdot}=K_{0,\cdot}.
               $$

            4. What is the difference between those two equalities?

            <details><summary>Hint</summary>
            $H_{10,\cdot}=(\Pr(10),\Pr(100),\Pr(101))$.
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
            """,
            show_solutions,
        ),
        md(
            r"""
            ## CORE 4/5 — Predictive equivalence and its caveat (10 minutes)

            Define two histories to be **predictively equivalent** when

            $$
            u\sim u'
            \quad\Longleftrightarrow\quad
            \Pr(v\mid u)=\Pr(v\mid u')
            \ \text{for every finite future word }v.
            $$

            1. The finite table suggests that histories `0` and `10` are
               equivalent. Why does matching on the displayed suffixes alone
               not prove the definition?
            2. Inspect the completed probabilities and hypothesize a simple rule
               for the process: what feature of the past appears to determine
               the distribution of the next symbol?
            3. Under that hypothesis, identify the recurrent predictive states
               and label their symbol-conditioned transitions with probabilities.
               **STRETCH:** draw the graph before revealing the supplied version.
            4. Is the empty history a third recurrent predictive state?

            <details><summary>Hint</summary>
            Compare histories ending in `0` with histories ending in `1`.
            Observing the next symbol also tells you which class the new history
            belongs to.
            </details>
            """
        ),
        response(
            r"""
            A finite block tests only finitely many futures. Two rows could agree
            there and differ on a longer suffix, so full equivalence needs either
            all future tests or additional structural knowledge.

            The table is consistent with a first-order binary Markov source:

            $$
            \Pr(0\mid\text{last }0)=3/4,\qquad
            \Pr(0\mid\text{last }1)=1/2.
            $$

            Thus histories ending in `0` form predictive state $C_0$ and those
            ending in `1` form $C_1$:

            $$
            C_0\xrightarrow{0:3/4}C_0,\quad
            C_0\xrightarrow{1:1/4}C_1,
            $$

            $$
            C_1\xrightarrow{0:1/2}C_0,\quad
            C_1\xrightarrow{1:1/2}C_1.
            $$

            The empty history carries the stationary mixture
            $2/3\,C_0+1/3\,C_1$; it is an initial predictive mixture, not a third
            recurrent class.
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
            ## CORE 5/5 — Choose predictions as coordinates (12 minutes)

            A **predictive state representation** uses the probabilities of a
            selected set of future experiments, or **core tests**, as state
            coordinates.

            Choose $\mathcal Q=\{\epsilon,0\}$ and define the normalized
            predictive state

            $$
            q(u)=\big(\Pr(\epsilon\mid u),\Pr(0\mid u)\big)=(1,p).
            $$

            1. Give $p$ for $C_0$, $C_1$, and the empty history.
            2. Express $\Pr(00\mid u)$ and $\Pr(10\mid u)$ in terms of $p$.
               **STRETCH:** derive the complementary `01` and `11` probabilities
               and verify that all four sum to one.
            3. Derive the updated value $p'$ after observing `0`, and after
               observing `1`.
            4. Why does the rank-two linear representation have only one free
               coordinate after normalization?

            <details><summary>Hint 1</summary>
            Condition on the first future symbol, then use the appropriate row
            of the two-state predictive machine.
            </details>

            <details><summary>Hint 2</summary>
            In general,
            $\Pr(v\mid ux)=\Pr(xv\mid u)/\Pr(x\mid u)$.
            </details>
            """
        ),
        response(
            r"""
            The coordinate $p=\Pr(0\mid u)$ is $3/4$ in $C_0$, $1/2$ in $C_1$,
            and $2/3$ initially. Conditioning on the first future symbol gives

            $$
            \Pr(00\mid u)=\frac34p,\qquad
            \Pr(01\mid u)=\frac14p,
            $$

            $$
            \Pr(10\mid u)=\frac12(1-p),\qquad
            \Pr(11\mid u)=\frac12(1-p).
            $$

            After observing `0`, the new history ends in `0`, so $p'=3/4$.
            After `1`, it ends in `1`, so $p'=1/2$. Equivalently these follow
            by dividing the appropriate extended-test probabilities by
            $\Pr(x\mid u)$.

            The unnormalized linear state needs one coordinate for total mass
            and one for predictive variation. Conditional states fix the mass
            coordinate $\Pr(\epsilon\mid u)=1$, leaving a one-dimensional affine
            set.
            """,
            show_solutions,
        ),
        md(
            r"""
            ## CORE SYNTHESIS — Factorization, not ontology (8 minutes)

            The finite table did not logically force a unique continuation to
            all word lengths. From here, adopt the simplest first-order
            continuation you reconstructed. It can be realized with two
            edge-emitting matrices, using states $C_0,C_1$:

            $$
            T^{(0)}=
            \begin{pmatrix}3/4&0\\1/2&0\end{pmatrix},
            \qquad
            T^{(1)}=
            \begin{pmatrix}0&1/4\\0&1/2\end{pmatrix},
            \qquad
            \pi=(2/3,1/3).
            $$

            For any history $u$ and future test $v$, derive

            $$
            H_{u,v}
            =\Pr(uv)
            =\underbrace{\pi T^{(u)}}_{\text{history row}}
             \underbrace{T^{(v)}\mathbf 1}_{\text{future column}}.
            $$

            What rank bound follows for the **full infinite Hankel matrix**?
            What additional statement would be unsafe to infer from finite rank
            alone?
            """
        ),
        response(
            r"""
            The word formula splits at the boundary between history and future:

            $$
            \Pr(uv)=\pi T^{(u)}T^{(v)}\mathbf1.
            $$

            Collecting $\pi T^{(u)}$ for all histories as rows and
            $T^{(v)}\mathbf1$ for all tests as columns factors the **adopted
            first-order process's full Hankel matrix** through a two-dimensional
            space, so its full rank is at most two. Separately, the displayed
            finite block is a submatrix with rank two, so the full rank is at
            least two. These two logically distinct bounds establish that the
            adopted continuation has full Hankel rank exactly two. The finite
            table alone did not supply the upper bound.

            More generally, finite Hankel rank guarantees a finite-dimensional
            linear realization. It does not by itself guarantee that a
            same-sized realization has nonnegative entries and stochastic HMM
            semantics; nonnegative realization requires additional conditions.
            """,
            show_solutions,
        ),
        md(
            r"""
            ## STRETCH — BELIEF ↔ PSR BRIDGE: longer tests recover Z1R belief (15–25 minutes)

            Return to fair Z1R. Order the belief coordinates as
            $(a,b,c)$ over $(S_0,S_1,S_R)$. Choose two observable tests:

            $$
            q_0=\Pr(0\mid h),\qquad q_{00}=\Pr(00\mid h).
            $$

            1. Derive $q_0$ and $q_{00}$ as functions of $(a,b,c)$.
            2. Invert those equations to recover $(a,b,c)$ from
               $(q_0,q_{00})$.
            3. Evaluate the coordinates after histories `01` and `10`. Which
               test separates the beliefs that one-step prediction merged?
            4. Derive the coordinate update after observing `0`. Stretch:
               derive it after `1`.
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

            After observing `0`, Bayes' rule in predictive coordinates gives

            $$
            F_0(q_0,q_{00})
            =\left(\frac{q_{00}}{q_0},0\right).
            $$

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

            # Optional WFA time-buffer branch

            Stop here if the group needs consolidation time. The remaining
            instructor demo shows how a finite Hankel basis yields linear
            symbol-update operators. Students need only verify one word and
            interpret the negative entry.
            """
        ),
        md(
            r"""
            ## INSTRUCTOR DEMO — Read observable symbol operators (8–10 minutes)

            Use basis prefixes and suffixes
            $\mathcal B=\{\epsilon,0\}$. Define

            $$
            B_{p,s}=\Pr(ps),\qquad
            (B_x)_{p,s}=\Pr(pxs).
            $$

            Reading the supplied word table gives

            $$
            B=
            \begin{pmatrix}1&2/3\\2/3&1/2\end{pmatrix},
            \quad
            B_0=
            \begin{pmatrix}2/3&1/2\\1/2&3/8\end{pmatrix},
            \quad
            B_1=
            \begin{pmatrix}1/3&1/6\\1/6&1/12\end{pmatrix}.
            $$

            The instructor now stacks the equations
            $B_{p,\cdot}A_x=(B_x)_{p,\cdot}$ and solves $BA_x=B_x$, obtaining

            $$
            A_0=
            \begin{pmatrix}0&0\\1&3/4\end{pmatrix},\qquad
            A_1=
            \begin{pmatrix}1&1/2\\-1&-1/2\end{pmatrix}.
            $$

            Let $\alpha=(1,2/3)$ be the empty-prefix row and let
            $\omega=(1,0)^\top$ select the empty-suffix column.

            1. Verify the single word

               $$
               \Pr(01)=\alpha A_0A_1\omega.
               $$

            2. $A_1$ contains negative entries. Why are these not negative
               transition probabilities? What observable condition must the
               complete calculation still satisfy?

            <details><summary>STRETCH — where did the operators come from?</summary>
            Each row of $B$ represents a basis prefix. Appending $x$ should turn
            it into the corresponding row of $B_x$, so stacking the row
            equations gives $BA_x=B_x$ and $A_x=B^{-1}B_x$. The supplied check
            cell performs this solve and tests several words.
            </details>
            """
        ),
        response(
            r"""
            Reading the entries from the word table gives

            $$
            B=
            \begin{pmatrix}1&2/3\\2/3&1/2\end{pmatrix},
            \qquad
            B_0=
            \begin{pmatrix}2/3&1/2\\1/2&3/8\end{pmatrix},
            \qquad
            B_1=
            \begin{pmatrix}1/3&1/6\\1/6&1/12\end{pmatrix}.
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

            For example,

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
            are allowed as long as every complete word evaluation is a valid
            nonnegative probability. This is a weighted finite automaton (or
            observable operator realization), not automatically an HMM.
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
            ## Final synthesis

            Complete this comparison:

            | Object | Coordinates refer to | Update type | Canonical caveat |
            |---|---|---|---|
            | HMM belief | posterior over a chosen model's hidden states | normalized $\,\eta T^{(x)}\,$ | model-relative; may be redundant |
            | Conditional Hankel row | probabilities of all future tests | conditioning by $\Pr(x\mid h)$ | infinite object |
            | Finite PSR | probabilities of selected core tests | generally rational after normalization | tests must span the predictive space |
            | WFA state | a finite linear coordinate basis | linear symbol operators | coordinates/operators need not be nonnegative |

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
