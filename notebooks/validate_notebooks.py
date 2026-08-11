"""Lightweight structural and execution checks for the generated notebooks."""

from __future__ import annotations

import json
import os
import re
import tempfile
import traceback
from pathlib import Path


HERE = Path(__file__).resolve().parent

EXPECTED_NOTEBOOKS = [
    "01_hmms_msps_exercises.ipynb",
    "01_hmms_msps_solutions.ipynb",
    "02_transformer_belief_geometry_exercises.ipynb",
    "02_transformer_belief_geometry_solutions.ipynb",
    "03_hankel_psr_wfa_exercises.ipynb",
    "03_hankel_psr_wfa_solutions.ipynb",
]
PAIR_STEMS = [
    "01_hmms_msps",
    "02_transformer_belief_geometry",
    "03_hankel_psr_wfa",
]
CORE_TOTALS = {"01_": 4, "02_": 5, "03_": 4}
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def check_local_links(path: Path, source: str) -> None:
    for raw_target in MARKDOWN_LINK.findall(source):
        target = raw_target.strip().strip("<>")
        if target.startswith(("http://", "https://", "mailto:", "data:", "#")):
            continue
        target = target.split("#", 1)[0]
        assert target, f"{path.name} contains an empty local link"
        resolved = (path.parent / target).resolve()
        assert resolved.exists(), f"{path.name} has a broken local link: {target}"


def validate(path: Path, plt: object) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["nbformat"] == 4
    assert isinstance(data["cells"], list) and data["cells"]
    ids = [cell["id"] for cell in data["cells"]]
    assert len(ids) == len(set(ids)), f"{path.name} contains duplicate cell IDs"
    image_count = sum(
        1
        for cell in data["cells"]
        for output in cell.get("outputs", [])
        if "image/png" in output.get("data", {})
    )
    assert image_count > 0, f"{path.name} has no embedded visual fallback"

    namespace: dict[str, object] = {
        "__name__": "__notebook__",
        "plt": plt,
    }
    code_count = 0
    execution_counts: list[int | None] = []
    all_source = ""
    for index, cell in enumerate(data["cells"]):
        assert cell["cell_type"] in {"markdown", "code"}
        source = "".join(cell["source"])
        all_source += source + "\n"
        assert source.strip(), f"{path.name} cell {index} is empty"
        if cell["cell_type"] == "markdown":
            check_local_links(path, source)
        tags = cell.get("metadata", {}).get("tags", [])
        if "solution" in tags:
            assert source.startswith("#### Worked solution\n"), (
                f"{path.name} solution cell {index} is incorrectly indented"
            )
        if "provided-code" in tags:
            assert cell["metadata"].get("jupyter", {}).get("source_hidden"), (
                f"{path.name} provided code cell {index} is not collapsed"
            )
        if cell["cell_type"] != "code":
            continue
        code_count += 1
        execution_counts.append(cell.get("execution_count"))
        try:
            compiled = compile(source, f"{path.name}:cell-{index}", "exec")
            exec(compiled, namespace)
        except Exception as exc:
            print(f"\nFAILED: {path.name}, code cell {index}: {exc}")
            traceback.print_exc()
            raise
        finally:
            for figure_number in plt.get_fignums():  # type: ignore[union-attr]
                plt.figure(figure_number).canvas.draw()  # type: ignore[union-attr]
            plt.close("all")  # type: ignore[union-attr]

    assert execution_counts == list(range(1, code_count + 1)), (
        f"{path.name} embedded execution counts are stale or incomplete"
    )

    expected_total = next(
        total for prefix, total in CORE_TOTALS.items() if path.name.startswith(prefix)
    )
    core_headings = [
        (int(index), int(total))
        for index, total in re.findall(r"## CORE (\d+)/(\d+)", all_source)
    ]
    assert core_headings == [
        (index, expected_total) for index in range(1, expected_total + 1)
    ], f"{path.name} has an incomplete or inconsistent numbered CORE route"

    normalized = " ".join(all_source.lower().split())
    assert ",qquad" not in all_source, (
        f"{path.name} contains a malformed LaTeX spacing command"
    )
    epigraph_texts = (
        [
            "ἀεὶ ὁ θεὸς γεωμετρεῖ",
            "god always geometrizes",
            "attributed to **plato**",
            "**plutarch**",
            "moralia* 718b–c",
        ]
        if path.name.startswith("02_")
        else [
            "longum iter est per praecepta, breve et efficax per exempla",
            "multum interest utrum non velit an nesciat",
            "velle non discitur",
            "seneca the younger",
            "moral letters to lucilius",
        ]
    )
    for epigraph_text in epigraph_texts:
        assert epigraph_text in normalized, (
            f"{path.name} is missing required epigraph text: "
            f"{epigraph_text}"
        )

    if path.name.startswith("01_"):
        core_2 = normalized.index("## core 2/4")
        core_3 = normalized.index("## core 3/4")
        core_4 = normalized.index("## core 4/4")
        belief_update = normalized.index("derive the recursive map")
        future_word = normalized.index("a whole new future word")
        assert core_2 < core_3 < belief_update < core_4 < future_word, (
            f"{path.name} introduces belief or future-word calculations "
            "before their designated core section"
        )
        assert "when $\\pr(u\\mid h)>0$" in normalized, (
            f"{path.name} omits the domain of the future-posterior identity"
        )

    if path.name.startswith("02_"):
        for label in [
            "act i — from a prediction collision to a representation question",
            "the collision — correct now is not enough to stay correct",
            "one-step objective, multi-step representational burden",
            "invent the experiment before seeing the paper's version",
            "reveal — compare your experiment with the paper's",
            "the actual test bed",
            "shai et al. (2024), figure 5b",
            "shai et al. (2024), figure 5a",
            "shai et al. (2024), figure 3",
            "shai et al. (2024), figure 4",
            "shai et al. (2024), figure 7",
            "shai et al. (2024), figures 5, 6, and 1",
            "local neurips pdf",
            "teaching simulation",
            "not transformer activations",
            "not paper data",
            "not the paper's generator parameters",
            "source boundary",
            "core synthesis — the discovery chain",
            "# optional extensions",
            "optional algebra + code lab",
            "observed symbol $x$ has positive conditional probability",
        ]:
            assert label in normalized, (
                f"{path.name} is missing simulation qualifier: {label}"
            )
        core_1 = normalized.index("## core 1/5")
        core_2 = normalized.index("## core 2/5")
        core_3 = normalized.index("## core 3/5")
        assert normalized.index(
            "this is not yet a theorem about transformers"
        ) < core_1, f"{path.name} delays the transformer-scope caveat"
        if "exercises" in path.name:
            discovery_prefix = normalized[:core_2]
            for leaked_answer in ["affine", "activation-to-belief"]:
                assert leaked_answer not in discovery_prefix, (
                    f"{path.name} reveals {leaked_answer!r} before CORE 2"
                )
        assert core_2 < normalized.index(
            "the actual test bed"
        ) < core_3, (
            f"{path.name} reveals the completed probe before students invent it"
        )
        assert core_2 < normalized.index(
            "source visual — shai et al. (2024), figure 3"
        ) < core_3, (
            f"{path.name} reveals the paper's target before students choose it"
        )
        assert normalized.index("the actual test bed") < normalized.index(
            "source visual — shai et al. (2024), figure 4"
        ) < core_3, (
            f"{path.name} places the paper's method figure outside its reveal"
        )
        assert normalized.index("## core 5/5") < normalized.index(
            "evidence visual — shai et al. (2024), figure 7"
        ) < normalized.index("## optional paper checkpoint"), (
            f"{path.name} reveals RRXOR evidence before CORE 5"
        )
        assert normalized.index("## optional paper checkpoint") < normalized.index(
            "evidence visuals — shai et al. (2024), figures 5, 6, and 1"
        ), f"{path.name} places the Mess3 evidence gallery too early"
        synthesis = normalized.index("## core synthesis — the discovery chain")
        assert "derive the normal equations" not in normalized[:synthesis], (
            f"{path.name} interleaves optional algebra with the core route"
        )
        assert synthesis < normalized.index(
            "# optional extensions"
        ) < normalized.index("## optional algebra + code lab"), (
            f"{path.name} interleaves optional mechanics with the core route"
        )

    if path.name.startswith("03_"):
        for guardrail in [
            "55-minute task budget",
            "dimension of a linear span, not generally the number",
            "whenever that observation has positive probability",
            "past embedding",
            "future embedding",
            "eckart–young–mirsky",
            "infinite structured hankel approximation (aak)",
            "it does **not** have to be a probability",
            "optional extension c — control theory",
            "balanced truncation theorem card",
            "sequences of logits reveal the low rank structure",
            "input switched affine networks",
            "optional wfa recap",
            "valid when $\\pr(hx)>0$",
        ]:
            assert guardrail in normalized, (
                f"{path.name} is missing Notebook 3 guardrail: {guardrail}"
            )
        assert normalized.index("## core synthesis table") < normalized.index(
            "# optional extension a"
        ), f"{path.name} places the required synthesis after optional material"
        assert normalized.index("# optional extension b") < normalized.index(
            "# optional extension c"
        ), f"{path.name} places the control bridge before the WFA construction"

    print(f"PASS {path.name}: {len(data['cells'])} cells, {code_count} executed")


def validate_pair(stem: str) -> None:
    exercise_path = HERE / f"{stem}_exercises.ipynb"
    solution_path = HERE / f"{stem}_solutions.ipynb"
    exercises = json.loads(exercise_path.read_text(encoding="utf-8"))["cells"]
    solutions = json.loads(solution_path.read_text(encoding="utf-8"))["cells"]
    assert len(exercises) == len(solutions), f"{stem} pair has different cell counts"

    student_code_differences = 0
    response_pairs = 0
    for index, (exercise, solution) in enumerate(zip(exercises, solutions)):
        assert exercise["cell_type"] == solution["cell_type"], (
            f"{stem} cell {index} changes type across variants"
        )
        exercise_tags = exercise.get("metadata", {}).get("tags", [])
        solution_tags = solution.get("metadata", {}).get("tags", [])
        exercise_source = "".join(exercise["source"])
        solution_source = "".join(solution["source"])

        if exercise_tags == ["student-answer"]:
            assert solution_tags == ["solution"], (
                f"{stem} cell {index} lacks its worked-solution counterpart"
            )
            response_pairs += 1
            continue

        if "student-code" in exercise_tags:
            assert solution_tags == exercise_tags, (
                f"{stem} cell {index} changes student-code tags"
            )
            student_code_differences += exercise_source != solution_source
            continue

        assert exercise_tags == solution_tags, (
            f"{stem} cell {index} changes tags unexpectedly"
        )
        normalized_exercise = exercise_source.replace(
            "**Exercises ·", "**VARIANT ·"
        )
        normalized_solution = solution_source.replace(
            "**Worked solutions ·", "**VARIANT ·"
        )
        assert normalized_exercise == normalized_solution, (
            f"{stem} cell {index} drifts across variants"
        )

    assert response_pairs > 0, f"{stem} contains no exercise/solution responses"
    expected_student_differences = 1 if stem.startswith("02_") else 0
    assert student_code_differences == expected_student_differences, (
        f"{stem} has unexpected student-code differences"
    )
    print(f"PASS {stem}: {response_pairs} response pairs aligned")


def main() -> None:
    actual_notebooks = sorted(path.name for path in HERE.glob("*.ipynb"))
    assert actual_notebooks == EXPECTED_NOTEBOOKS, (
        f"notebook inventory differs: {actual_notebooks}"
    )
    for figure_number in [1, 3, 4, 5, 6, 7]:
        asset = HERE / "assets" / f"shai_et_al_2024_figure_{figure_number}.png"
        assert asset.exists(), f"missing Notebook 2 source figure: {asset.name}"
    assert (HERE.parent / "papers" / "shai_et_al_2024.pdf").exists(), (
        "missing local Shai et al. paper"
    )
    with tempfile.TemporaryDirectory(prefix="iliad-mpl-") as mpl_dir:
        os.environ["MPLBACKEND"] = "Agg"
        os.environ["MPLCONFIGDIR"] = mpl_dir
        import matplotlib.pyplot as plt

        def draw_show(*_args: object, **_kwargs: object) -> None:
            for figure_number in plt.get_fignums():
                plt.figure(figure_number).canvas.draw()

        plt.show = draw_show  # type: ignore[assignment]
        for filename in EXPECTED_NOTEBOOKS:
            validate(HERE / filename, plt)
    for stem in PAIR_STEMS:
        validate_pair(stem)


if __name__ == "__main__":
    main()
