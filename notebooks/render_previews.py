"""Execute a notebook and save each matplotlib show() call as a PNG preview."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    os.environ["MPLBACKEND"] = "Agg"
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/iliad-matplotlib")
    import matplotlib.pyplot as plt

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"output directory must be empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counter = 0

    def save_show(*_args: object, **_kwargs: object) -> None:
        nonlocal counter
        for figure_number in plt.get_fignums():
            counter += 1
            figure = plt.figure(figure_number)
            target = args.output_dir / f"{counter:02d}.png"
            figure.savefig(target, dpi=150, bbox_inches="tight")
        plt.close("all")

    plt.show = save_show  # type: ignore[assignment]

    data = json.loads(args.notebook.read_text(encoding="utf-8"))
    namespace: dict[str, object] = {"__name__": "__notebook__"}
    for index, cell in enumerate(data["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        exec(compile(source, f"{args.notebook.name}:cell-{index}", "exec"),
             namespace)
    print(f"saved {counter} previews to {args.output_dir}")


if __name__ == "__main__":
    main()
