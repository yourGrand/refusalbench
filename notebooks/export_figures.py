"""
Regenerate rendered .ipynb copies and export the figures FINDINGS.md uses.

The marimo `.py` files in this directory are the source of truth, but they store no cell
outputs, so this script does two jobs in one pass:

1. Executes each notebook with `marimo export ipynb --include-outputs`, writing the tracked
   copy under `rendered/`. That keeps the marimo sources and GitHub-renderable exports separate.
2. Pulls each figure out of the freshly executed outputs and writes it into `figures/`.

Because the images come from the executed notebook rather than being re-plotted here, a figure in
the report can never disagree with the analysis it came from.

Figures are located by a `# figure: <name>` marker comment in the plotting cell, not by cell
index, so cells can be added, removed or reordered without touching this file.

Run from the repository root:

    uv run python notebooks/export_figures.py
"""

from __future__ import annotations

import base64
import json
import re
import subprocess
import sys
from pathlib import Path

NOTEBOOKS = Path(__file__).parent
RENDERED = NOTEBOOKS / "rendered"
FIGURES = NOTEBOOKS / "figures"

STEMS = [
    "pilot_v2_eval_exploration",
    "pilot_v2_leakage_subsets",
    "pilot_v2_judge_prompt_language",
]

# every marker that must be found across all notebooks, so a silently dropped
# figure fails the run instead of leaving a stale png in the report
EXPECTED = {
    "01-judge-agreement",
    "02-all-900-items",
    "03-refusal-code-collapse",
    "04-paired-language-disagreement",
    "05-perturbation-class",
    "06-leakage-subsets",
    "07-judge-prompt-language",
}

MARKER = re.compile(r"#\s*figure:\s*(\S+)")


def execute(stem: str) -> Path:
    """Run the marimo notebook and rewrite its rendered .ipynb copy, outputs included."""
    source = NOTEBOOKS / f"{stem}.py"
    RENDERED.mkdir(exist_ok=True)
    target = RENDERED / f"{stem}.ipynb"

    print(f"executing {source.name} ...")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "marimo",
            "export",
            "ipynb",
            "--include-outputs",
            str(source),
            "-o",
            str(target),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.exit(f"marimo export failed for {source.name}:\n{result.stderr}")

    fix_cross_links(target)
    return target


def fix_cross_links(notebook: Path) -> None:
    """Drop the rendered/ prefix from cross-notebook links inside exported ipynb copies."""
    text = notebook.read_text()
    fixed = text.replace("](rendered/pilot_v2_", "](pilot_v2_")
    if fixed != text:
        notebook.write_text(fixed)


def figures_in(notebook: Path) -> dict[str, bytes]:
    """Map every `# figure: name` marker in the notebook to that cell's image."""
    found: dict[str, bytes] = {}

    for cell in json.loads(notebook.read_text())["cells"]:
        names = MARKER.findall("".join(cell.get("source", [])))
        if not names:
            continue

        images = [
            out["data"]["image/png"]
            for out in cell.get("outputs", [])
            if "image/png" in out.get("data", {})
        ]

        for name in names:
            # a marked cell that plotted nothing means the export ran but the
            # figure is gone, which would otherwise leave a stale png behind
            if len(images) != 1:
                sys.exit(
                    f"{notebook.name}: cell marked '{name}' holds {len(images)} images, "
                    "expected exactly 1. Did the cell stop plotting, or did it fail to run?"
                )
            found[name] = base64.b64decode(images[0])

    return found


def export() -> None:
    FIGURES.mkdir(exist_ok=True)

    exported: set[str] = set()
    for stem in STEMS:
        for name, image in figures_in(execute(stem)).items():
            (FIGURES / f"{name}.png").write_bytes(image)
            print(f"  {name}.png  <- {stem}")
            exported.add(name)

    if missing := EXPECTED - exported:
        sys.exit(f"no figure marker found for: {', '.join(sorted(missing))}")

    if extra := exported - EXPECTED:
        print(
            f"note: exported figures not used by FINDINGS.md: {', '.join(sorted(extra))}"
        )


if __name__ == "__main__":
    export()
