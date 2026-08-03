"""
Export the figures used by FINDINGS.md straight out of the executed notebooks.

The images are pulled from stored cell outputs rather than re-plotted, so a figure
in the report can never disagree with the notebook it came from. Re-run this after
re-executing any notebook:

    uv run python notebooks/export_figures.py
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

NOTEBOOKS = Path(__file__).parent
FIGURES = NOTEBOOKS / "figures"

# (notebook stem, cell index, output filename stem)
WANTED = [
    ("pilot_v2_eval_exploration", 11, "01-judge-agreement"),
    ("pilot_v2_eval_exploration", 16, "02-all-900-items"),
    ("pilot_v2_eval_exploration", 19, "03-refusal-code-collapse"),
    ("pilot_v2_eval_exploration", 26, "04-paired-language-disagreement"),
    ("pilot_v2_eval_exploration", 31, "05-perturbation-class"),
    ("pilot_v2_leakage_subsets", 14, "06-leakage-subsets"),
    ("pilot_v2_judge_prompt_language", 8, "07-judge-prompt-language"),
]


def export() -> None:
    FIGURES.mkdir(exist_ok=True)
    
    for notebook, index, stem in WANTED:
        cells = json.loads((NOTEBOOKS / f"{notebook}.ipynb").read_text())["cells"]
        images = [out["data"]["image/png"] for out in cells[index].get("outputs", [])
                  if "image/png" in out.get("data", {})]
        
        # a silent miss here would leave a stale figure in the report, so fail loudly
        if len(images) != 1:
            raise SystemExit(
                f"{notebook} cell {index} holds {len(images)} images, expected 1. "
                "Has the notebook been re-executed, or the cells renumbered?"
            )
        
        (FIGURES / f"{stem}.png").write_bytes(base64.b64decode(images[0]))
        print(f"{stem}.png  <- {notebook} cell {index}")


if __name__ == "__main__":
    export()
