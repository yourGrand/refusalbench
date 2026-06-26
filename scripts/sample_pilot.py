#!/usr/bin/env python3
"""Sample RefusalBench-NQ test rows per perturbation stratum."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


DATASET_ID = "aashiqmuhamed/RefusalBench-NQ"
DEFAULT_OUTPUT = Path("data/pilot/pilot_en.jsonl")
PERTURBATION_CLASSES = [
    "P-Ambiguity",
    "P-Contradiction",
    "P-MissingInfo",
    "P-FalsePremise",
    "P-GranularityMismatch",
    "P-EpistemicMismatch",
]
INTENSITIES = ["LOW", "MEDIUM", "HIGH"]
REQUIRED_FIELDS = [
    "perturbed_query",
    "perturbed_context",
    "expected_rag_behavior",
    "original_answers",
    "perturbation_class",
    "intensity",
]


def stratum_key(row: Dict[str, Any]) -> str:
    """Match the stratum key used by naturalquestions/filter_stratified.py."""
    perturbation_class = row.get("perturbation_class", "unknown")
    intensity = str(row.get("intensity", "unknown")).upper()
    return f"{perturbation_class}_{intensity}"


def expected_strata() -> List[str]:
    return [
        f"{perturbation_class}_{intensity}"
        for perturbation_class in PERTURBATION_CLASSES
        for intensity in INTENSITIES
    ]


def load_refusalbench_test_split(dataset_id: str) -> Iterable[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'datasets'. Install it to sample the Hugging Face "
            "RefusalBench-NQ test split."
        ) from exc

    return load_dataset(dataset_id, split="test")


def normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    missing = [field for field in REQUIRED_FIELDS if field not in row]
    if missing:
        raise ValueError(f"row is missing required fields: {missing}")

    intensity = str(row["intensity"]).upper()
    output = {field: row.get(field) for field in REQUIRED_FIELDS}
    output["intensity"] = intensity
    output["language"] = "en"

    if row.get("unique_id"):
        output["unique_id"] = row["unique_id"]
    elif row.get("source_qid"):
        output["unique_id"] = str(row["source_qid"])

    return output


def sample_per_stratum(
    rows: Iterable[Dict[str, Any]], seed: int, per_stratum: int
) -> List[Dict[str, Any]]:
    random.seed(seed)
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for raw_row in rows:
        row = dict(raw_row)
        grouped[stratum_key(row)].append(row)

    selected = []
    missing = []
    insufficient = []
    for key in expected_strata():
        candidates = grouped.get(key, [])
        if not candidates:
            missing.append(key)
            continue
        if len(candidates) < per_stratum:
            insufficient.append(f"{key} ({len(candidates)} available)")
            continue
        selected.extend(normalize_row(row) for row in random.sample(candidates, per_stratum))

    if missing:
        raise SystemExit(
            "Could not sample all 18 strata. Missing strata: " + ", ".join(missing)
        )
    if insufficient:
        raise SystemExit(
            f"Could not sample {per_stratum} rows for every stratum. "
            "Insufficient strata: " + ", ".join(insufficient)
        )

    return selected


def write_jsonl(rows: Iterable[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an English RefusalBench-NQ pilot dataset by stratum."
    )
    parser.add_argument("--dataset", default=DATASET_ID, help="Hugging Face dataset ID")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--per-stratum",
        type=int,
        default=10,
        help="Number of rows to sample from each of the 18 strata (default: 10)",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_refusalbench_test_split(args.dataset)
    selected = sample_per_stratum(rows, seed=args.seed, per_stratum=args.per_stratum)
    write_jsonl(selected, args.output)

    expected_rows = len(expected_strata()) * args.per_stratum
    print(f"Wrote {len(selected)} pilot rows to {args.output} ({args.per_stratum} per stratum, expected {expected_rows})")
    counts: Dict[str, int] = defaultdict(int)
    for row in selected:
        counts[stratum_key(row)] += 1
    print("Strata:")
    for key in expected_strata():
        print(f"  {key}: {counts[key]}")


if __name__ == "__main__":
    main()
