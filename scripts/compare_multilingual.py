#!/usr/bin/env python3
"""Compare multilingual RefusalBench-NQ pilot outputs by model and language."""

import argparse
import json
from pathlib import Path
from typing import Iterable, List

try:
    import pandas as pd
except ImportError:
    pd = None


DEFAULT_RESULTS = Path("refusalbench_evaluation_results_all_stratified/refusalbench_evaluation_results.csv")
DEFAULT_OUTPUT_DIR = Path("data/pilot/analysis")
VALID_REFUSAL_CODES = {
    "REFUSE_AMBIGUOUS_QUERY",
    "REFUSE_CONTRADICTORY_CONTEXT",
    "REFUSE_INFO_MISSING_IN_CONTEXT",
    "REFUSE_FALSE_PREMISE_IN_QUERY",
    "REFUSE_GRANULARITY_MISMATCH",
    "REFUSE_NONFACTUAL_QUERY",
    "REFUSE_OTHER",
}
METRIC_COLUMNS = [
    "answer_accuracy",
    "refusal_accuracy",
    "false_refusal_rate",
    "missed_refusal_rate",
    "calibrated_refusal_score",
]


def discover_csvs(paths: Iterable[Path]) -> List[Path]:
    csvs: List[Path] = []
    for path in paths:
        if path.is_dir():
            csvs.extend(sorted(path.rglob("*.csv")))
        elif path.exists():
            csvs.append(path)
        else:
            raise FileNotFoundError(path)
    return csvs


def require_pandas() -> None:
    if pd is None:
        raise SystemExit("Missing dependency 'pandas'. Install requirements.txt before comparing results.")


def load_results(paths: Iterable[Path]):
    require_pandas()
    csvs = discover_csvs(paths)
    if not csvs:
        raise SystemExit("No CSV result files found.")

    frames = []
    for csv_path in csvs:
        frame = pd.read_csv(csv_path)
        frame["source_file"] = str(csv_path)
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    if "language" not in df.columns:
        df["language"] = "en"
    df["language"] = df["language"].fillna("en")
    return df


def as_bool(series):
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def compute_group_metrics(group):
    answerable = group[group["ground_truth_label"] == "ANSWER_CORRECTLY"].copy()
    unanswerable = group[group["ground_truth_label"] != "ANSWER_CORRECTLY"].copy()

    answer_accuracy = 0.0
    if len(answerable) > 0:
        answer_scores = pd.to_numeric(answerable["answer_quality_score"], errors="coerce")
        correct_answers = (
            (answerable["model_predicted_type"] == "answer_attempt") & (answer_scores >= 4)
        ).sum()
        answer_accuracy = correct_answers / len(answerable)

    false_refusal_rate = 0.0
    if len(answerable) > 0:
        false_refusals = answerable["model_predicted_type"].isin(VALID_REFUSAL_CODES).sum()
        false_refusal_rate = false_refusals / len(answerable)

    refusal_accuracy = 0.0
    missed_refusal_rate = 0.0
    if len(unanswerable) > 0:
        refusal_accuracy = as_bool(unanswerable["refusal_match_correct"]).mean()
        missed_refusal_rate = (unanswerable["model_predicted_type"] == "answer_attempt").mean()

    calibrated_refusal_score = 0.5 * answer_accuracy + 0.5 * refusal_accuracy

    return pd.Series(
        {
            "n": len(group),
            "num_answerable": len(answerable),
            "num_unanswerable": len(unanswerable),
            "answer_accuracy": answer_accuracy,
            "refusal_accuracy": refusal_accuracy,
            "false_refusal_rate": false_refusal_rate,
            "missed_refusal_rate": missed_refusal_rate,
            "calibrated_refusal_score": calibrated_refusal_score,
        }
    )


def add_english_deltas(summary, baseline_keys: List[str]):
    baseline = summary[summary["language"] == "en"][baseline_keys + METRIC_COLUMNS].copy()
    baseline = baseline.rename(columns={column: f"{column}_en" for column in METRIC_COLUMNS})

    merged = summary.merge(baseline, on=baseline_keys, how="left")
    for column in METRIC_COLUMNS:
        merged[f"delta_vs_en_{column}"] = merged[column] - merged[f"{column}_en"]
        merged = merged.drop(columns=[f"{column}_en"])
    return merged


def compute_summary(df):
    summary = (
        df.groupby(["model_id", "language"], dropna=False)
        .apply(compute_group_metrics)
        .reset_index()
    )
    return add_english_deltas(summary, baseline_keys=["model_id"])


def compute_per_stratum(df):
    group_columns = ["model_id", "language", "perturbation_class", "intensity"]
    available_columns = [column for column in group_columns if column in df.columns]
    if len(available_columns) < 2:
        return pd.DataFrame()

    per_stratum = (
        df.groupby(available_columns, dropna=False)
        .apply(compute_group_metrics)
        .reset_index()
    )
    baseline_keys = [column for column in available_columns if column != "language"]
    return add_english_deltas(per_stratum, baseline_keys=baseline_keys)


def write_outputs(summary, per_stratum, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "multilingual_summary.csv"
    stratum_path = output_dir / "multilingual_per_stratum.csv"
    json_path = output_dir / "multilingual_summary.json"

    summary.to_csv(summary_path, index=False)
    if not per_stratum.empty:
        per_stratum.to_csv(stratum_path, index=False)

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "summary_rows": json.loads(summary.to_json(orient="records")),
                "per_stratum_rows": json.loads(per_stratum.to_json(orient="records")),
            },
            handle,
            indent=2,
        )

    print(f"Wrote summary CSV to {summary_path}")
    if not per_stratum.empty:
        print(f"Wrote per-stratum CSV to {stratum_path}")
    print(f"Wrote JSON summary to {json_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute multilingual pilot metrics by model, language, and stratum."
    )
    parser.add_argument("results", nargs="*", type=Path, default=[DEFAULT_RESULTS])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_results(args.results)
    summary = compute_summary(df)
    per_stratum = compute_per_stratum(df)
    write_outputs(summary, per_stratum, args.output_dir)


if __name__ == "__main__":
    main()
