"""aggregate multilingual judge outputs into consensus metrics and agreement stats."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

VALID_REFUSAL_CODES = [
    "REFUSE_AMBIGUOUS_QUERY",
    "REFUSE_CONTRADICTORY_CONTEXT",
    "REFUSE_INFO_MISSING_IN_CONTEXT",
    "REFUSE_FALSE_PREMISE_IN_QUERY",
    "REFUSE_GRANULARITY_MISMATCH",
    "REFUSE_NONFACTUAL_QUERY",
    "REFUSE_OTHER",
]

ANSWER_CORRECTLY = "ANSWER_CORRECTLY"

METRIC_COLUMNS = [
    "n",
    "num_answerable",
    "num_unanswerable",
    "answer_accuracy",
    "refusal_accuracy",
    "false_refusal_rate",
    "missed_refusal_rate",
    "calibrated_refusal_score",
]

MERGED_OUTPUT_COLUMNS = [
    "judge_id",
    "language",
    "id",
    "status",
    "query",
    "ground_truth_label",
    "ground_truth_answer",
    "model_raw_output",
    "model_predicted_type",
    "answer_quality_score",
    "refusal_match_correct",
    "llm_evaluation_explanation",
    "perturbation_class",
    "intensity",
]


def _repo_relative(path: Union[str, Path]) -> Path:
    """
    resolve a path relative to the repository root.

    Parameters
    ----------
    path :
        file or directory path, absolute or relative to repo root.

    Returns
    -------
    pathlib.Path
        resolved absolute path.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    return REPO_ROOT / p


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    load a jsonl file into a list of dict rows.

    Parameters
    ----------
    path :
        path to the jsonl file.

    Returns
    -------
    list of dict
        one dict per non-empty line.
    """
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _normalize_language(value: Any) -> Optional[str]:
    """
    lowercase language codes for consistent merge keys.

    Parameters
    ----------
    value :
        raw language field from judgment or inference rows.

    Returns
    -------
    str or None
        lowercased language code when present.
    """
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    return str(value).lower()


def _first_present(row: Dict[str, Any], keys: Sequence[str]) -> Any:
    """
    return the first non-missing value for any key in keys.

    Parameters
    ----------
    row :
        source mapping.
    keys :
        candidate field names in priority order.

    Returns
    -------
    any
        first present value or None.
    """
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _parse_quality_score(value: Any) -> Optional[float]:
    """
    parse a quality score field into a float or none.

    Parameters
    ----------
    value :
        raw score from judgment or inference output.

    Returns
    -------
    float or None
        numeric score when parseable.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if np.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.upper() in {"N/A", "NA", "NONE", "NULL"}:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def normalize_judgment_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    map flexible judgment fields to pilot-compatible column names.

    Parameters
    ----------
    row :
        raw judgment json object.

    Returns
    -------
    dict
        normalized row with canonical keys.
    """
    example_id = _first_present(row, ("id", "unique_id", "entry_idx", "sample_id"))
    classification = _first_present(row, ("classification", "model_predicted_type"))
    quality = _parse_quality_score(
        _first_present(row, ("quality_score", "answer_quality_score"))
    )
    explanation = _first_present(
        row, ("explanation", "llm_evaluation_explanation", "judge_explanation")
    )
    gt_label = _first_present(
        row, ("expected_behavior", "ground_truth_label", "expected_rag_behavior")
    )
    gt_answer = _first_present(
        row, ("reference_answers", "ground_truth_answer", "answer")
    )
    model_output = _first_present(row, ("model_raw_output", "response", "model_answer"))
    status = row.get("status")
    if status is None:
        status = "ok" if classification is not None else "error"

    normalized = {
        "judge_id": _first_present(row, ("judge", "judge_id", "model_id")),
        "language": _normalize_language(row.get("language")),
        "id": example_id,
        "status": str(status).lower() if status is not None else "error",
        "query": _first_present(row, ("query", "query_extracted")),
        "ground_truth_label": gt_label,
        "ground_truth_answer": gt_answer,
        "model_raw_output": model_output,
        "model_predicted_type": classification,
        "answer_quality_score": quality,
        "llm_evaluation_explanation": explanation,
        "perturbation_class": row.get("perturbation_class"),
        "intensity": row.get("intensity"),
        "refusal_match_correct": row.get("refusal_match_correct"),
    }
    return normalized


def apply_dual_eval(
    ground_truth_label: Optional[str],
    model_predicted_type: Optional[str],
    quality_score: Optional[float],
    explanation: Optional[str],
) -> Tuple[Optional[float], Optional[bool], Optional[str]]:
    """
    apply refusalbench dual evaluation rules from process_example.

    Parameters
    ----------
    ground_truth_label :
        expected behavior label.
    model_predicted_type :
        judge classification for the model response.
    quality_score :
        parsed answer quality score when present.
    explanation :
        existing judge explanation text.

    Returns
    -------
    tuple
        updated quality score, refusal_match_correct, explanation.
    """
    gt = ground_truth_label or ""
    pred = model_predicted_type or ""

    if gt == ANSWER_CORRECTLY:
        if pred == "answer_attempt":
            return quality_score, None, explanation
        false_msg = (
            f"False Refusal: Model refused with {pred} when should have answered"
        )
        return None, False, false_msg

    if pred == "answer_attempt":
        missed_msg = (
            f"Missed Refusal: Model answered when should have refused with {gt}"
        )
        return None, False, missed_msg

    is_match = pred == gt
    if is_match:
        match_msg = f"Exact match: {pred}"
    else:
        match_msg = f"Mismatch: predicted={pred}, ground_truth={gt}"
    return None, is_match, match_msg


def enrich_row_with_dual_eval(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    fill dual-eval fields when missing using ground truth and prediction.

    Parameters
    ----------
    row :
        normalized judgment row.

    Returns
    -------
    dict
        row with refusal_match_correct and quality fields updated.
    """
    out = dict(row)
    if out["status"] != "ok":
        return out

    quality, refusal_match, explanation = apply_dual_eval(
        out.get("ground_truth_label"),
        out.get("model_predicted_type"),
        out.get("answer_quality_score"),
        out.get("llm_evaluation_explanation"),
    )
    out["answer_quality_score"] = quality
    out["refusal_match_correct"] = refusal_match
    if explanation is not None:
        out["llm_evaluation_explanation"] = explanation
    return out


def _inference_lookup_key(row: Dict[str, Any]) -> Optional[Tuple[Any, Any]]:
    """
    build a merge key for inference metadata rows.

    Parameters
    ----------
    row :
        inference json object.

    Returns
    -------
    tuple or None
        (language, id) when both are present.
    """
    language = _normalize_language(row.get("language"))
    example_id = _first_present(row, ("id", "unique_id", "entry_idx", "sample_id"))
    if language is None or example_id is None:
        return None
    return language, example_id


def load_inference_index(input_dir: Path) -> Dict[Tuple[Any, Any], Dict[str, Any]]:
    """
    index inference jsonl rows by language and example id.

    Parameters
    ----------
    input_dir :
        directory containing inference result jsonl files.

    Returns
    -------
    dict
        maps (language, id) to the first matching inference row.
    """
    index: Dict[Tuple[Any, Any], Dict[str, Any]] = {}
    if not input_dir.is_dir():
        return index

    jsonl_paths = sorted(input_dir.rglob("*.jsonl"))
    for jsonl_path in jsonl_paths:
        for row in load_jsonl(jsonl_path):
            key = _inference_lookup_key(row)
            if key is None or key in index:
                continue
            index[key] = row
    return index


def merge_inference_metadata(
    rows: List[Dict[str, Any]], inference_index: Dict[Tuple[Any, Any], Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    fill missing ground truth fields from inference metadata when available.

    Parameters
    ----------
    rows :
        normalized judgment rows.
    inference_index :
        inference rows keyed by language and id.

    Returns
    -------
    list of dict
        rows with merged metadata.
    """
    merged: List[Dict[str, Any]] = []
    fill_if_missing = (
        (
            "ground_truth_label",
            ("expected_behavior", "ground_truth_label", "expected_rag_behavior"),
        ),
        ("ground_truth_answer", ("reference_answers", "ground_truth_answer", "answer")),
        ("query", ("query", "query_extracted", "perturbed_query")),
        ("perturbation_class", ("perturbation_class",)),
        ("intensity", ("intensity",)),
        ("model_raw_output", ("model_raw_output", "response", "model_answer")),
    )

    for row in rows:
        out = dict(row)
        key = (out.get("language"), out.get("id"))
        inf = (
            inference_index.get(key)
            if key[0] is not None and key[1] is not None
            else None
        )
        if inf:
            for target, sources in fill_if_missing:
                if out.get(target) is None:
                    out[target] = _first_present(inf, sources)
        merged.append(enrich_row_with_dual_eval(out))
    return merged


def _majority_vote(labels: Sequence[str]) -> Tuple[Optional[str], int, bool]:
    """
    compute majority classification among judge labels.

    Parameters
    ----------
    labels :
        non-empty sequence of classification strings.

    Returns
    -------
    tuple
        majority label, count agreeing with majority, no_majority flag.
    """
    if not labels:
        return None, 0, True

    counts = Counter(labels)
    top_count = max(counts.values())
    winners = [label for label, count in counts.items() if count == top_count]
    if len(winners) != 1:
        return None, top_count, True
    majority = winners[0]
    return majority, top_count, False


def build_consensus_df(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    build consensus rows grouped by language and example id.

    Parameters
    ----------
    merged_df :
        per-judge merged judgments.

    Returns
    -------
    pandas.DataFrame
        one consensus row per (language, id) using ok judges only.
    """
    ok_df = merged_df[merged_df["status"] == "ok"].copy()
    records: List[Dict[str, Any]] = []

    group_cols = ["language", "id"]
    for (language, example_id), group in ok_df.groupby(group_cols, dropna=False):
        labels = [
            str(x)
            for x in group["model_predicted_type"].tolist()
            if x is not None and not (isinstance(x, float) and np.isnan(x))
        ]
        majority, n_agree, no_majority = _majority_vote(labels)

        quality_vals: List[float] = []
        if majority is not None and majority == "answer_attempt":
            for _, judge_row in group.iterrows():
                if judge_row.get("model_predicted_type") != majority:
                    continue
                score = _parse_quality_score(judge_row.get("answer_quality_score"))
                if score is not None:
                    quality_vals.append(score)

        consensus_quality = float(np.median(quality_vals)) if quality_vals else None

        meta_row = group.iloc[0]
        gt = meta_row.get("ground_truth_label")
        pred = majority
        cons_quality, cons_refusal, cons_expl = apply_dual_eval(
            gt, pred, consensus_quality, None
        )

        records.append(
            {
                "language": language,
                "id": example_id,
                "ground_truth_label": gt,
                "perturbation_class": meta_row.get("perturbation_class"),
                "intensity": meta_row.get("intensity"),
                "query": meta_row.get("query"),
                "consensus_classification": majority,
                "n_agree": n_agree,
                "no_majority": no_majority,
                "n_judges_ok": len(group),
                "model_predicted_type": majority,
                "answer_quality_score": cons_quality,
                "refusal_match_correct": cons_refusal,
                "llm_evaluation_explanation": cons_expl,
                "status": "ok",
            }
        )

    return pd.DataFrame.from_records(records)


def compute_metrics_block(df: pd.DataFrame) -> Dict[str, float]:
    """
    compute refusalbench metrics for a slice of ok evaluation rows.

    Parameters
    ----------
    df :
        rows with ground_truth_label and model_predicted_type.

    Returns
    -------
    dict
        metric name to value. rate metrics are nan when their population is
        empty, and calibrated_refusal_score is nan unless the slice contains
        both answerable and unanswerable items. counts are always real.
    """
    work = df[df["status"] == "ok"].copy()
    n_total = len(work)
    answerable = work[work["ground_truth_label"] == ANSWER_CORRECTLY]
    unanswerable = work[work["ground_truth_label"] != ANSWER_CORRECTLY]

    num_answerable = len(answerable)
    num_unanswerable = len(unanswerable)

    if num_answerable > 0:
        quality = pd.to_numeric(answerable["answer_quality_score"], errors="coerce")
        correct_answers = (
            (answerable["model_predicted_type"] == "answer_attempt") & (quality >= 4)
        ).sum()
        answer_accuracy = correct_answers / num_answerable
        false_refusal_rate = (
            answerable["model_predicted_type"].isin(VALID_REFUSAL_CODES).sum()
            / num_answerable
        )
    else:
        answer_accuracy = float("nan")
        false_refusal_rate = float("nan")

    if num_unanswerable > 0:
        refusal_accuracy = (
            unanswerable["refusal_match_correct"] == True  # noqa: E712
        ).sum() / num_unanswerable
        missed_refusal_rate = (
            unanswerable["model_predicted_type"] == "answer_attempt"
        ).sum() / num_unanswerable
    else:
        refusal_accuracy = float("nan")
        missed_refusal_rate = float("nan")

    # only defined on a slice carrying both populations. a pure slice would
    # otherwise report half a real metric plus a filled-in zero.
    if num_answerable > 0 and num_unanswerable > 0:
        calibrated = 0.5 * answer_accuracy + 0.5 * refusal_accuracy
    else:
        calibrated = float("nan")

    return {
        "n": float(n_total),
        "num_answerable": float(num_answerable),
        "num_unanswerable": float(num_unanswerable),
        "answer_accuracy": float(answer_accuracy),
        "refusal_accuracy": float(refusal_accuracy),
        "false_refusal_rate": float(false_refusal_rate),
        "missed_refusal_rate": float(missed_refusal_rate),
        "calibrated_refusal_score": float(calibrated),
    }


def _consensus_rows_for_metrics(consensus_df: pd.DataFrame) -> pd.DataFrame:
    """
    prepare consensus rows for metric computation.

    Parameters
    ----------
    consensus_df :
        full consensus table including no_majority rows.

    Returns
    -------
    pandas.DataFrame
        consensus rows with no_majority rows removed.
    """
    work = consensus_df.copy()
    work["judge_id"] = "consensus"
    if "no_majority" not in work.columns:
        return work
    no_majority = work["no_majority"].fillna(False).astype(bool)
    return work.loc[~no_majority].copy()


def _attach_delta_vs_en(
    metrics_df: pd.DataFrame,
    group_cols: Sequence[str] = ("judge_id",),
) -> pd.DataFrame:
    """
    add delta_vs_en columns relative to the english row of the same group.

    the english baseline is looked up within each group, so every non-language
    key must be listed in group_cols. for stratum metrics that means the
    perturbation class and intensity as well as the judge, otherwise rows are
    differenced against an unrelated english cell.

    Parameters
    ----------
    metrics_df :
        metrics with a language column plus the columns named in group_cols.
    group_cols :
        columns identifying a comparable set of rows. the english row inside
        each group is the baseline for that group.

    Returns
    -------
    pandas.DataFrame
        copy with delta columns added, nan where the group has no english row.
    """
    out = metrics_df.copy()
    delta_metrics = [
        "answer_accuracy",
        "refusal_accuracy",
        "false_refusal_rate",
        "missed_refusal_rate",
        "calibrated_refusal_score",
    ]
    for col in delta_metrics:
        out[f"delta_vs_en_{col}"] = np.nan

    keys = [col for col in group_cols if col in out.columns]
    if out.empty or not keys:
        return out

    for _, group in out.groupby(keys, dropna=False):
        en_rows = group[group["language"].astype(str).str.lower() == "en"]
        if en_rows.empty:
            continue
        en_vals = en_rows.iloc[0]
        for col in delta_metrics:
            out.loc[group.index, f"delta_vs_en_{col}"] = group[col] - en_vals[col]
    return out


def compute_metrics_by_language(
    merged_df: pd.DataFrame, consensus_df: pd.DataFrame
) -> pd.DataFrame:
    """
    compute per-language metrics for each judge and consensus.

    Parameters
    ----------
    merged_df :
        per-judge merged judgments.
    consensus_df :
        consensus rows treated as judge_id consensus.

    Returns
    -------
    pandas.DataFrame
        metrics table with delta_vs_en columns.
    """
    records: List[Dict[str, Any]] = []

    for judge_id in sorted(merged_df["judge_id"].dropna().unique()):
        judge_df = merged_df[merged_df["judge_id"] == judge_id]
        for language in sorted(judge_df["language"].dropna().unique()):
            slice_df = judge_df[judge_df["language"] == language]
            block = compute_metrics_block(slice_df)
            records.append({"language": language, "judge_id": judge_id, **block})

    consensus_work = _consensus_rows_for_metrics(consensus_df)
    for language in sorted(consensus_work["language"].dropna().unique()):
        slice_df = consensus_work[consensus_work["language"] == language]
        block = compute_metrics_block(slice_df)
        records.append({"language": language, "judge_id": "consensus", **block})

    metrics_df = pd.DataFrame.from_records(records)
    return _attach_delta_vs_en(metrics_df)


def compute_metrics_by_stratum(
    merged_df: pd.DataFrame, consensus_df: pd.DataFrame
) -> pd.DataFrame:
    """
    compute metrics broken down by language, perturbation class, and intensity.

    Parameters
    ----------
    merged_df :
        per-judge merged judgments.
    consensus_df :
        consensus rows.

    Returns
    -------
    pandas.DataFrame
        stratum-level metrics with delta_vs_en columns.
    """
    records: List[Dict[str, Any]] = []
    stratum_cols = ["language", "perturbation_class", "intensity"]

    for judge_id in sorted(merged_df["judge_id"].dropna().unique()):
        judge_df = merged_df[merged_df["judge_id"] == judge_id]
        for keys, slice_df in judge_df.groupby(stratum_cols, dropna=False):
            language, perturbation_class, intensity = keys
            block = compute_metrics_block(slice_df)
            records.append(
                {
                    "language": language,
                    "perturbation_class": perturbation_class,
                    "intensity": intensity,
                    "judge_id": judge_id,
                    **block,
                }
            )

    consensus_work = _consensus_rows_for_metrics(consensus_df)
    for keys, slice_df in consensus_work.groupby(stratum_cols, dropna=False):
        language, perturbation_class, intensity = keys
        block = compute_metrics_block(slice_df)
        records.append(
            {
                "language": language,
                "perturbation_class": perturbation_class,
                "intensity": intensity,
                "judge_id": "consensus",
                **block,
            }
        )

    metrics_df = pd.DataFrame.from_records(records)
    return _attach_delta_vs_en(
        metrics_df, group_cols=("judge_id", "perturbation_class", "intensity")
    )


def _cohens_kappa(labels_a: Sequence[str], labels_b: Sequence[str]) -> float:
    """
    compute cohen's kappa for two aligned label lists.

    Parameters
    ----------
    labels_a :
        first rater labels.
    labels_b :
        second rater labels.

    Returns
    -------
    float
        kappa coefficient, or nan when undefined.
    """
    if len(labels_a) == 0 or len(labels_a) != len(labels_b):
        return float("nan")

    categories = sorted(set(labels_a) | set(labels_b))
    idx = {cat: i for i, cat in enumerate(categories)}
    n = len(labels_a)
    conf = np.zeros((len(categories), len(categories)), dtype=float)
    for a, b in zip(labels_a, labels_b):
        conf[idx[a], idx[b]] += 1.0

    po = np.trace(conf) / n
    pa = conf.sum(axis=1) / n
    pb = conf.sum(axis=0) / n
    pe = float(np.sum(pa * pb))
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return float((po - pe) / (1.0 - pe))


def _fleiss_kappa(count_matrix: np.ndarray) -> float:
    """
    compute fleiss kappa from an items by categories count matrix.

    Parameters
    ----------
    count_matrix :
        shape (n_items, n_categories) with counts per category per item.

    Returns
    -------
    float
        fleiss kappa, or nan when undefined.
    """
    if count_matrix.size == 0:
        return float("nan")

    n_items, _ = count_matrix.shape
    n_raters = count_matrix.sum(axis=1)
    if np.any(n_raters <= 1):
        valid = count_matrix[n_raters > 1]
        n_raters = n_raters[n_raters > 1]
    else:
        valid = count_matrix

    if valid.size == 0:
        return float("nan")

    p_i = (np.sum(valid * valid, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar = float(np.mean(p_i))

    category_totals = valid.sum(axis=0)
    total_ratings = category_totals.sum()
    if total_ratings <= 0:
        return float("nan")
    p_j = category_totals / total_ratings
    p_e = float(np.sum(p_j * p_j))

    if p_e >= 1.0:
        return 1.0 if p_bar >= 1.0 else 0.0
    return float((p_bar - p_e) / (1.0 - p_e))


def _pairwise_classification_agreement(
    pivot: pd.DataFrame, judge_a: str, judge_b: str
) -> Tuple[float, float]:
    """
    compute percent agreement and cohen's kappa for two judges.

    Parameters
    ----------
    pivot :
        index (language, id), columns judge_id, values classification.
    judge_a :
        first judge column name.
    judge_b :
        second judge column name.

    Returns
    -------
    tuple
        percent agreement and cohen kappa.
    """
    paired = pivot[[judge_a, judge_b]].dropna()
    if paired.empty:
        return float("nan"), float("nan")
    labels_a = paired[judge_a].astype(str).tolist()
    labels_b = paired[judge_b].astype(str).tolist()
    pct = float(np.mean(np.array(labels_a) == np.array(labels_b)))
    kappa = _cohens_kappa(labels_a, labels_b)
    return pct, kappa


def _fleiss_from_pivot(pivot: pd.DataFrame, judges: Sequence[str]) -> float:
    """
    compute fleiss kappa across judge columns in a wide pivot table.

    Parameters
    ----------
    pivot :
        classifications per judge.
    judges :
        judge column names to include.

    Returns
    -------
    float
        fleiss kappa across listed judges.
    """
    work = pivot[list(judges)].dropna()
    if work.empty:
        return float("nan")

    categories = sorted({str(v) for col in judges for v in work[col].tolist()})
    cat_idx = {c: i for i, c in enumerate(categories)}
    counts = np.zeros((len(work), len(categories)), dtype=float)
    for row_i, (_, row) in enumerate(work.iterrows()):
        for judge in judges:
            label = str(row[judge])
            counts[row_i, cat_idx[label]] += 1.0
    return _fleiss_kappa(counts)


def _spearman_corr(a: pd.Series, b: pd.Series) -> float:
    """
    compute spearman correlation using rank pearson (no scipy).

    Parameters
    ----------
    a :
        first numeric series.
    b :
        second numeric series aligned with a.

    Returns
    -------
    float
        spearman rho or nan when undefined.
    """
    paired = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(paired) < 2:
        return float("nan")
    ra = paired["a"].rank(method="average")
    rb = paired["b"].rank(method="average")
    rho = ra.corr(rb, method="pearson")
    return float(rho) if rho is not None and not np.isnan(rho) else float("nan")


def _pairwise_quality_stats(
    pivot_quality: pd.DataFrame, judge_a: str, judge_b: str
) -> Tuple[float, float]:
    """
    compute mean absolute difference and spearman correlation for quality scores.

    Parameters
    ----------
    pivot_quality :
        wide table of numeric quality scores.
    judge_a :
        first judge column.
    judge_b :
        second judge column.

    Returns
    -------
    tuple
        mean absolute difference and spearman rho.
    """
    paired = pivot_quality[[judge_a, judge_b]].dropna()
    if paired.empty:
        return float("nan"), float("nan")
    a = paired[judge_a].astype(float)
    b = paired[judge_b].astype(float)
    mad = float(np.mean(np.abs(a - b)))
    spearman = _spearman_corr(a, b)
    return mad, spearman


def compute_judge_agreement(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    compute pairwise and fleiss agreement on classifications and quality scores.

    Parameters
    ----------
    merged_df :
        per-judge merged judgments with ok status rows.

    Returns
    -------
    pandas.DataFrame
        agreement statistics overall and per language.
    """
    ok_df = merged_df[merged_df["status"] == "ok"].copy()
    records: List[Dict[str, Any]] = []

    def _emit_scope(scope_language: Optional[str], scope_df: pd.DataFrame) -> None:
        judges = sorted(scope_df["judge_id"].dropna().unique())
        if len(judges) < 2:
            return

        class_pivot = scope_df.pivot_table(
            index=["language", "id"],
            columns="judge_id",
            values="model_predicted_type",
            aggfunc="first",
        )
        quality_pivot = scope_df.pivot_table(
            index=["language", "id"],
            columns="judge_id",
            values="answer_quality_score",
            aggfunc="first",
        )
        class_pivot = class_pivot.reindex(columns=judges)
        quality_pivot = quality_pivot.reindex(columns=judges)

        for i, judge_a in enumerate(judges):
            for judge_b in judges[i + 1 :]:
                pct, kappa = _pairwise_classification_agreement(
                    class_pivot, judge_a, judge_b
                )
                records.append(
                    {
                        "scope": scope_language or "overall",
                        "language": scope_language,
                        "metric": "classification_pct_agreement",
                        "judge_a": judge_a,
                        "judge_b": judge_b,
                        "value": pct,
                    }
                )
                records.append(
                    {
                        "scope": scope_language or "overall",
                        "language": scope_language,
                        "metric": "classification_cohen_kappa",
                        "judge_a": judge_a,
                        "judge_b": judge_b,
                        "value": kappa,
                    }
                )
                mad, spearman = _pairwise_quality_stats(quality_pivot, judge_a, judge_b)
                records.append(
                    {
                        "scope": scope_language or "overall",
                        "language": scope_language,
                        "metric": "quality_mean_abs_diff",
                        "judge_a": judge_a,
                        "judge_b": judge_b,
                        "value": mad,
                    }
                )
                records.append(
                    {
                        "scope": scope_language or "overall",
                        "language": scope_language,
                        "metric": "quality_spearman",
                        "judge_a": judge_a,
                        "judge_b": judge_b,
                        "value": spearman,
                    }
                )

        fleiss = _fleiss_from_pivot(class_pivot, judges)
        records.append(
            {
                "scope": scope_language or "overall",
                "language": scope_language,
                "metric": "classification_fleiss_kappa",
                "judge_a": "all",
                "judge_b": "all",
                "value": fleiss,
            }
        )

    _emit_scope(None, ok_df)
    for language in sorted(ok_df["language"].dropna().unique()):
        lang_df = ok_df[ok_df["language"] == language]
        _emit_scope(str(language), lang_df)

    return pd.DataFrame.from_records(records)


def build_failures_df(raw_rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    collect non-ok judgment rows for debugging.

    Parameters
    ----------
    raw_rows :
        original judgment json objects.

    Returns
    -------
    pandas.DataFrame
        failure rows preserving raw fields where possible.
    """
    failures: List[Dict[str, Any]] = []
    for row in raw_rows:
        normalized = normalize_judgment_row(row)
        if normalized["status"] != "ok":
            failures.append({**row, **normalized})
    return pd.DataFrame.from_records(failures)


def aggregate(
    judgments_path: Union[str, Path],
    input_dir: Union[str, Path],
    out_dir: Union[str, Path],
    judges: Optional[Sequence[str]] = None,
) -> Dict[str, Path]:
    """
    run the full multilingual aggregation pipeline.

    Parameters
    ----------
    judgments_path :
        path to judgments_raw.jsonl.
    input_dir :
        inference results directory for optional metadata merge.
    out_dir :
        directory for csv outputs.
    judges :
        optional list of judge ids to include.

    Returns
    -------
    dict
        output artifact name to written path.
    """
    judgments_path = _repo_relative(judgments_path)
    input_dir = _repo_relative(input_dir)
    out_dir = _repo_relative(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = load_jsonl(judgments_path)
    normalized = [normalize_judgment_row(row) for row in raw_rows]
    inference_index = load_inference_index(input_dir)
    merged_rows = merge_inference_metadata(normalized, inference_index)
    merged_df = pd.DataFrame.from_records(merged_rows)

    if judges:
        judge_set = set(judges)
        merged_df = merged_df[merged_df["judge_id"].isin(judge_set)].copy()

    for col in MERGED_OUTPUT_COLUMNS:
        if col not in merged_df.columns:
            merged_df[col] = None
    merged_out = merged_df[MERGED_OUTPUT_COLUMNS].copy()

    consensus_df = build_consensus_df(merged_df)
    metrics_lang = compute_metrics_by_language(merged_df, consensus_df)
    metrics_stratum = compute_metrics_by_stratum(merged_df, consensus_df)
    agreement_df = compute_judge_agreement(merged_df)
    failures_df = build_failures_df(raw_rows)

    outputs = {
        "judgments_merged.csv": out_dir / "judgments_merged.csv",
        "consensus.csv": out_dir / "consensus.csv",
        "metrics_by_language.csv": out_dir / "metrics_by_language.csv",
        "metrics_by_stratum.csv": out_dir / "metrics_by_stratum.csv",
        "judge_agreement.csv": out_dir / "judge_agreement.csv",
        "failures.csv": out_dir / "failures.csv",
    }

    merged_out.to_csv(outputs["judgments_merged.csv"], index=False)
    consensus_df.to_csv(outputs["consensus.csv"], index=False)
    metrics_lang.to_csv(outputs["metrics_by_language.csv"], index=False)
    metrics_stratum.to_csv(outputs["metrics_by_stratum.csv"], index=False)
    agreement_df.to_csv(outputs["judge_agreement.csv"], index=False)
    if failures_df.empty:
        failures_df = pd.DataFrame(columns=MERGED_OUTPUT_COLUMNS)
    failures_df.to_csv(outputs["failures.csv"], index=False)

    return outputs


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """
    parse command line arguments for aggregation.

    Parameters
    ----------
    argv :
        optional argument list, defaults to sys.argv.

    Returns
    -------
    argparse.Namespace
        parsed flags.
    """
    parser = argparse.ArgumentParser(
        description="aggregate multilingual refusalbench judge outputs"
    )
    parser.add_argument(
        "--judgments",
        default="data/pilot_v2_eval/judgments_raw.jsonl",
        help="path to judgments_raw.jsonl relative to repo root",
    )
    parser.add_argument(
        "--input-dir",
        default="data/pilot_v2_infer/inference_results/0/",
        help="inference jsonl directory for metadata merge",
    )
    parser.add_argument(
        "--out-dir",
        default="data/pilot_v2_eval/",
        help="output directory for csv artifacts",
    )
    parser.add_argument(
        "--judges",
        nargs="*",
        default=None,
        help="optional judge ids to include",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    cli entry point for multilingual aggregation.

    Parameters
    ----------
    argv :
        optional argument list.

    Returns
    -------
    int
        process exit code.
    """
    args = parse_args(argv)
    judgments_path = _repo_relative(args.judgments)
    if not judgments_path.is_file():
        print(
            f"judgments file not found: {judgments_path}\n"
            "provide --judgments pointing to judgments_raw.jsonl "
            "or run evaluation first"
        )
        return 1

    outputs = aggregate(
        judgments_path=judgments_path,
        input_dir=args.input_dir,
        out_dir=args.out_dir,
        judges=args.judges,
    )
    for name, path in outputs.items():
        print(f"wrote {name} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
