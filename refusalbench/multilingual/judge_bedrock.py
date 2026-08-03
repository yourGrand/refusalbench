"""bedrock llm judges for multilingual refusalbench inference outputs."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm.asyncio import tqdm as atqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from refusalbench.multilingual.bedrock_retry import (  # noqa: E402
    AdaptiveConcurrency,
    PermanentBedrockError,
    RetryBudgetExceeded,
    acompletion_with_retry,
    configure_bedrock_client_logging,
)
from refusalbench.multilingual.env import (  # noqa: E402
    ensure_aws_credentials_hint,
    get_aws_region,
    get_judge_models,
    load_project_env,
    normalize_bedrock_bearer_env,
)

logger = logging.getLogger(__name__)

DEFAULT_LOG_FILENAME = "judge_bedrock.log"


def configure_judging_logging(log_path: Path) -> None:
    """
    route application logs to a file so the terminal can show tqdm only.

    Parameters
    ----------
    log_path
        path to the log file (parent directories are created).
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    root.addHandler(handler)


DEFAULT_LANGUAGES = ("en", "pl", "ru", "zh_cmn", "zh_yue")
LANG_COLUMN_MAP = {
    "en": "english_text",
    "pl": "PL",
    "ru": "RU",
    "zh_cmn": "ZH_CMN",
    "zh_yue": "ZH_YUE",
}
LABEL_SEGMENT_IDS = ("sys_query_label", "sys_context_label")

# segments of the judge prompt, in the order they are concatenated. the shared
# prompts csv carries a translation of each one per language.
JUDGE_SEGMENT_IDS = (
    "judge_intro",
    "judge_step1",
    "judge_step2",
    "judge_input_labels",
    "judge_output_format",
)
FORMAT_SEGMENT_ID = "judge_output_format"

PROMPT_LANGUAGE_ENGLISH = "english"
PROMPT_LANGUAGE_TARGET = "target"
PROMPT_LANGUAGES = (PROMPT_LANGUAGE_ENGLISH, PROMPT_LANGUAGE_TARGET)

VALID_REFUSAL_CODES = (
    "REFUSE_AMBIGUOUS_QUERY",
    "REFUSE_CONTRADICTORY_CONTEXT",
    "REFUSE_INFO_MISSING_IN_CONTEXT",
    "REFUSE_FALSE_PREMISE_IN_QUERY",
    "REFUSE_GRANULARITY_MISMATCH",
    "REFUSE_NONFACTUAL_QUERY",
    "REFUSE_OTHER",
)

ANSWER_ATTEMPT = "answer_attempt"
VALID_CLASSIFICATIONS = frozenset(VALID_REFUSAL_CODES) | {ANSWER_ATTEMPT}

# how many times a single record may be re-armed after the wall clock retry
# budget is exhausted before it is recorded as a permanent failure.
MAX_RETRY_BUDGET_REARMS = 3

FORMAT_REMINDER = (
    "Respond in this exact format:\n"
    "CLASSIFICATION: [answer_attempt OR one of the REFUSE_* codes]\n"
    "QUALITY_SCORE: [1-5 if answer_attempt with references, otherwise N/A]\n"
    "EXPLANATION: [brief reasoning for both classification and score]"
)


@dataclass(frozen=True)
class WorkItem:
    """one judge call for a single inference record."""

    judge_name: str
    model_id: str
    language: str
    record_id: str
    record: dict[str, Any]


def resolve_repo_path(path: str | Path) -> Path:
    """
    resolve a path relative to the repository root.

    Parameters
    ----------
    path
        file or directory path.

    Returns
    -------
    Path
        absolute path under the repo root when relative.
    """
    p = Path(path)
    if p.is_absolute():
        return p

    return REPO_ROOT / p


def first_nonempty_line(cell: str | None) -> str:
    """
    take the first non-empty line from a csv cell.

    Parameters
    ----------
    cell
        raw csv cell text.

    Returns
    -------
    str
        first non-empty line or empty string.
    """
    if cell is None:
        return ""

    for line in str(cell).splitlines():
        stripped = line.strip()

        if stripped:
            return stripped

    return str(cell).strip()


def load_prompt_labels(prompts_csv: Path) -> dict[str, dict[str, str]]:
    """
    load query and context labels per language from the shared prompts csv.

    Parameters
    ----------
    prompts_csv
        path to refusalbench_translation_shared_prompts.csv.

    Returns
    -------
    dict[str, dict[str, str]]
        language code to query_label and context_label strings.
    """
    labels_by_lang: dict[str, dict[str, str]] = {lang: {} for lang in LANG_COLUMN_MAP}
    with prompts_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            segment_id = row.get("segment_id", "")

            if segment_id not in LABEL_SEGMENT_IDS:
                continue

            key = "query_label" if segment_id == "sys_query_label" else "context_label"

            for lang, column in LANG_COLUMN_MAP.items():
                labels_by_lang[lang][key] = first_nonempty_line(row.get(column))

    return labels_by_lang


def load_judge_prompt_segments(prompts_csv: Path) -> dict[str, dict[str, str]]:
    """
    load the judge prompt segments per language from the shared prompts csv.

    unlike the query and context labels, these are multi-line blocks, so the
    whole cell is kept rather than its first line.

    Parameters
    ----------
    prompts_csv
        path to refusalbench_translation_shared_prompts.csv.

    Returns
    -------
    dict[str, dict[str, str]]
        language code to segment id to segment text.

    Raises
    ------
    ValueError
        when a language is missing any judge segment, since a partly
        translated judge prompt would silently mix languages.
    """
    segments: dict[str, dict[str, str]] = {lang: {} for lang in LANG_COLUMN_MAP}
    with prompts_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            segment_id = row.get("segment_id", "")

            if segment_id not in JUDGE_SEGMENT_IDS:
                continue

            for lang, column in LANG_COLUMN_MAP.items():
                text = (row.get(column) or "").strip()

                if text:
                    segments[lang][segment_id] = text

    incomplete = {
        lang: sorted(set(JUDGE_SEGMENT_IDS) - set(found))
        for lang, found in segments.items()
        if set(JUDGE_SEGMENT_IDS) - set(found)
    }

    if incomplete:
        raise ValueError(
            f"judge prompt segments missing in {prompts_csv}: {incomplete}. "
            "a partly translated judge prompt would mix languages silently"
        )

    return segments


def build_judge_prompt_localised(
    segments: dict[str, str],
    query: str,
    model_output: str,
    formatted_correct: str,
) -> str:
    """
    build the judge prompt from per-language segments.

    the instructions are translated but the classification labels, the refusal
    codes, and the CLASSIFICATION, QUALITY_SCORE, and EXPLANATION output keys
    stay in english in every translation, so the response parser is unchanged.

    Parameters
    ----------
    segments
        segment id to text for one language.
    query
        user query text.
    model_output
        model response to judge.
    formatted_correct
        reference answers block.

    Returns
    -------
    str
        full judge prompt string in the segment language.
    """
    parts: list[str] = []

    for segment_id in JUDGE_SEGMENT_IDS:
        text = segments[segment_id]

        if segment_id == "judge_input_labels":
            text = (
                text.replace("{query}", query)
                .replace("{model_output}", model_output)
                .replace("{formatted_correct}", formatted_correct)
            )

        parts.append(text)

    return "\n\n".join(parts)


def strip_dash_lines(text: str) -> str:
    """
    remove separator lines made of dashes.

    Parameters
    ----------
    text
        text block to clean.

    Returns
    -------
    str
        text without dash-only lines.
    """
    kept: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()

        if stripped and all(ch == "-" for ch in stripped):
            continue

        kept.append(line)

    return "\n".join(kept).strip()


def extract_translated_query(
    messages: list[dict[str, Any]] | None,
    query_label: str,
    context_label: str,
) -> str:
    """
    extract the translated query from the last user message.

    Parameters
    ----------
    messages
        chat messages from an inference record.
    query_label
        localized query header label.
    context_label
        localized context header label.

    Returns
    -------
    str
        extracted query text or empty string.
    """
    if not messages:
        return ""

    content = messages[-1].get("content", "")

    if isinstance(content, list):
        parts = []

        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))

        content = "\n".join(parts)

    content = str(content)

    if not query_label or not context_label:
        return strip_dash_lines(content)

    q_idx = content.find(query_label)

    if q_idx < 0:
        return strip_dash_lines(content)

    after_query = content[q_idx + len(query_label) :]
    c_idx = after_query.find(context_label)

    if c_idx < 0:
        return strip_dash_lines(after_query)

    return strip_dash_lines(after_query[:c_idx])


def judgment_record_metadata(record: dict[str, Any]) -> dict[str, Any]:
    """
    copy inference fields needed for downstream aggregation.

    Parameters
    ----------
    record
        inference jsonl record.

    Returns
    -------
    dict[str, Any]
        metadata fields embedded in each judgment row.
    """
    model_output = record.get("response") or record.get("model_output") or ""
    return {
        "expected_behavior": record.get("expected_behavior")
        or record.get("expected_rag_behavior"),
        "perturbation_class": record.get("perturbation_class"),
        "intensity": record.get("intensity"),
        "reference_answers": record.get("reference_answers"),
        "response": model_output,
    }


def build_formatted_correct(record: dict[str, Any]) -> str:
    """
    build reference answer block for the judge prompt.

    Parameters
    ----------
    record
        inference jsonl record.

    Returns
    -------
    str
        formatted reference answers or the unanswerable placeholder.
    """
    expected = record.get("expected_behavior") or record.get("expected_rag_behavior")
    if expected != "ANSWER_CORRECTLY":
        return "[No reference answers provided - this may be an unanswerable instance]"
    refs = record.get("reference_answers")
    if not refs:
        return "[No reference answers provided - this may be an unanswerable instance]"
    if isinstance(refs, list):
        lines = [str(ans) for ans in refs if ans]

        if not lines:
            return (
                "[No reference answers provided - this may be an unanswerable instance]"
            )

        return "\n".join(lines)

    return str(refs)


def build_judge_prompt(query: str, model_output: str, formatted_correct: str) -> str:
    """
    build the universal judge prompt (byte-identical template to run_models_all).

    Parameters
    ----------
    query
        user query text.
    model_output
        model response to judge.
    formatted_correct
        reference answers block.

    Returns
    -------
    str
        full judge prompt string.
    """
    return f"""Analyze the following model response by performing BOTH classification and quality evaluation:

STEP 1: CLASSIFICATION
Determine if this model response is:
1. An attempt to answer the query → "answer_attempt"  
2. A refusal with a specific code:
   - REFUSE_AMBIGUOUS_QUERY
   - REFUSE_CONTRADICTORY_CONTEXT
   - REFUSE_INFO_MISSING_IN_CONTEXT
   - REFUSE_FALSE_PREMISE_IN_QUERY
   - REFUSE_GRANULARITY_MISMATCH
   - REFUSE_NONFACTUAL_QUERY
   - REFUSE_OTHER

Look for refusal codes even if they appear with additional text.

STEP 2: QUALITY EVALUATION
If the response is an answer attempt AND reference answers are provided, evaluate quality:
- Score 1: Not relevant to query and reference answers
- Score 2: According to reference answers but not relevant to query  
- Score 3: Relevant to query and reference answers but contains mistakes
- Score 4: Relevant and has exact same metrics as reference, but not concise
- Score 5: Relevant to query and fully correct according to reference answers

If the response is a refusal OR no reference answers provided, use "N/A".

QUERY:
{query}

MODEL RESPONSE:
{model_output}

REFERENCE ANSWERS:
{formatted_correct}

Respond in this exact format:
CLASSIFICATION: [answer_attempt OR one of the REFUSE_* codes]
QUALITY_SCORE: [1-5 if answer_attempt with references, otherwise N/A]
EXPLANATION: [brief reasoning for both classification and score]"""


class InvalidClassificationError(ValueError):
    """judge returned a label outside the allowed classification set."""


def validate_classification(classification: str) -> str:
    """
    check a judge classification against the allowed label set.

    an unrecognised label such as a misspelled refusal code would otherwise
    flow into aggregation, where it counts as neither an answer attempt nor a
    refusal and silently disappears from every rate numerator while still
    sitting in the denominator.

    Parameters
    ----------
    classification
        label parsed from the judge response.

    Returns
    -------
    str
        the classification unchanged when it is valid.

    Raises
    ------
    InvalidClassificationError
        when the label is not answer_attempt or a known refusal code.
    """
    if classification not in VALID_CLASSIFICATIONS:
        raise InvalidClassificationError(
            f"classification {classification!r} is not one of "
            f"{sorted(VALID_CLASSIFICATIONS)}"
        )

    return classification


def parse_judge_response_strict(response: str) -> tuple[str, float | None, str]:
    """
    parse judge output using required CLASSIFICATION lines only.

    Parameters
    ----------
    response
        raw judge model text.

    Returns
    -------
    tuple[str, float | None, str]
        classification, quality score, and explanation.

    Raises
    ------
    ValueError
        when structured fields are missing or the response is empty.
    """
    if not response or not response.strip():
        raise ValueError("empty judge response")

    lines = response.strip().split("\n")
    classification_line = [
        line for line in lines if line.startswith("CLASSIFICATION:")
    ][0]
    classification = classification_line.split("CLASSIFICATION:")[1].strip()
    score_line = [line for line in lines if line.startswith("QUALITY_SCORE:")][0]
    score_text = score_line.split("QUALITY_SCORE:")[1].strip()
    quality_score = None if score_text == "N/A" else float(score_text)
    explanation_line = [line for line in lines if line.startswith("EXPLANATION:")]
    explanation = (
        explanation_line[0].split("EXPLANATION:")[1].strip() if explanation_line else ""
    )

    return classification, quality_score, explanation


def parse_judge_response_fallback(response: str) -> tuple[str, float | None, str]:
    """
    parse judge output with refusal-code fallback like run_models_all.

    Parameters
    ----------
    response
        raw judge model text.

    Returns
    -------
    tuple[str, float | None, str]
        classification, quality score, and explanation.

    Raises
    ------
    ValueError
        when the response is empty or otherwise unusable.
    """
    if not response or not response.strip():
        raise ValueError("empty judge response")

    try:
        return parse_judge_response_strict(response)
    except (IndexError, ValueError):
        response_text = response.upper()
        for code in VALID_REFUSAL_CODES:
            if code in response_text:
                return (
                    code,
                    None,
                    "Detected refusal code in response (fallback parsing)",
                )
        return (
            "answer_attempt",
            None,
            "Could not parse response, defaulting to answer attempt",
        )


async def call_judge_model(
    model_id: str,
    prompt: str,
    adaptive: AdaptiveConcurrency,
    max_retry_minutes: float | None,
) -> str:
    """
    run one judge completion with retries.

    Parameters
    ----------
    model_id
        litellm bedrock model id.
    prompt
        judge prompt text.
    adaptive
        per-judge adaptive concurrency helper.
    max_retry_minutes
        optional retry wall clock budget in minutes.

    Returns
    -------
    str
        judge model response text.
    """
    messages = [{"role": "user", "content": prompt}]
    return await acompletion_with_retry(
        model=model_id,
        messages=messages,
        temperature=0,
        max_retry_minutes=max_retry_minutes,
        adaptive=adaptive,
        region=get_aws_region(),
    )


async def judge_one_record(
    item: WorkItem,
    labels: dict[str, str],
    adaptive: AdaptiveConcurrency,
    max_retry_minutes: float | None,
    segments: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    classify and score one inference record with optional format re-call.

    Parameters
    ----------
    item
        work item describing judge, language, and record.
    labels
        query and context labels for the language.
    adaptive
        per-judge adaptive concurrency helper.
    max_retry_minutes
        optional retry wall clock budget in minutes.
    segments
        judge prompt segments in the target language. none keeps the canonical
        english prompt, which is what the published numbers were produced with.

    Returns
    -------
    dict[str, Any]
        judgment record ready for jsonl output.
    """
    prompt_language = item.language if segments else "en"
    base = {
        "judge": item.judge_name,
        "language": item.language,
        "id": item.record_id,
        "model_id": item.model_id,
        "judge_prompt_language": prompt_language,
        **judgment_record_metadata(item.record),
    }
    query = extract_translated_query(
        item.record.get("messages"),
        labels.get("query_label", ""),
        labels.get("context_label", ""),
    )
    model_output = item.record.get("response") or item.record.get("model_output") or ""
    formatted_correct = build_formatted_correct(item.record)

    if segments:
        prompt = build_judge_prompt_localised(
            segments, query, model_output, formatted_correct
        )
        format_reminder = segments[FORMAT_SEGMENT_ID]
    else:
        prompt = build_judge_prompt(query, model_output, formatted_correct)
        format_reminder = FORMAT_REMINDER

    raw = await call_judge_model(item.model_id, prompt, adaptive, max_retry_minutes)

    try:
        classification, quality_score, explanation = parse_judge_response_strict(raw)
        validate_classification(classification)
    except (IndexError, ValueError):
        reminder_prompt = f"{prompt}\n\n{format_reminder}"
        raw = await call_judge_model(
            item.model_id, reminder_prompt, adaptive, max_retry_minutes
        )

        try:
            classification, quality_score, explanation = parse_judge_response_fallback(
                raw
            )
            validate_classification(classification)
        except ValueError as exc:
            raise PermanentBedrockError(
                f"unusable judge response after retry: {exc}"
            ) from exc

    return {
        **base,
        "status": "ok",
        "classification": classification,
        "quality_score": quality_score,
        "explanation": explanation,
        "query_extracted": query,
        "raw_judge_response": raw,
        "judged_at": datetime.now(timezone.utc).isoformat(),
    }


def load_completed_ok_keys(judgments_path: Path) -> set[tuple[str, str, str, str]]:
    """
    load resume keys for judgments that already succeeded.

    the judge prompt language is part of the key, so an english run and a
    target-language run over the same records do not skip each other if they
    share an output directory. rows written before that field existed are
    treated as english.

    Parameters
    ----------
    judgments_path
        path to judgments_raw.jsonl.

    Returns
    -------
    set[tuple[str, str, str, str]]
        keys of (judge, language, id, judge_prompt_language) with status ok.
    """
    if not judgments_path.is_file():
        return set()

    done: set[tuple[str, str, str, str]] = set()
    with judgments_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()

            if not line:
                continue

            row = json.loads(line)
            if row.get("status") != "ok":
                continue

            done.add(
                (
                    row["judge"],
                    str(row["language"]).lower(),
                    row["id"],
                    str(row.get("judge_prompt_language") or "en").lower(),
                )
            )

    return done


def load_inference_records(path: Path, limit: int | None) -> list[dict[str, Any]]:
    """
    read inference jsonl records with an optional limit.

    Parameters
    ----------
    path
        jsonl file path.
    limit
        maximum records to load or none for all.

    Returns
    -------
    list[dict[str, Any]]
        parsed records.
    """
    if not path.is_file():
        logger.warning("missing inference file %s", path)

        return []

    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()

            if not line:
                continue

            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break

    return records


async def upsert_failure_csv(
    failures_path: Path,
    row: dict[str, str],
    lock: asyncio.Lock,
) -> None:
    """
    append a permanent failure row to failures.csv.

    Parameters
    ----------
    failures_path
        csv output path.
    row
        failure fields to write.
    lock
        asyncio lock guarding the file.
    """
    async with lock:
        failures_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["judge", "language", "id", "error", "recorded_at"]
        write_header = not failures_path.is_file() or failures_path.stat().st_size == 0

        with failures_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)

            if write_header:
                writer.writeheader()

            writer.writerow({key: row.get(key, "") for key in fieldnames})


async def process_work_item(
    item: WorkItem,
    labels_by_lang: dict[str, dict[str, str]],
    adaptives: dict[str, AdaptiveConcurrency],
    max_retry_minutes: float | None,
    judgments_path: Path,
    failures_path: Path,
    jsonl_lock: asyncio.Lock,
    failure_lock: asyncio.Lock,
    segments_by_lang: dict[str, dict[str, str]] | None = None,
) -> None:
    """
    run one work item and append success or permanent error outputs.

    Parameters
    ----------
    item
        judge work item.
    labels_by_lang
        per-language label map.
    adaptives
        per-judge adaptive concurrency helpers.
    max_retry_minutes
        optional retry budget in minutes.
    judgments_path
        judgments_raw.jsonl path.
    failures_path
        failures.csv path.
    jsonl_lock
        lock for judgments append.
    failure_lock
        lock for failures.csv append.
    segments_by_lang
        per-language judge prompt segments, or none for the english prompt.
    """
    adaptive = adaptives[item.judge_name]
    labels = labels_by_lang.get(item.language, {})
    segments = segments_by_lang.get(item.language) if segments_by_lang else None
    try:
        for rearm in range(MAX_RETRY_BUDGET_REARMS + 1):
            try:
                result = await judge_one_record(
                    item, labels, adaptive, max_retry_minutes, segments
                )

                async with jsonl_lock:
                    judgments_path.parent.mkdir(parents=True, exist_ok=True)
                    with judgments_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(result, ensure_ascii=False) + "\n")

                logger.info(
                    "judged ok judge=%s language=%s id=%s classification=%s",
                    item.judge_name,
                    item.language,
                    item.record_id,
                    result.get("classification"),
                )
                return

            except RetryBudgetExceeded as exc:
                # re-arming is bounded, otherwise --max-retry-minutes has no
                # effect and a permanently broken item loops for ever.
                if rearm >= MAX_RETRY_BUDGET_REARMS:
                    raise PermanentBedrockError(
                        f"retry budget exhausted {MAX_RETRY_BUDGET_REARMS + 1} "
                        f"times: {exc}"
                    ) from exc
                logger.warning(
                    "retry budget hit for %s %s %s, re-arm %s of %s",
                    item.judge_name,
                    item.language,
                    item.record_id,
                    rearm + 1,
                    MAX_RETRY_BUDGET_REARMS,
                )
                await asyncio.sleep(2.0)
    except PermanentBedrockError as exc:
        logger.error(
            "judged failed judge=%s language=%s id=%s: %s",
            item.judge_name,
            item.language,
            item.record_id,
            exc,
        )
        error_row = {
            "judge": item.judge_name,
            "language": item.language,
            "id": item.record_id,
            "model_id": item.model_id,
            "judge_prompt_language": item.language if segments else "en",
            **judgment_record_metadata(item.record),
            "status": "error",
            "error": str(exc),
            "judged_at": datetime.now(timezone.utc).isoformat(),
        }

        async with jsonl_lock:
            judgments_path.parent.mkdir(parents=True, exist_ok=True)
            with judgments_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(error_row, ensure_ascii=False) + "\n")

        await upsert_failure_csv(
            failures_path,
            {
                "judge": item.judge_name,
                "language": item.language,
                "id": item.record_id,
                "error": str(exc),
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            },
            failure_lock,
        )


async def run_judging(args: argparse.Namespace) -> None:
    """
    main async entry for batch judging.

    Parameters
    ----------
    args
        parsed cli arguments. requires judge_models on args.
    """
    judge_models: dict[str, str] = args.judge_models
    region = get_aws_region()
    os.environ["AWS_REGION_NAME"] = region

    prompts_csv = resolve_repo_path(args.prompts_csv)
    input_dir = resolve_repo_path(args.input_dir)
    out_dir = resolve_repo_path(args.out_dir)
    judgments_path = out_dir / "judgments_raw.jsonl"
    failures_path = out_dir / "failures.csv"

    labels_by_lang = load_prompt_labels(prompts_csv)
    segments_by_lang = None

    if args.judge_prompt_language == PROMPT_LANGUAGE_TARGET:
        segments_by_lang = load_judge_prompt_segments(prompts_csv)
        logger.info("judging with target-language judge prompts from %s", prompts_csv)
    done_keys = load_completed_ok_keys(judgments_path)

    judge_names = args.judges or list(judge_models.keys())
    for name in judge_names:
        if name not in judge_models:
            raise SystemExit(
                f"unknown judge {name!r}, choose from {list(judge_models)}"
            )

    adaptives = {
        name: AdaptiveConcurrency(default_limit=args.concurrency)
        for name in judge_names
    }

    work: list[WorkItem] = []
    for lang in args.languages:
        lang_norm = lang.lower()
        infer_path = input_dir / f"{lang_norm}.jsonl"
        records = load_inference_records(infer_path, args.limit)

        for record in records:
            record_id = str(record.get("id") or record.get("example_id") or "")

            if not record_id:
                continue

            prompt_lang = lang_norm if segments_by_lang else "en"
            for judge_name in judge_names:
                key = (judge_name, lang_norm, record_id, prompt_lang)

                if key in done_keys:
                    continue

                work.append(
                    WorkItem(
                        judge_name=judge_name,
                        model_id=judge_models[judge_name],
                        language=lang_norm,
                        record_id=record_id,
                        record=record,
                    )
                )

    logger.info("queued %s judgment calls", len(work))
    jsonl_lock = asyncio.Lock()
    failure_lock = asyncio.Lock()

    async def worker(item: WorkItem) -> None:
        await process_work_item(
            item,
            labels_by_lang,
            adaptives,
            args.max_retry_minutes,
            judgments_path,
            failures_path,
            jsonl_lock,
            failure_lock,
            segments_by_lang,
        )

    if not work:
        print("nothing to judge (all records already completed)", flush=True)
        return

    await atqdm.gather(
        *(worker(item) for item in work),
        desc=f"judging ({len(work)} calls)",
    )
    logger.info("finished judging %s calls", len(work))


def build_arg_parser() -> argparse.ArgumentParser:
    """
    build the cli argument parser.

    Returns
    -------
    argparse.ArgumentParser
        configured parser.
    """
    parser = argparse.ArgumentParser(
        description="run bedrock judges on multilingual inference"
    )
    parser.add_argument(
        "--input-dir",
        default="data/pilot_v2_infer/inference_results/0",
        help="directory with per-language inference jsonl files",
    )
    parser.add_argument(
        "--out-dir",
        default="data/pilot_v2_eval",
        help="directory for judgments_raw.jsonl and failures.csv",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=list(DEFAULT_LANGUAGES),
        help="language codes to judge",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="max records per language file",
    )
    parser.add_argument(
        "--judges",
        nargs="+",
        default=None,
        help="subset of judge names (sonnet4_5, mistral-large-3, nemotron-3-super)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="default concurrent calls per judge",
    )
    parser.add_argument(
        "--max-retry-minutes",
        type=float,
        default=None,
        help="optional wall clock retry budget per completion",
    )
    parser.add_argument(
        "--prompts-csv",
        default="data/pilot_v2/refusalbench_translation_shared_prompts.csv",
        help="shared translation prompts csv for label extraction",
    )
    parser.add_argument(
        "--judge-prompt-language",
        choices=PROMPT_LANGUAGES,
        default=PROMPT_LANGUAGE_ENGLISH,
        help=(
            "language of the judge instructions. english (default) keeps the "
            "canonical prompt the published numbers use. target reads the "
            "translated judge segments from --prompts-csv and judges each "
            "language in its own language. write target runs to a separate "
            "--out-dir so the two are not mixed"
        ),
    )
    parser.add_argument(
        "--env-file",
        default=None,
        help="path to .env file (default: .env at repository root)",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help=(
            "log file path (default: <out-dir>/judge_bedrock.log). "
            "info and warnings go here; the terminal shows tqdm progress only"
        ),
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    """
    cli entry point.

    Parameters
    ----------
    argv
        optional argument list override.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    out_dir = resolve_repo_path(args.out_dir)
    log_path = (
        resolve_repo_path(args.log_file)
        if args.log_file
        else out_dir / DEFAULT_LOG_FILENAME
    )

    configure_judging_logging(log_path)
    configure_bedrock_client_logging()
    print(f"logging to {log_path}", flush=True)

    env_path = load_project_env(env_file=args.env_file)
    normalize_bedrock_bearer_env()

    if env_path:
        logger.info("loaded environment from %s", env_path)
    else:
        default_env = REPO_ROOT / ".env"
        logger.warning(
            "no env file at %s, using shell environment only (see .env.example)",
            default_env,
        )

    ensure_aws_credentials_hint()
    args.judge_models = get_judge_models()

    asyncio.run(run_judging(args))


if __name__ == "__main__":
    main(sys.argv[1:])
