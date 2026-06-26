#!/usr/bin/env python3
"""Translate the English RefusalBench-NQ pilot with Amazon Translate."""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from tqdm import tqdm


DEFAULT_INPUT = Path("data/pilot/pilot_en.jsonl")
DEFAULT_OUTPUT = Path("data/pilot/pilot_multilingual.jsonl")
DEFAULT_TARGETS = ["ru", "zh", "pl", "de"]
SOURCE_LANGUAGE = "en"
TRANSLATION_PROVIDER = "amazon-translate"
MAX_TRANSLATE_BYTES = 10_000
REFUSAL_TOKEN_RE = re.compile(r"REFUSE_[A-Z0-9_]+")


def load_aws_config() -> str:
    """Load AWS credentials and region from config.py, matching run_models_all.py."""
    config_dir = Path(__file__).resolve().parents[1] / "refusalbench" / "naturalquestions"
    sys.path.insert(0, str(config_dir))
    region = os.environ.get("AWS_REGION_NAME") or os.environ.get("AWS_DEFAULT_REGION", "")
    try:
        from config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION_NAME

        os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
        os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
        os.environ["AWS_REGION_NAME"] = AWS_REGION_NAME
        return AWS_REGION_NAME
    except Exception:
        return region


def load_config_region() -> str:
    return load_aws_config()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def protect_refusal_tokens(text: str) -> Tuple[str, Dict[str, str]]:
    token_map: Dict[str, str] = {}

    def replace(match: re.Match) -> str:
        token = match.group(0)
        placeholder = f"__RB_REFUSAL_TOKEN_{len(token_map)}__"
        token_map[placeholder] = token
        return placeholder

    return REFUSAL_TOKEN_RE.sub(replace, text), token_map


def restore_refusal_tokens(text: str, token_map: Dict[str, str]) -> Tuple[str, List[str]]:
    warnings = []
    restored = text
    for placeholder, token in token_map.items():
        if placeholder not in restored:
            warnings.append(f"placeholder_missing:{token}")
            continue
        restored = restored.replace(placeholder, token)
    return restored, warnings


def translate_field(
    client: Any,
    text: str,
    target_language: str,
    field_name: str,
    row_index: int,
    max_bytes: int,
    progress: Any = None,
) -> Tuple[str, Dict[str, Any]]:
    if text is None:
        text = ""
    if not isinstance(text, str):
        text = str(text)

    byte_length = len(text.encode("utf-8"))
    if byte_length > max_bytes:
        raise ValueError(
            f"Row {row_index} field '{field_name}' is {byte_length} bytes, "
            f"which exceeds the Amazon Translate {max_bytes}-byte segment limit."
        )

    if not text:
        if progress is not None:
            progress.update(1)
        return "", {
            "length_ratio": 1.0,
            "warnings": [],
            "preserved_refusal_tokens": [],
        }

    protected_text, token_map = protect_refusal_tokens(text)
    response = client.translate_text(
        Text=protected_text,
        SourceLanguageCode=SOURCE_LANGUAGE,
        TargetLanguageCode=target_language,
    )
    translated = response["TranslatedText"]
    translated, warnings = restore_refusal_tokens(translated, token_map)

    ratio = len(translated) / len(text) if text else 1.0
    if ratio < 0.5 or ratio > 2.0:
        warnings.append(f"length_ratio_anomaly:{ratio:.3f}")

    if progress is not None:
        progress.update(1)
        progress.set_postfix(lang=target_language, field=field_name, refresh=False)

    return translated, {
        "length_ratio": ratio,
        "warnings": warnings,
        "preserved_refusal_tokens": sorted(set(token_map.values())),
    }


def make_english_row(row: Dict[str, Any]) -> Dict[str, Any]:
    output = dict(row)
    output["language"] = SOURCE_LANGUAGE
    output["translation_provider"] = None
    output["original_query"] = row.get("perturbed_query", "")
    output["original_context"] = row.get("perturbed_context", "")
    output["translation_metadata"] = {
        "source_language": SOURCE_LANGUAGE,
        "target_language": SOURCE_LANGUAGE,
        "is_source_row": True,
    }
    return output


def make_translated_row(
    client: Any,
    row: Dict[str, Any],
    target_language: str,
    row_index: int,
    max_bytes: int,
    progress: Any = None,
) -> Dict[str, Any]:
    translated_query, query_metadata = translate_field(
        client,
        row.get("perturbed_query", ""),
        target_language,
        "perturbed_query",
        row_index,
        max_bytes,
        progress=progress,
    )
    translated_context, context_metadata = translate_field(
        client,
        row.get("perturbed_context", ""),
        target_language,
        "perturbed_context",
        row_index,
        max_bytes,
        progress=progress,
    )

    warnings = query_metadata["warnings"] + context_metadata["warnings"]
    anomalies = [warning for warning in warnings if warning.startswith("length_ratio_anomaly")]

    output = dict(row)
    output["language"] = target_language
    output["translation_provider"] = TRANSLATION_PROVIDER
    output["original_query"] = row.get("perturbed_query", "")
    output["original_context"] = row.get("perturbed_context", "")
    output["perturbed_query"] = translated_query
    output["perturbed_context"] = translated_context
    output["translation_metadata"] = {
        "source_language": SOURCE_LANGUAGE,
        "target_language": target_language,
        "provider": TRANSLATION_PROVIDER,
        "length_ratios": {
            "perturbed_query": query_metadata["length_ratio"],
            "perturbed_context": context_metadata["length_ratio"],
        },
        "length_ratio_anomalies": anomalies,
        "warnings": warnings,
        "preserved_refusal_tokens": sorted(
            set(query_metadata["preserved_refusal_tokens"])
            | set(context_metadata["preserved_refusal_tokens"])
        ),
    }
    output["length_ratio_anomaly"] = bool(anomalies)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate pilot_en.jsonl to a multilingual Amazon Translate pilot file."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--region", default=load_config_region())
    parser.add_argument("--max-bytes", type=int, default=MAX_TRANSLATE_BYTES)
    return parser.parse_args()


def main() -> None:
    load_aws_config()
    args = parse_args()
    rows = read_jsonl(args.input)

    try:
        import boto3
    except ImportError as exc:
        raise SystemExit("Missing dependency 'boto3'. Install requirements.txt first.") from exc

    client_kwargs = {"region_name": args.region} if args.region else {}
    client = boto3.client("translate", **client_kwargs)

    output_rows: List[Dict[str, Any]] = []
    translation_jobs = len(rows) * len(args.targets) * 2
    with tqdm(total=translation_jobs, desc="Amazon Translate", unit="field") as progress:
        for row_index, row in enumerate(rows):
            output_rows.append(make_english_row(row))
            for target_language in args.targets:
                output_rows.append(
                    make_translated_row(
                        client=client,
                        row=row,
                        target_language=target_language,
                        row_index=row_index,
                        max_bytes=args.max_bytes,
                        progress=progress,
                    )
                )

    write_jsonl(output_rows, args.output)
    anomaly_count = sum(1 for row in output_rows if row.get("length_ratio_anomaly"))
    print(f"Wrote {len(output_rows)} multilingual rows to {args.output}")
    print(f"Length-ratio anomaly rows: {anomaly_count}")


if __name__ == "__main__":
    main()
