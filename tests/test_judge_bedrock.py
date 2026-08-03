"""tests for judge prompt construction, response parsing, and label validation."""

from __future__ import annotations

import asyncio
import csv
import json
from pathlib import Path
from typing import Any

import pytest

from refusalbench.multilingual import judge_bedrock
from refusalbench.multilingual.bedrock_retry import (
    AdaptiveConcurrency,
    RetryBudgetExceeded,
)
from refusalbench.multilingual.judge_bedrock import (
    JUDGE_SEGMENT_IDS,
    MAX_RETRY_BUDGET_REARMS,
    VALID_CLASSIFICATIONS,
    InvalidClassificationError,
    WorkItem,
    build_formatted_correct,
    build_judge_prompt,
    build_judge_prompt_localised,
    extract_translated_query,
    first_nonempty_line,
    load_completed_ok_keys,
    load_judge_prompt_segments,
    load_prompt_labels,
    parse_judge_response_fallback,
    parse_judge_response_strict,
    process_work_item,
    strip_dash_lines,
    validate_classification,
)

PROMPTS_CSV = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "pilot_v2"
    / "refusalbench_translation_shared_prompts.csv"
)
HAVE_PROMPTS = PROMPTS_CSV.is_file()
needs_prompts = pytest.mark.skipif(
    not HAVE_PROMPTS, reason="shared prompts csv is not in git, see the package README"
)

# the labels the pilot actually shipped, taken from
# data/pilot_v2/refusalbench_translation_shared_prompts.csv
LABELS = {
    "en": {"query_label": "QUERY:", "context_label": "CONTEXT PASSAGES:"},
    "pl": {"query_label": "PYTANIE:", "context_label": "FRAGMENTY KONTEKSTOWE:"},
    "ru": {"query_label": "ЗАПРОС:", "context_label": "КОНТЕКСТНЫЕ ОТРЫВКИ:"},
    "zh_cmn": {"query_label": "查询：", "context_label": "上下文段落："},
    "zh_yue": {"query_label": "查詢：", "context_label": "上下文段落："},
}

QUERIES = {
    "en": "who plays meredith quill in guardians of the galaxy 2",
    "pl": "kto gra meredith quill w guardians of the galaxy 2",
    "ru": "кто играет мередит квилл в стражах галактики 2",
    "zh_cmn": "谁在银河护卫队2中扮演梅雷迪思奎尔",
    "zh_yue": "邊個喺銀河守護隊2度演梅雷迪思奎爾",
}

PREAMBLES = {
    "en": "Read the following Query and Context Passages carefully.",
    "pl": "Uważnie przeczytaj poniższe Pytanie i Fragmenty Kontekstowe.",
    "ru": "Внимательно прочитайте следующий Запрос и Контекстные Отрывки.",
    "zh_cmn": "请仔细阅读以下查询和上下文段落。",
    "zh_yue": "請細心閱讀以下查詢同上下文段落。",
}


def _user_message(language: str) -> list[dict[str, str]]:
    """build a user message shaped like the pilot inference prompts."""
    labels = LABELS[language]
    content = (
        f"{PREAMBLES[language]}\n\n\n"
        f"{labels['query_label']}\n{QUERIES[language]}\n\n"
        f"{labels['context_label']}\nTitle: Laura Haddock\nsome context text\n"
    )
    return [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": content},
    ]


class TestFirstNonemptyLine:
    def test_plain_value(self) -> None:
        assert first_nonempty_line("QUERY:") == "QUERY:"

    def test_skips_leading_blank_lines(self) -> None:
        assert first_nonempty_line("\n\n  QUERY:  \n") == "QUERY:"

    def test_drops_trailing_separator(self) -> None:
        """
        the mandarin label cell ships as '查询：\\n-----', so only the first line
        is the label.
        """
        assert first_nonempty_line("查询：\n-----") == "查询："

    def test_none_becomes_empty(self) -> None:
        assert first_nonempty_line(None) == ""


class TestStripDashLines:
    def test_removes_separator_rows_only(self) -> None:
        text = "line one\n-----\nline two"
        assert strip_dash_lines(text) == "line one\nline two"

    def test_keeps_hyphenated_words(self) -> None:
        assert strip_dash_lines("well-known fact") == "well-known fact"


class TestExtractTranslatedQuery:
    @pytest.mark.parametrize("language", sorted(LABELS))
    def test_extracts_the_query_in_every_pilot_language(self, language: str) -> None:
        """
        a failure here would hand the judge the whole prompt including context,
        which would confound exactly the cross-language comparison the pilot is
        built to make.
        """
        extracted = extract_translated_query(
            _user_message(language),
            LABELS[language]["query_label"],
            LABELS[language]["context_label"],
        )
        assert extracted == QUERIES[language]

    @pytest.mark.parametrize("language", sorted(LABELS))
    def test_context_never_leaks_into_the_query(self, language: str) -> None:
        extracted = extract_translated_query(
            _user_message(language),
            LABELS[language]["query_label"],
            LABELS[language]["context_label"],
        )
        assert "Laura Haddock" not in extracted
        assert "some context text" not in extracted

    def test_handles_content_blocks(self) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "QUERY:\nwho sang it\n"},
                    {"type": "text", "text": "CONTEXT PASSAGES:\nsome text"},
                ],
            }
        ]
        extracted = extract_translated_query(messages, "QUERY:", "CONTEXT PASSAGES:")
        assert extracted == "who sang it"

    def test_missing_labels_fall_back_to_whole_content(self) -> None:
        extracted = extract_translated_query(_user_message("en"), "", "")
        assert QUERIES["en"] in extracted

    def test_label_absent_from_content_falls_back(self) -> None:
        messages = [{"role": "user", "content": "no labels here at all"}]
        extracted = extract_translated_query(messages, "QUERY:", "CONTEXT PASSAGES:")
        assert extracted == "no labels here at all"

    def test_empty_messages_give_empty_string(self) -> None:
        assert extract_translated_query(None, "QUERY:", "CONTEXT PASSAGES:") == ""
        assert extract_translated_query([], "QUERY:", "CONTEXT PASSAGES:") == ""


class TestLoadPromptLabels:
    def test_reads_labels_for_every_language(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "prompts.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "segment_id",
                    "component",
                    "english_text",
                    "PL",
                    "RU",
                    "ZH_YUE",
                    "ZH_CMN",
                    "notes",
                ]
            )
            writer.writerow(
                [
                    "sys_query_label",
                    "system",
                    "QUERY:",
                    "PYTANIE:",
                    "ЗАПРОС:",
                    "查詢：",
                    "查询：\n-----",
                    "",
                ]
            )
            writer.writerow(
                [
                    "sys_context_label",
                    "system",
                    "CONTEXT PASSAGES:",
                    "FRAGMENTY KONTEKSTOWE:",
                    "КОНТЕКСТНЫЕ ОТРЫВКИ:",
                    "上下文段落：",
                    "上下文段落：",
                    "",
                ]
            )
            writer.writerow(["sys_role", "system", "ignored", "", "", "", "", ""])

        labels = load_prompt_labels(csv_path)
        assert labels["en"] == LABELS["en"]
        assert labels["pl"] == LABELS["pl"]
        assert labels["zh_cmn"]["query_label"] == "查询："


class TestValidateClassification:
    @pytest.mark.parametrize("label", sorted(VALID_CLASSIFICATIONS))
    def test_accepts_every_allowed_label(self, label: str) -> None:
        assert validate_classification(label) == label

    def test_rejects_misspelled_refusal_code(self) -> None:
        """
        this exact typo appeared in the pilot judgments. unvalidated it counts
        as neither a refusal nor an answer attempt downstream.
        """
        with pytest.raises(InvalidClassificationError):
            validate_classification("REFUSE_NONFACTICAL_QUERY")

    def test_rejects_case_variants(self) -> None:
        with pytest.raises(InvalidClassificationError):
            validate_classification("Answer_Attempt")

    def test_rejects_prose(self) -> None:
        with pytest.raises(InvalidClassificationError):
            validate_classification("the model attempted an answer")

    def test_error_is_a_value_error(self) -> None:
        """
        judge_one_record catches ValueError to trigger the format reminder, so
        the subclass relationship is load bearing.
        """
        assert issubclass(InvalidClassificationError, ValueError)


class TestParseJudgeResponseStrict:
    def test_answer_attempt_with_score(self) -> None:
        raw = (
            "CLASSIFICATION: answer_attempt\n"
            "QUALITY_SCORE: 5\n"
            "EXPLANATION: matches the reference exactly"
        )
        classification, score, explanation = parse_judge_response_strict(raw)
        assert classification == "answer_attempt"
        assert score == pytest.approx(5.0)
        assert explanation == "matches the reference exactly"

    def test_refusal_with_na_score(self) -> None:
        raw = (
            "CLASSIFICATION: REFUSE_AMBIGUOUS_QUERY\n"
            "QUALITY_SCORE: N/A\n"
            "EXPLANATION: the query is ambiguous"
        )
        classification, score, _ = parse_judge_response_strict(raw)
        assert classification == "REFUSE_AMBIGUOUS_QUERY"
        assert score is None

    def test_missing_explanation_is_tolerated(self) -> None:
        raw = "CLASSIFICATION: answer_attempt\nQUALITY_SCORE: 4"
        classification, score, explanation = parse_judge_response_strict(raw)
        assert classification == "answer_attempt"
        assert explanation == ""

    def test_empty_response_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_judge_response_strict("   ")

    def test_missing_classification_line_raises(self) -> None:
        with pytest.raises(IndexError):
            parse_judge_response_strict("QUALITY_SCORE: 4\nEXPLANATION: nope")

    def test_unparseable_score_raises(self) -> None:
        raw = "CLASSIFICATION: answer_attempt\nQUALITY_SCORE: four\nEXPLANATION: x"
        with pytest.raises(ValueError):
            parse_judge_response_strict(raw)


class TestParseJudgeResponseFallback:
    def test_delegates_to_strict_when_well_formed(self) -> None:
        raw = "CLASSIFICATION: answer_attempt\nQUALITY_SCORE: 4\nEXPLANATION: fine"
        assert parse_judge_response_fallback(raw)[0] == "answer_attempt"

    def test_recovers_a_refusal_code_from_prose(self) -> None:
        raw = "I think this is REFUSE_INFO_MISSING_IN_CONTEXT because the passage is silent"
        classification, score, _ = parse_judge_response_fallback(raw)
        assert classification == "REFUSE_INFO_MISSING_IN_CONTEXT"
        assert score is None

    def test_unrecognisable_output_defaults_to_answer_attempt(self) -> None:
        """
        documents the default. it fired zero times on the pilot, and the
        explanation string is what makes such rows findable afterwards.
        """
        classification, _, explanation = parse_judge_response_fallback("who knows")
        assert classification == "answer_attempt"
        assert "Could not parse" in explanation

    def test_empty_response_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_judge_response_fallback("")


class TestBuildFormattedCorrect:
    def test_answerable_with_references(self) -> None:
        record = {
            "expected_behavior": "ANSWER_CORRECTLY",
            "reference_answers": ["Laura Haddock", "Haddock"],
        }
        assert build_formatted_correct(record) == "Laura Haddock\nHaddock"

    def test_unanswerable_gets_the_placeholder(self) -> None:
        record = {
            "expected_behavior": "REFUSE_AMBIGUOUS_QUERY",
            "reference_answers": ["should not be shown"],
        }
        assert "No reference answers provided" in build_formatted_correct(record)

    def test_answerable_without_references_gets_the_placeholder(self) -> None:
        record = {"expected_behavior": "ANSWER_CORRECTLY", "reference_answers": []}
        assert "No reference answers provided" in build_formatted_correct(record)

    def test_legacy_field_name_is_honoured(self) -> None:
        record = {
            "expected_rag_behavior": "ANSWER_CORRECTLY",
            "reference_answers": ["Laura Haddock"],
        }
        assert build_formatted_correct(record) == "Laura Haddock"


class TestBuildJudgePrompt:
    def test_matches_the_original_pipeline_template(self) -> None:
        """
        comparability with the published english numbers rests on this prompt
        being the one run_models_all.py uses, so pin it byte for byte.
        """
        source = (
            Path(__file__).resolve().parents[1]
            / "refusalbench"
            / "naturalquestions"
            / "run_models_all.py"
        ).read_text(encoding="utf-8")
        start = source.find("Analyze the following model response")
        assert start > 0, "template not found in run_models_all.py"
        original = source[start : source.find('"""', start)]

        ours = build_judge_prompt("{query}", "{model_output}", "{formatted_correct}")
        assert ours == original

    def test_inputs_are_interpolated(self) -> None:
        prompt = build_judge_prompt("my query", "my output", "my references")
        assert "QUERY:\nmy query" in prompt
        assert "MODEL RESPONSE:\nmy output" in prompt
        assert "REFERENCE ANSWERS:\nmy references" in prompt


def _fake_segments(language: str) -> dict[str, str]:
    """minimal but complete judge segments, shaped like the real csv rows."""
    return {
        "judge_intro": f"[{language}] intro",
        "judge_step1": f"[{language}] step one, codes stay english: REFUSE_OTHER",
        "judge_step2": f"[{language}] step two",
        "judge_input_labels": (
            "[Q]:\n{query}\n\n[R]:\n{model_output}\n\n[A]:\n{formatted_correct}"
        ),
        "judge_output_format": (
            f"[{language}] format:\nCLASSIFICATION: [...]\n"
            "QUALITY_SCORE: [...]\nEXPLANATION: [...]"
        ),
    }


class TestBuildJudgePromptLocalised:
    def test_segments_are_joined_in_order(self) -> None:
        prompt = build_judge_prompt_localised(
            _fake_segments("pl"), "my query", "my output", "my references"
        )
        positions = [
            prompt.index("[pl] intro"),
            prompt.index("[pl] step one"),
            prompt.index("[pl] step two"),
            prompt.index("[Q]:"),
            prompt.index("[pl] format:"),
        ]
        assert positions == sorted(positions)

    def test_placeholders_are_filled(self) -> None:
        prompt = build_judge_prompt_localised(
            _fake_segments("ru"), "my query", "my output", "my references"
        )
        assert "[Q]:\nmy query" in prompt
        assert "[R]:\nmy output" in prompt
        assert "[A]:\nmy references" in prompt
        assert "{query}" not in prompt

    def test_braces_in_the_content_are_left_alone(self) -> None:
        """
        substitution is by replace rather than format, so a model response that
        happens to contain braces does not blow up or get reinterpreted.
        """
        prompt = build_judge_prompt_localised(
            _fake_segments("en"), "q", "the answer is {not a placeholder}", "r"
        )
        assert "{not a placeholder}" in prompt

    @needs_prompts
    def test_english_composition_matches_the_canonical_template(self) -> None:
        """
        the localised builder must reduce to the canonical english prompt, up to
        trailing whitespace the csv does not preserve. anything more than that
        would mean the two paths have drifted apart.
        """
        segments = load_judge_prompt_segments(PROMPTS_CSV)["en"]
        localised = build_judge_prompt_localised(segments, "Q", "M", "R")
        canonical = build_judge_prompt("Q", "M", "R")

        def normalise(text: str) -> str:
            return "\n".join(line.rstrip() for line in text.splitlines())

        assert normalise(localised) == normalise(canonical)

    @needs_prompts
    @pytest.mark.parametrize("language", sorted(LABELS))
    def test_output_keys_and_codes_stay_english(self, language: str) -> None:
        """
        the parser looks for english keys and label values, so a translation
        that localised them would silently break every judgment.
        """
        segments = load_judge_prompt_segments(PROMPTS_CSV)[language]
        prompt = build_judge_prompt_localised(segments, "Q", "M", "R")
        for token in [
            "CLASSIFICATION:",
            "QUALITY_SCORE:",
            "EXPLANATION:",
            "answer_attempt",
            "REFUSE_AMBIGUOUS_QUERY",
            "REFUSE_OTHER",
        ]:
            assert token in prompt, f"{token} missing from the {language} prompt"


class TestLoadJudgePromptSegments:
    @needs_prompts
    def test_every_language_has_every_segment(self) -> None:
        segments = load_judge_prompt_segments(PROMPTS_CSV)
        assert set(segments) == {"en", "pl", "ru", "zh_cmn", "zh_yue"}
        for language, found in segments.items():
            assert set(found) == set(JUDGE_SEGMENT_IDS), language

    @needs_prompts
    def test_translations_are_not_just_the_english_text(self) -> None:
        segments = load_judge_prompt_segments(PROMPTS_CSV)
        for language in ["pl", "ru", "zh_cmn", "zh_yue"]:
            assert segments[language]["judge_intro"] != segments["en"]["judge_intro"]

    def test_a_missing_segment_is_rejected(self, tmp_path: Path) -> None:
        """
        a half translated judge prompt would mix languages inside one call, so
        this has to fail loudly rather than fall back per segment.
        """
        csv_path = tmp_path / "prompts.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "segment_id",
                    "component",
                    "english_text",
                    "PL",
                    "RU",
                    "ZH_YUE",
                    "ZH_CMN",
                    "notes",
                ]
            )
            for segment_id in JUDGE_SEGMENT_IDS:
                pl = "" if segment_id == "judge_step2" else f"pl {segment_id}"
                writer.writerow(
                    [
                        segment_id,
                        "judge",
                        f"en {segment_id}",
                        pl,
                        f"ru {segment_id}",
                        f"yue {segment_id}",
                        f"cmn {segment_id}",
                        "",
                    ]
                )

        with pytest.raises(ValueError, match="judge_step2"):
            load_judge_prompt_segments(csv_path)


class TestResumeKeys:
    def test_prompt_language_is_part_of_the_resume_key(self, tmp_path: Path) -> None:
        """
        an english run and a target-language run over the same records must not
        skip each other when they share an output directory.
        """
        path = tmp_path / "judgments_raw.jsonl"
        path.write_text(
            "\n".join(
                json.dumps(row)
                for row in [
                    {
                        "judge": "j",
                        "language": "pl",
                        "id": "1",
                        "status": "ok",
                        "judge_prompt_language": "en",
                    },
                    {
                        "judge": "j",
                        "language": "pl",
                        "id": "1",
                        "status": "ok",
                        "judge_prompt_language": "pl",
                    },
                ]
            ),
            encoding="utf-8",
        )
        assert load_completed_ok_keys(path) == {
            ("j", "pl", "1", "en"),
            ("j", "pl", "1", "pl"),
        }

    def test_older_rows_without_the_field_count_as_english(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "judgments_raw.jsonl"
        path.write_text(
            json.dumps({"judge": "j", "language": "pl", "id": "1", "status": "ok"}),
            encoding="utf-8",
        )
        assert load_completed_ok_keys(path) == {("j", "pl", "1", "en")}

    def test_failed_rows_are_not_treated_as_done(self, tmp_path: Path) -> None:
        path = tmp_path / "judgments_raw.jsonl"
        path.write_text(
            json.dumps({"judge": "j", "language": "pl", "id": "1", "status": "error"}),
            encoding="utf-8",
        )
        assert load_completed_ok_keys(path) == set()


def _work_item() -> WorkItem:
    """a minimal work item for the process_work_item tests."""
    return WorkItem(
        judge_name="judge_a",
        model_id="bedrock/converse/test",
        language="en",
        record_id="RB-1",
        record={
            "id": "RB-1",
            "expected_behavior": "ANSWER_CORRECTLY",
            "reference_answers": ["Laura Haddock"],
            "response": "Laura Haddock",
            "messages": _user_message("en"),
        },
    )


def _run_process(tmp_path: Path) -> tuple[Path, Path]:
    """drive process_work_item against the currently patched judge stub."""
    judgments_path = tmp_path / "judgments_raw.jsonl"
    failures_path = tmp_path / "failures.csv"

    async def scenario() -> None:
        await process_work_item(
            _work_item(),
            {"en": LABELS["en"]},
            {"judge_a": AdaptiveConcurrency(default_limit=1)},
            None,
            judgments_path,
            failures_path,
            asyncio.Lock(),
            asyncio.Lock(),
        )

    asyncio.run(scenario())
    return judgments_path, failures_path


class TestProcessWorkItem:
    def test_successful_judgment_is_appended(self, tmp_path: Path, monkeypatch) -> None:
        async def fake_judge(
            item, labels, adaptive, max_retry_minutes, segments=None
        ) -> dict:
            return {
                "judge": item.judge_name,
                "id": item.record_id,
                "status": "ok",
                "classification": "answer_attempt",
            }

        monkeypatch.setattr(judge_bedrock, "judge_one_record", fake_judge)
        judgments_path, failures_path = _run_process(tmp_path)

        rows = [json.loads(line) for line in judgments_path.read_text().splitlines()]
        assert len(rows) == 1
        assert rows[0]["status"] == "ok"
        assert not failures_path.exists()

    def test_retry_budget_rearms_are_bounded(self, tmp_path: Path, monkeypatch) -> None:
        """
        regression test. re-arming for ever means --max-retry-minutes has no
        effect and one broken item can hold a worker until the run is killed.
        """
        calls = {"n": 0}

        async def always_over_budget(
            item, labels, adaptive, max_retry_minutes, segments=None
        ) -> dict:
            calls["n"] += 1
            raise RetryBudgetExceeded("budget gone")

        async def no_sleep(_seconds: float) -> None:
            return None

        monkeypatch.setattr(judge_bedrock, "judge_one_record", always_over_budget)
        monkeypatch.setattr(judge_bedrock.asyncio, "sleep", no_sleep)
        judgments_path, failures_path = _run_process(tmp_path)

        assert calls["n"] == MAX_RETRY_BUDGET_REARMS + 1

        rows = [json.loads(line) for line in judgments_path.read_text().splitlines()]
        assert len(rows) == 1
        assert rows[0]["status"] == "error"
        assert "retry budget exhausted" in rows[0]["error"]

        failure_text = failures_path.read_text(encoding="utf-8")
        assert "RB-1" in failure_text

    def test_recovers_when_a_rearm_succeeds(self, tmp_path: Path, monkeypatch) -> None:
        calls = {"n": 0}

        async def fails_once(
            item, labels, adaptive, max_retry_minutes, segments=None
        ) -> dict:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RetryBudgetExceeded("budget gone")
            return {
                "judge": item.judge_name,
                "id": item.record_id,
                "status": "ok",
                "classification": "answer_attempt",
            }

        async def no_sleep(_seconds: float) -> None:
            return None

        monkeypatch.setattr(judge_bedrock, "judge_one_record", fails_once)
        monkeypatch.setattr(judge_bedrock.asyncio, "sleep", no_sleep)
        judgments_path, failures_path = _run_process(tmp_path)

        rows = [json.loads(line) for line in judgments_path.read_text().splitlines()]
        assert len(rows) == 1
        assert rows[0]["status"] == "ok"
        assert not failures_path.exists()


class TestJudgeOneRecord:
    def test_invalid_label_triggers_the_format_reminder(self, monkeypatch) -> None:
        """
        a misspelled code must not pass through. the first call gets one more
        chance with the format reminder appended.
        """
        prompts: list[str] = []

        async def fake_call(
            model_id: str, prompt: str, adaptive: Any, max_retry_minutes: Any
        ) -> str:
            prompts.append(prompt)
            if len(prompts) == 1:
                return (
                    "CLASSIFICATION: REFUSE_NONFACTICAL_QUERY\n"
                    "QUALITY_SCORE: N/A\n"
                    "EXPLANATION: typo code"
                )
            return (
                "CLASSIFICATION: REFUSE_NONFACTUAL_QUERY\n"
                "QUALITY_SCORE: N/A\n"
                "EXPLANATION: corrected"
            )

        monkeypatch.setattr(judge_bedrock, "call_judge_model", fake_call)
        result = asyncio.run(
            judge_bedrock.judge_one_record(_work_item(), LABELS["en"], None, None)
        )
        assert len(prompts) == 2
        assert judge_bedrock.FORMAT_REMINDER in prompts[1]
        assert result["classification"] == "REFUSE_NONFACTUAL_QUERY"

    def test_persistently_invalid_label_becomes_a_permanent_failure(
        self, monkeypatch
    ) -> None:
        async def fake_call(
            model_id: str, prompt: str, adaptive: Any, max_retry_minutes: Any
        ) -> str:
            return (
                "CLASSIFICATION: REFUSE_NONFACTICAL_QUERY\n"
                "QUALITY_SCORE: N/A\n"
                "EXPLANATION: still wrong"
            )

        monkeypatch.setattr(judge_bedrock, "call_judge_model", fake_call)
        with pytest.raises(judge_bedrock.PermanentBedrockError):
            asyncio.run(
                judge_bedrock.judge_one_record(_work_item(), LABELS["en"], None, None)
            )

    def test_target_language_prompt_is_used_and_recorded(self, monkeypatch) -> None:
        prompts: list[str] = []

        async def fake_call(
            model_id: str, prompt: str, adaptive: Any, max_retry_minutes: Any
        ) -> str:
            prompts.append(prompt)
            return (
                "CLASSIFICATION: answer_attempt\nQUALITY_SCORE: 5\nEXPLANATION: correct"
            )

        monkeypatch.setattr(judge_bedrock, "call_judge_model", fake_call)
        result = asyncio.run(
            judge_bedrock.judge_one_record(
                _work_item(), LABELS["en"], None, None, _fake_segments("pl")
            )
        )
        assert "[pl] intro" in prompts[0]
        assert result["judge_prompt_language"] == "en", (
            "the work item language is en here, so the prompt language follows it"
        )

    def test_english_prompt_is_the_default_and_is_recorded(self, monkeypatch) -> None:
        async def fake_call(
            model_id: str, prompt: str, adaptive: Any, max_retry_minutes: Any
        ) -> str:
            assert prompt.startswith("Analyze the following model response")
            return (
                "CLASSIFICATION: answer_attempt\nQUALITY_SCORE: 5\nEXPLANATION: correct"
            )

        monkeypatch.setattr(judge_bedrock, "call_judge_model", fake_call)
        result = asyncio.run(
            judge_bedrock.judge_one_record(_work_item(), LABELS["en"], None, None)
        )
        assert result["judge_prompt_language"] == "en"

    def test_localised_format_reminder_is_appended_on_a_bad_response(
        self, monkeypatch
    ) -> None:
        """
        the reminder has to be in the same language as the prompt, otherwise a
        target-language run switches language halfway through a single call.
        """
        prompts: list[str] = []

        async def fake_call(
            model_id: str, prompt: str, adaptive: Any, max_retry_minutes: Any
        ) -> str:
            prompts.append(prompt)
            if len(prompts) == 1:
                return "no structure at all"
            return "CLASSIFICATION: answer_attempt\nQUALITY_SCORE: 4\nEXPLANATION: fine"

        monkeypatch.setattr(judge_bedrock, "call_judge_model", fake_call)
        asyncio.run(
            judge_bedrock.judge_one_record(
                _work_item(), LABELS["en"], None, None, _fake_segments("ru")
            )
        )
        assert len(prompts) == 2
        assert prompts[1].endswith(_fake_segments("ru")["judge_output_format"])
        assert judge_bedrock.FORMAT_REMINDER not in prompts[1]

    def test_valid_response_is_accepted_on_the_first_call(self, monkeypatch) -> None:
        calls = {"n": 0}

        async def fake_call(
            model_id: str, prompt: str, adaptive: Any, max_retry_minutes: Any
        ) -> str:
            calls["n"] += 1
            return (
                "CLASSIFICATION: answer_attempt\nQUALITY_SCORE: 5\nEXPLANATION: correct"
            )

        monkeypatch.setattr(judge_bedrock, "call_judge_model", fake_call)
        result = asyncio.run(
            judge_bedrock.judge_one_record(_work_item(), LABELS["en"], None, None)
        )
        assert calls["n"] == 1
        assert result["classification"] == "answer_attempt"
        assert result["quality_score"] == pytest.approx(5.0)
        assert result["query_extracted"] == QUERIES["en"]
