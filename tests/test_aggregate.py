"""tests for multilingual aggregation maths and dual-eval rules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from refusalbench.multilingual.aggregate import (
    ANSWER_CORRECTLY,
    _attach_delta_vs_en,
    _cohens_kappa,
    _fleiss_kappa,
    _majority_vote,
    _spearman_corr,
    apply_dual_eval,
    build_consensus_df,
    compute_metrics_block,
    find_unknown_classifications,
)


class TestCohensKappa:
    """cohen's kappa against hand-computable confusion matrices."""

    def test_known_two_by_two(self) -> None:
        """
        confusion [[20, 5], [10, 15]] gives po=0.70, pe=0.50, kappa=0.40.
        """
        labels_a = ["a"] * 25 + ["b"] * 25
        labels_b = ["a"] * 20 + ["b"] * 5 + ["a"] * 10 + ["b"] * 15
        assert _cohens_kappa(labels_a, labels_b) == pytest.approx(0.40)

    def test_perfect_agreement(self) -> None:
        labels = ["a", "b", "a", "c", "b"]
        assert _cohens_kappa(labels, labels) == pytest.approx(1.0)

    def test_chance_agreement_is_zero(self) -> None:
        """
        independent raters with identical marginals score zero, not the raw
        agreement rate.
        """
        labels_a = ["a", "a", "b", "b"]
        labels_b = ["a", "b", "a", "b"]
        assert _cohens_kappa(labels_a, labels_b) == pytest.approx(0.0)

    def test_systematic_disagreement_is_negative(self) -> None:
        labels_a = ["a"] * 5 + ["b"] * 5
        labels_b = ["b"] * 5 + ["a"] * 5
        assert _cohens_kappa(labels_a, labels_b) == pytest.approx(-1.0)

    def test_single_category_everywhere(self) -> None:
        """pe is 1.0 here, so the degenerate branch must return 1.0."""
        labels = ["a"] * 6
        assert _cohens_kappa(labels, labels) == pytest.approx(1.0)

    def test_undefined_inputs_return_nan(self) -> None:
        assert np.isnan(_cohens_kappa([], []))
        assert np.isnan(_cohens_kappa(["a"], ["a", "b"]))


class TestFleissKappa:
    """fleiss kappa against published and hand-computable matrices."""

    def test_fleiss_1971_example(self) -> None:
        """
        the worked example from fleiss (1971), 10 subjects rated by 14 raters
        across 5 categories, published kappa 0.210.
        """
        counts = np.array(
            [
                [0, 0, 0, 0, 14],
                [0, 2, 6, 4, 2],
                [0, 0, 3, 5, 6],
                [0, 3, 9, 2, 0],
                [2, 2, 8, 1, 1],
                [7, 7, 0, 0, 0],
                [3, 2, 6, 3, 0],
                [2, 5, 3, 2, 2],
                [6, 5, 2, 1, 0],
                [0, 2, 2, 3, 7],
            ],
            dtype=float,
        )
        assert _fleiss_kappa(counts) == pytest.approx(0.210, abs=0.001)

    def test_unanimous_items_score_one(self) -> None:
        counts = np.array([[2, 0], [0, 2]], dtype=float)
        assert _fleiss_kappa(counts) == pytest.approx(1.0)

    def test_maximally_split_items_score_minus_one(self) -> None:
        counts = np.array([[1, 1], [1, 1]], dtype=float)
        assert _fleiss_kappa(counts) == pytest.approx(-1.0)

    def test_partial_agreement_hand_computed(self) -> None:
        """
        four items, three raters, two categories. p_bar is 2/3 and p_e is 0.5,
        so kappa is 1/3.
        """
        counts = np.array([[3, 0], [2, 1], [1, 2], [0, 3]], dtype=float)
        assert _fleiss_kappa(counts) == pytest.approx(1.0 / 3.0)

    def test_items_with_one_rater_are_dropped(self) -> None:
        """
        a single-rater item carries no within-item agreement information and
        would divide by zero, so it must be filtered out rather than poison
        the result.
        """
        counts = np.array([[3, 0], [2, 1], [1, 2], [0, 3], [1, 0]], dtype=float)
        assert _fleiss_kappa(counts) == pytest.approx(1.0 / 3.0)

    def test_empty_matrix_returns_nan(self) -> None:
        assert np.isnan(_fleiss_kappa(np.zeros((0, 0))))


class TestSpearman:
    def test_monotone_increasing(self) -> None:
        a = pd.Series([1.0, 2.0, 3.0, 4.0])
        b = pd.Series([10.0, 20.0, 30.0, 40.0])
        assert _spearman_corr(a, b) == pytest.approx(1.0)

    def test_monotone_decreasing(self) -> None:
        a = pd.Series([1.0, 2.0, 3.0, 4.0])
        b = pd.Series([4.0, 3.0, 2.0, 1.0])
        assert _spearman_corr(a, b) == pytest.approx(-1.0)

    def test_rank_based_not_value_based(self) -> None:
        """a nonlinear but monotone relation still scores 1.0."""
        a = pd.Series([1.0, 2.0, 3.0, 4.0])
        b = pd.Series([1.0, 4.0, 9.0, 16.0])
        assert _spearman_corr(a, b) == pytest.approx(1.0)

    def test_too_few_pairs_returns_nan(self) -> None:
        assert np.isnan(_spearman_corr(pd.Series([1.0]), pd.Series([2.0])))


class TestMajorityVote:
    def test_unanimous(self) -> None:
        assert _majority_vote(["a", "a", "a"]) == ("a", 3, False)

    def test_simple_majority(self) -> None:
        assert _majority_vote(["a", "a", "b"]) == ("a", 2, False)

    def test_three_way_split_has_no_majority(self) -> None:
        majority, n_agree, no_majority = _majority_vote(["a", "b", "c"])
        assert majority is None
        assert no_majority is True
        assert n_agree == 1

    def test_even_tie_has_no_majority(self) -> None:
        majority, _, no_majority = _majority_vote(["a", "a", "b", "b"])
        assert majority is None
        assert no_majority is True

    def test_empty_has_no_majority(self) -> None:
        assert _majority_vote([]) == (None, 0, True)


class TestApplyDualEval:
    """the four branches of the refusalbench dual-eval rule."""

    def test_answerable_answered_keeps_quality_score(self) -> None:
        quality, refusal_match, explanation = apply_dual_eval(
            ANSWER_CORRECTLY, "answer_attempt", 4.0, "looks right"
        )
        assert quality == 4.0
        assert refusal_match is None
        assert explanation == "looks right"

    def test_answerable_refused_is_a_false_refusal(self) -> None:
        quality, refusal_match, explanation = apply_dual_eval(
            ANSWER_CORRECTLY, "REFUSE_INFO_MISSING_IN_CONTEXT", 5.0, "ignored"
        )
        assert quality is None
        assert refusal_match is False
        assert "False Refusal" in explanation

    def test_unanswerable_answered_is_a_missed_refusal(self) -> None:
        quality, refusal_match, explanation = apply_dual_eval(
            "REFUSE_AMBIGUOUS_QUERY", "answer_attempt", 5.0, "ignored"
        )
        assert quality is None
        assert refusal_match is False
        assert "Missed Refusal" in explanation

    def test_unanswerable_refused_with_matching_code(self) -> None:
        quality, refusal_match, explanation = apply_dual_eval(
            "REFUSE_AMBIGUOUS_QUERY", "REFUSE_AMBIGUOUS_QUERY", None, None
        )
        assert quality is None
        assert refusal_match is True
        assert "Exact match" in explanation

    def test_unanswerable_refused_with_wrong_code(self) -> None:
        """
        the largest bucket in the pilot. it must score as incorrect, not as a
        partially correct refusal.
        """
        _, refusal_match, explanation = apply_dual_eval(
            "REFUSE_GRANULARITY_MISMATCH", "REFUSE_INFO_MISSING_IN_CONTEXT", None, None
        )
        assert refusal_match is False
        assert "Mismatch" in explanation

    def test_quality_is_dropped_on_every_refusal_path(self) -> None:
        """a quality score on a refusal row would leak into answer accuracy."""
        for gt, pred in [
            (ANSWER_CORRECTLY, "REFUSE_OTHER"),
            ("REFUSE_OTHER", "answer_attempt"),
            ("REFUSE_OTHER", "REFUSE_OTHER"),
        ]:
            quality, _, _ = apply_dual_eval(gt, pred, 5.0, None)
            assert quality is None


def _row(
    language: str,
    example_id: str,
    ground_truth: str,
    predicted: str,
    quality: float | None = None,
    refusal_match: bool | None = None,
    judge_id: str = "judge_a",
    perturbation_class: str = "P-Ambiguity",
    intensity: str = "LOW",
) -> dict:
    """build one merged judgment row for the metric tests."""
    return {
        "judge_id": judge_id,
        "language": language,
        "id": example_id,
        "status": "ok",
        "ground_truth_label": ground_truth,
        "model_predicted_type": predicted,
        "answer_quality_score": quality,
        "refusal_match_correct": refusal_match,
        "perturbation_class": perturbation_class,
        "intensity": intensity,
    }


class TestComputeMetricsBlock:
    def test_mixed_slice_computes_every_metric(self) -> None:
        df = pd.DataFrame(
            [
                _row("en", "1", ANSWER_CORRECTLY, "answer_attempt", 5.0),
                _row("en", "2", ANSWER_CORRECTLY, "answer_attempt", 3.0),
                _row("en", "3", ANSWER_CORRECTLY, "REFUSE_OTHER"),
                _row("en", "4", ANSWER_CORRECTLY, "answer_attempt", 4.0),
                _row(
                    "en",
                    "5",
                    "REFUSE_AMBIGUOUS_QUERY",
                    "REFUSE_AMBIGUOUS_QUERY",
                    refusal_match=True,
                ),
                _row(
                    "en",
                    "6",
                    "REFUSE_AMBIGUOUS_QUERY",
                    "REFUSE_OTHER",
                    refusal_match=False,
                ),
                _row(
                    "en",
                    "7",
                    "REFUSE_AMBIGUOUS_QUERY",
                    "answer_attempt",
                    refusal_match=False,
                ),
                _row(
                    "en",
                    "8",
                    "REFUSE_AMBIGUOUS_QUERY",
                    "REFUSE_AMBIGUOUS_QUERY",
                    refusal_match=True,
                ),
            ]
        )
        block = compute_metrics_block(df)
        assert block["n"] == 8
        assert block["num_answerable"] == 4
        assert block["num_unanswerable"] == 4
        # scores of 5 and 4 clear the gate, 3 does not, and the refusal does not
        assert block["answer_accuracy"] == pytest.approx(0.5)
        assert block["false_refusal_rate"] == pytest.approx(0.25)
        assert block["refusal_accuracy"] == pytest.approx(0.5)
        assert block["missed_refusal_rate"] == pytest.approx(0.25)
        assert block["calibrated_refusal_score"] == pytest.approx(0.5)

    def test_quality_gate_is_at_least_four(self) -> None:
        df = pd.DataFrame(
            [
                _row("en", "1", ANSWER_CORRECTLY, "answer_attempt", 3.9),
                _row("en", "2", ANSWER_CORRECTLY, "answer_attempt", 4.0),
            ]
        )
        assert compute_metrics_block(df)["answer_accuracy"] == pytest.approx(0.5)

    def test_pure_answerable_slice_leaves_refusal_metrics_undefined(self) -> None:
        """
        every stratum row in the pilot is pure. reporting zero here rather than
        nan is what would make calibrated_refusal_score look like half a real
        number.
        """
        df = pd.DataFrame([_row("en", "1", ANSWER_CORRECTLY, "answer_attempt", 5.0)])
        block = compute_metrics_block(df)
        assert block["answer_accuracy"] == pytest.approx(1.0)
        assert np.isnan(block["refusal_accuracy"])
        assert np.isnan(block["missed_refusal_rate"])
        assert np.isnan(block["calibrated_refusal_score"])
        assert block["num_unanswerable"] == 0

    def test_pure_unanswerable_slice_leaves_answer_metrics_undefined(self) -> None:
        df = pd.DataFrame(
            [
                _row("en", "1", "REFUSE_OTHER", "REFUSE_OTHER", refusal_match=True),
            ]
        )
        block = compute_metrics_block(df)
        assert block["refusal_accuracy"] == pytest.approx(1.0)
        assert np.isnan(block["answer_accuracy"])
        assert np.isnan(block["false_refusal_rate"])
        assert np.isnan(block["calibrated_refusal_score"])

    def test_non_ok_rows_are_excluded(self) -> None:
        rows = [
            _row("en", "1", ANSWER_CORRECTLY, "answer_attempt", 5.0),
            _row("en", "2", ANSWER_CORRECTLY, "answer_attempt", 5.0),
        ]
        rows[1]["status"] = "error"
        assert compute_metrics_block(pd.DataFrame(rows))["n"] == 1

    def test_unknown_label_counts_in_denominator_only(self) -> None:
        """
        documents the behaviour that motivates label validation at judge time.
        a misspelled code is neither an answer attempt nor a refusal, so both
        answerable numerators ignore it while n still counts it.
        """
        df = pd.DataFrame(
            [
                _row("en", "1", ANSWER_CORRECTLY, "REFUSE_NONFACTICAL_QUERY"),
                _row("en", "2", ANSWER_CORRECTLY, "answer_attempt", 5.0),
            ]
        )
        block = compute_metrics_block(df)
        assert block["num_answerable"] == 2
        assert block["answer_accuracy"] == pytest.approx(0.5)
        assert block["false_refusal_rate"] == pytest.approx(0.0)


class TestFindUnknownClassifications:
    def test_flags_misspelled_code(self) -> None:
        df = pd.DataFrame(
            [
                _row("en", "1", ANSWER_CORRECTLY, "REFUSE_NONFACTICAL_QUERY"),
                _row("en", "2", ANSWER_CORRECTLY, "answer_attempt", 5.0),
                _row("en", "3", "REFUSE_OTHER", "REFUSE_OTHER", refusal_match=True),
            ]
        )
        assert find_unknown_classifications(df) == {"REFUSE_NONFACTICAL_QUERY": 1}

    def test_clean_frame_reports_nothing(self) -> None:
        df = pd.DataFrame([_row("en", "1", ANSWER_CORRECTLY, "answer_attempt", 5.0)])
        assert find_unknown_classifications(df) == {}


class TestBuildConsensus:
    def test_majority_label_and_median_quality(self) -> None:
        rows = [
            _row("en", "1", ANSWER_CORRECTLY, "answer_attempt", 5.0, judge_id="a"),
            _row("en", "1", ANSWER_CORRECTLY, "answer_attempt", 3.0, judge_id="b"),
            _row("en", "1", ANSWER_CORRECTLY, "REFUSE_OTHER", None, judge_id="c"),
        ]
        consensus = build_consensus_df(pd.DataFrame(rows))
        assert len(consensus) == 1
        row = consensus.iloc[0]
        assert row["consensus_classification"] == "answer_attempt"
        assert row["n_agree"] == 2
        assert bool(row["no_majority"]) is False
        # median over the two judges that voted with the majority
        assert row["answer_quality_score"] == pytest.approx(4.0)

    def test_quality_from_dissenting_judges_is_ignored(self) -> None:
        """
        a judge that called it a refusal must not contribute a quality score to
        the consensus answer.
        """
        rows = [
            _row("en", "1", ANSWER_CORRECTLY, "answer_attempt", 5.0, judge_id="a"),
            _row("en", "1", ANSWER_CORRECTLY, "answer_attempt", 5.0, judge_id="b"),
            _row("en", "1", ANSWER_CORRECTLY, "REFUSE_OTHER", 1.0, judge_id="c"),
        ]
        consensus = build_consensus_df(pd.DataFrame(rows))
        assert consensus.iloc[0]["answer_quality_score"] == pytest.approx(5.0)

    def test_three_way_split_is_marked_no_majority(self) -> None:
        rows = [
            _row("en", "1", "REFUSE_OTHER", "REFUSE_OTHER", judge_id="a"),
            _row("en", "1", "REFUSE_OTHER", "REFUSE_AMBIGUOUS_QUERY", judge_id="b"),
            _row("en", "1", "REFUSE_OTHER", "answer_attempt", judge_id="c"),
        ]
        consensus = build_consensus_df(pd.DataFrame(rows))
        assert bool(consensus.iloc[0]["no_majority"]) is True
        assert consensus.iloc[0]["consensus_classification"] is None

    def test_errored_judge_rows_are_excluded(self) -> None:
        rows = [
            _row("en", "1", ANSWER_CORRECTLY, "answer_attempt", 5.0, judge_id="a"),
            _row("en", "1", ANSWER_CORRECTLY, "answer_attempt", 5.0, judge_id="b"),
            _row("en", "1", ANSWER_CORRECTLY, "REFUSE_OTHER", None, judge_id="c"),
        ]
        rows[2]["status"] = "error"
        consensus = build_consensus_df(pd.DataFrame(rows))
        assert consensus.iloc[0]["n_judges_ok"] == 2


class TestAttachDeltaVsEn:
    def test_english_differs_from_itself_by_zero(self) -> None:
        metrics = pd.DataFrame(
            [
                {
                    "language": "en",
                    "judge_id": "a",
                    "answer_accuracy": 0.8,
                    "refusal_accuracy": 0.4,
                    "false_refusal_rate": 0.1,
                    "missed_refusal_rate": 0.2,
                    "calibrated_refusal_score": 0.6,
                },
                {
                    "language": "pl",
                    "judge_id": "a",
                    "answer_accuracy": 0.5,
                    "refusal_accuracy": 0.4,
                    "false_refusal_rate": 0.3,
                    "missed_refusal_rate": 0.2,
                    "calibrated_refusal_score": 0.45,
                },
            ]
        )
        out = _attach_delta_vs_en(metrics)
        en = out[out["language"] == "en"].iloc[0]
        pl = out[out["language"] == "pl"].iloc[0]
        assert en["delta_vs_en_answer_accuracy"] == pytest.approx(0.0)
        assert pl["delta_vs_en_answer_accuracy"] == pytest.approx(-0.3)

    def test_stratum_baseline_is_the_matching_english_cell(self) -> None:
        """
        the baseline must be the english row of the same stratum. grouping on
        the judge alone would difference a HIGH row against a LOW english cell.
        """
        metrics = pd.DataFrame(
            [
                {
                    "language": "en",
                    "judge_id": "a",
                    "perturbation_class": "P-A",
                    "intensity": "LOW",
                    "answer_accuracy": 0.9,
                    "refusal_accuracy": np.nan,
                    "false_refusal_rate": 0.1,
                    "missed_refusal_rate": np.nan,
                    "calibrated_refusal_score": np.nan,
                },
                {
                    "language": "en",
                    "judge_id": "a",
                    "perturbation_class": "P-A",
                    "intensity": "HIGH",
                    "answer_accuracy": 0.2,
                    "refusal_accuracy": np.nan,
                    "false_refusal_rate": 0.5,
                    "missed_refusal_rate": np.nan,
                    "calibrated_refusal_score": np.nan,
                },
                {
                    "language": "pl",
                    "judge_id": "a",
                    "perturbation_class": "P-A",
                    "intensity": "HIGH",
                    "answer_accuracy": 0.1,
                    "refusal_accuracy": np.nan,
                    "false_refusal_rate": 0.6,
                    "missed_refusal_rate": np.nan,
                    "calibrated_refusal_score": np.nan,
                },
            ]
        )
        out = _attach_delta_vs_en(
            metrics, group_cols=("judge_id", "perturbation_class", "intensity")
        )
        pl_high = out[(out["language"] == "pl")].iloc[0]
        # against the HIGH english cell (0.2), not the LOW one (0.9)
        assert pl_high["delta_vs_en_answer_accuracy"] == pytest.approx(-0.1)

    def test_group_without_english_row_gets_nan(self) -> None:
        metrics = pd.DataFrame(
            [
                {
                    "language": "pl",
                    "judge_id": "a",
                    "answer_accuracy": 0.5,
                    "refusal_accuracy": 0.4,
                    "false_refusal_rate": 0.3,
                    "missed_refusal_rate": 0.2,
                    "calibrated_refusal_score": 0.45,
                },
            ]
        )
        out = _attach_delta_vs_en(metrics)
        assert np.isnan(out.iloc[0]["delta_vs_en_answer_accuracy"])
