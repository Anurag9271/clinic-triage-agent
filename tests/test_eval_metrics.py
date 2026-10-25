"""Unit tests for triage evaluation metrics (eval/metrics.py).

Tests accuracy, under-triage rate, over-triage rate, emergency recall,
JSON validity rate, and latency percentiles with hand-built examples.
Runs hermetically without models or local Ollama instances.
"""

from __future__ import annotations

import pytest

from eval.metrics import (
    URGENCY_RANKS,
    accuracy,
    compute_metrics,
    emergency_recall,
    json_validity_rate,
    median_latency,
    over_triage_rate,
    percentile_95,
    under_triage_rate,
)


def test_urgency_ranks_hierarchy():
    """Verify standard clinical ordering routine < soon < urgent < emergency."""
    assert URGENCY_RANKS["routine"] < URGENCY_RANKS["soon"]
    assert URGENCY_RANKS["soon"] < URGENCY_RANKS["urgent"]
    assert URGENCY_RANKS["urgent"] < URGENCY_RANKS["emergency"]


def test_accuracy_hand_built():
    """Test classification accuracy with simple hand-built cases."""
    # Perfect accuracy
    assert accuracy(["a", "b", "c"], ["a", "b", "c"]) == 1.0

    # 3 out of 4 correct = 0.75
    assert accuracy(["a", "b", "c", "d"], ["a", "b", "c", "wrong"]) == 0.75

    # 0 out of 3 correct = 0.0
    assert accuracy(["a", "b", "c"], ["x", "y", "z"]) == 0.0

    # Empty inputs
    assert accuracy([], []) == 0.0

    # Mismatched lengths
    with pytest.raises(ValueError, match="Length mismatch"):
        accuracy(["a"], ["a", "b"])


def test_under_and_over_triage_rates_hand_built():
    """Test under-triage and over-triage with explicit hand-built scenarios."""
    # Case 1: Mixed clinical prioritization
    # true: routine (0), pred: routine (0) -> correct
    # true: soon (1),    pred: routine (0) -> UNDER-TRIAGE (pred < true)
    # true: urgent (2),  pred: emergency (3) -> OVER-TRIAGE (pred > true)
    # true: emergency (3), pred: urgent (2) -> UNDER-TRIAGE (pred < true)
    y_true = ["routine", "soon", "urgent", "emergency"]
    y_pred = ["routine", "routine", "emergency", "urgent"]

    # 2 under-triages out of 4 = 0.50
    assert under_triage_rate(y_true, y_pred) == 0.50

    # 1 over-triage out of 4 = 0.25
    assert over_triage_rate(y_true, y_pred) == 0.25


def test_perfect_calibration_has_zero_under_and_over_triage():
    """Test that perfectly matching urgencies yield 0% under- and over-triage."""
    y = ["routine", "soon", "urgent", "emergency"]
    assert under_triage_rate(y, y) == 0.0
    assert over_triage_rate(y, y) == 0.0


def test_extreme_under_and_over_triage():
    """Test 100% under-triage and 100% over-triage scenarios."""
    # Everything predicted as routine when actual is urgent/emergency
    y_true_under = ["urgent", "emergency"]
    y_pred_under = ["routine", "routine"]
    assert under_triage_rate(y_true_under, y_pred_under) == 1.0
    assert over_triage_rate(y_true_under, y_pred_under) == 0.0

    # Everything predicted as emergency when actual is routine/soon
    y_true_over = ["routine", "soon"]
    y_pred_over = ["emergency", "emergency"]
    assert over_triage_rate(y_true_over, y_pred_over) == 1.0
    assert under_triage_rate(y_true_over, y_pred_over) == 0.0


def test_invalid_urgency_raises_value_error():
    """Test that unrecognized urgency values raise ValueError in triage rate calculations."""
    with pytest.raises(ValueError, match="Unrecognized urgency label"):
        under_triage_rate(["unknown_urgency"], ["routine"])

    with pytest.raises(ValueError, match="Unrecognized urgency label"):
        over_triage_rate(["routine"], ["invalid_val"])


def test_emergency_recall_hand_built():
    """Test emergency recall on hand-built subsets."""
    # 2 true emergencies, model caught 1
    y_true = ["emergency", "emergency", "urgent", "routine"]
    y_pred = ["emergency", "urgent", "emergency", "routine"]

    assert emergency_recall(y_true, y_pred) == 0.5

    # 100% recall
    assert emergency_recall(["emergency", "emergency"], ["emergency", "emergency"]) == 1.0

    # 0% recall (missed both emergencies)
    assert emergency_recall(["emergency", "emergency"], ["urgent", "soon"]) == 0.0

    # No true emergencies present in ground truth -> returns 1.0
    assert emergency_recall(["routine", "soon"], ["routine", "urgent"]) == 1.0

    # Empty inputs -> 0.0
    assert emergency_recall([], []) == 0.0


def test_json_validity_rate_hand_built():
    """Test JSON validity calculation based on fallback invocations."""
    # 1 fallback out of 4 -> 3/4 = 0.75 valid
    assert json_validity_rate([False, False, True, False]) == 0.75

    # All fallbacks -> 0.0 valid
    assert json_validity_rate([True, True]) == 0.0

    # Zero fallbacks -> 1.0 valid
    assert json_validity_rate([False, False, False]) == 1.0

    # Empty inputs -> 0.0
    assert json_validity_rate([]) == 0.0


def test_percentile_and_median_latency():
    """Test 95th percentile and median calculation on hand-built series."""
    latencies = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    p95 = percentile_95(latencies)
    assert 9.0 <= p95 <= 10.0

    med = median_latency(latencies)
    assert med == 5.5

    # Empty checks
    assert percentile_95([]) == 0.0
    assert median_latency([]) == 0.0


def test_compute_metrics_consolidated():
    """Test end-to-end consolidated metrics dictionary generation."""
    y_true_intent = ["appointment", "refill", "test_results", "administrative"]
    y_pred_intent = ["appointment", "refill", "test_results", "other"]  # 3/4 = 0.75

    y_true_urgency = ["routine", "soon", "urgent", "emergency"]
    y_pred_urgency = ["routine", "routine", "urgent", "emergency"]  # 3/4 = 0.75; 1 under-triage
    llm_only_urgency = ["routine", "routine", "urgent", "urgent"]  # missed emergency -> recall 0.0

    y_true_route = ["front_desk", "pharmacist_or_ma", "on_call_clinician", "on_call_clinician"]
    y_pred_route = ["front_desk", "pharmacist_or_ma", "on_call_clinician", "on_call_clinician"]  # 1.0

    used_fallback = [False, False, False, False]  # 1.0
    latencies = [0.1, 0.2, 0.3, 0.4]

    results = compute_metrics(
        y_true_intent=y_true_intent,
        y_pred_intent=y_pred_intent,
        y_true_urgency=y_true_urgency,
        y_pred_urgency=y_pred_urgency,
        y_true_route=y_true_route,
        y_pred_route=y_pred_route,
        llm_only_urgency=llm_only_urgency,
        used_fallback=used_fallback,
        latencies=latencies,
    )

    assert results["intent_accuracy"] == 0.75
    assert results["urgency_accuracy"] == 0.75
    assert results["route_accuracy"] == 1.0
    assert results["under_triage_rate"] == 0.25
    assert results["over_triage_rate"] == 0.0
    assert results["pipeline_red_flag_recall"] == 1.0
    assert results["llm_only_emergency_recall"] == 0.0
    assert results["json_validity_rate"] == 1.0
    assert results["median_latency_s"] == 0.25
    assert results["p95_latency_s"] > 0.3
