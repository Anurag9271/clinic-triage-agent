"""Evaluation metrics module for clinic message triage.

Computes accuracy, clinical safety rates (under-triage, over-triage),
emergency red-flag recall, JSON schema validity rate, and latency percentiles.
Can be tested independently without model or Ollama dependencies.
"""

from __future__ import annotations

from typing import Any, Final, Sequence

import numpy as np

# Clinical urgency hierarchy for triage prioritization
URGENCY_RANKS: Final[dict[str, int]] = {
    "routine": 0,
    "soon": 1,
    "urgent": 2,
    "emergency": 3,
}


def accuracy(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    """Compute overall classification accuracy.

    Args:
        y_true: True ground truth class labels.
        y_pred: Predicted class labels.

    Returns:
        Fraction of correctly matched predictions in [0.0, 1.0]. Returns 0.0 if empty.
    """
    if len(y_true) == 0:
        return 0.0
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)}, y_pred has {len(y_pred)}"
        )
    return float(sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true))


def under_triage_rate(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Compute the under-triage rate based on urgency hierarchy.

    An under-triage occurs whenever predicted urgency rank is lower than
    the ground truth urgency rank (e.g. predicting routine when urgency is urgent).
    Ordering: routine (0) < soon (1) < urgent (2) < emergency (3).

    Args:
        y_true: Ground truth urgency labels.
        y_pred: Predicted urgency labels.

    Returns:
        Fraction of under-triaged messages in [0.0, 1.0]. Returns 0.0 if empty.
    """
    if len(y_true) == 0:
        return 0.0
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)}, y_pred has {len(y_pred)}"
        )

    under_count = 0
    for t, p in zip(y_true, y_pred):
        true_rank = URGENCY_RANKS.get(str(t).lower().strip())
        pred_rank = URGENCY_RANKS.get(str(p).lower().strip())
        if true_rank is None or pred_rank is None:
            raise ValueError(f"Unrecognized urgency label in under_triage calculation: {t}, {p}")
        if pred_rank < true_rank:
            under_count += 1

    return float(under_count / len(y_true))


def over_triage_rate(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Compute the over-triage rate based on urgency hierarchy.

    An over-triage occurs whenever predicted urgency rank is higher than
    the ground truth urgency rank (e.g. predicting urgent when urgency is routine).
    Ordering: routine (0) < soon (1) < urgent (2) < emergency (3).

    Args:
        y_true: Ground truth urgency labels.
        y_pred: Predicted urgency labels.

    Returns:
        Fraction of over-triaged messages in [0.0, 1.0]. Returns 0.0 if empty.
    """
    if len(y_true) == 0:
        return 0.0
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)}, y_pred has {len(y_pred)}"
        )

    over_count = 0
    for t, p in zip(y_true, y_pred):
        true_rank = URGENCY_RANKS.get(str(t).lower().strip())
        pred_rank = URGENCY_RANKS.get(str(p).lower().strip())
        if true_rank is None or pred_rank is None:
            raise ValueError(f"Unrecognized urgency label in over_triage calculation: {t}, {p}")
        if pred_rank > true_rank:
            over_count += 1

    return float(over_count / len(y_true))


def emergency_recall(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Compute recall for emergency conditions (expected emergencies predicted as emergency).

    Args:
        y_true: Ground truth urgency labels.
        y_pred: Predicted urgency labels.

    Returns:
        Recall rate in [0.0, 1.0]. Returns 1.0 if there are no ground-truth emergencies,
        or 0.0 if inputs are empty.
    """
    if len(y_true) == 0:
        return 0.0
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)}, y_pred has {len(y_pred)}"
        )

    emergency_preds = [
        str(p).lower().strip()
        for t, p in zip(y_true, y_pred)
        if str(t).lower().strip() == "emergency"
    ]
    if len(emergency_preds) == 0:
        return 1.0

    caught = sum(1 for p in emergency_preds if p == "emergency")
    return float(caught / len(emergency_preds))


def json_validity_rate(used_fallback: Sequence[bool]) -> float:
    """Compute the rate of successful schema adherence without fallback invocation.

    Args:
        used_fallback: Sequence of booleans where True indicates fallback output was used.

    Returns:
        Fraction of responses that conformed to schema (1.0 - fallback_rate).
        Returns 0.0 if empty.
    """
    if len(used_fallback) == 0:
        return 0.0
    fallback_count = sum(1 for f in used_fallback if bool(f))
    return float(1.0 - (fallback_count / len(used_fallback)))


def percentile_95(latencies: Sequence[float]) -> float:
    """Compute the 95th percentile latency in seconds.

    Args:
        latencies: Sequence of elapsed times in seconds.

    Returns:
        95th percentile value as a float. Returns 0.0 if empty.
    """
    if len(latencies) == 0:
        return 0.0
    return float(np.percentile(latencies, 95))


def median_latency(latencies: Sequence[float]) -> float:
    """Compute the median (50th percentile) latency in seconds.

    Args:
        latencies: Sequence of elapsed times in seconds.

    Returns:
        Median value as a float. Returns 0.0 if empty.
    """
    if len(latencies) == 0:
        return 0.0
    return float(np.percentile(latencies, 50))


def compute_metrics(
    y_true_intent: Sequence[str],
    y_pred_intent: Sequence[str],
    y_true_urgency: Sequence[str],
    y_pred_urgency: Sequence[str],
    y_true_route: Sequence[str],
    y_pred_route: Sequence[str],
    llm_only_urgency: Sequence[str],
    used_fallback: Sequence[bool],
    latencies: Sequence[float],
) -> dict[str, float]:
    """Compute consolidated dictionary of triage evaluation metrics.

    Returns:
        Dictionary of computed metric names and float scores.
    """
    return {
        "intent_accuracy": accuracy(y_true_intent, y_pred_intent),
        "urgency_accuracy": accuracy(y_true_urgency, y_pred_urgency),
        "route_accuracy": accuracy(y_true_route, y_pred_route),
        "under_triage_rate": under_triage_rate(y_true_urgency, y_pred_urgency),
        "over_triage_rate": over_triage_rate(y_true_urgency, y_pred_urgency),
        "pipeline_red_flag_recall": emergency_recall(y_true_urgency, y_pred_urgency),
        "llm_only_emergency_recall": emergency_recall(y_true_urgency, llm_only_urgency),
        "json_validity_rate": json_validity_rate(used_fallback),
        "median_latency_s": median_latency(latencies),
        "p95_latency_s": percentile_95(latencies),
    }


def format_markdown_table(df: Any) -> str:
    """Format a pandas DataFrame as a GitHub-flavored Markdown table without tabulate.

    Args:
        df: pandas DataFrame or dictionary-like structure.

    Returns:
        Formatted markdown table string.
    """
    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in df.iterrows():
        row_str = [str(row[col]) for col in df.columns]
        lines.append("| " + " | ".join(row_str) + " |")
    return "\n".join(lines)


__all__ = [
    "URGENCY_RANKS",
    "accuracy",
    "compute_metrics",
    "emergency_recall",
    "format_markdown_table",
    "json_validity_rate",
    "median_latency",
    "over_triage_rate",
    "percentile_95",
    "under_triage_rate",
]
