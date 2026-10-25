"""Classical ML baseline for clinic inbox triage evaluation.

Trains TF-IDF (word 1-2 grams) + LogisticRegression classifiers on the dev split
(28 messages) for intent and urgency, evaluates on the test split (12 messages),
computes evaluation metrics, and appends the result to results/summary.csv.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Add repository root to pythonpath
REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.metrics import (
    accuracy,
    emergency_recall,
    format_markdown_table,
    over_triage_rate,
    under_triage_rate,
)
from src.routing import route

DATASET_PATH: Final[Path] = REPO_ROOT / "data" / "synthetic_messages.csv"
RESULTS_DIR: Final[Path] = REPO_ROOT / "results"
SUMMARY_CSV_PATH: Final[Path] = RESULTS_DIR / "summary.csv"

SUMMARY_COLUMNS: Final[list[str]] = [
    "timestamp",
    "model",
    "prompt_version",
    "split",
    "n_samples",
    "intent_accuracy",
    "urgency_accuracy",
    "route_accuracy",
    "under_triage_rate",
    "over_triage_rate",
    "pipeline_red_flag_recall",
    "llm_only_emergency_recall",
    "json_validity_rate",
    "median_latency_s",
    "p95_latency_s",
]


def run_baseline() -> dict[str, Any]:
    """Train classical TF-IDF + LogisticRegression baseline and evaluate on test set.

    Note: The training set is only 28 messages (dev split). This serves as a classical
    machine learning baseline under a small-sample regime.

    Returns:
        Summary metrics dictionary for the baseline.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Synthetic dataset not found at: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)

    # Note: Training set is only 28 messages (dev split).
    train_df = df[df["split"] == "dev"].copy()
    test_df = df[df["split"] == "test"].copy()

    print(
        f"Training classical ML baseline on dev ({len(train_df)} messages)..."
    )

    # 1. Pipeline for Intent Classification
    intent_pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    intent_pipeline.fit(train_df["message"], train_df["intent"])

    # 2. Pipeline for Urgency Classification
    urgency_pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    urgency_pipeline.fit(train_df["message"], train_df["urgency"])

    print(f"Evaluating baseline on test split ({len(test_df)} messages)...")

    # 3. Predict on test set with latency measurement
    start_time = time.perf_counter()
    pred_intents: list[str] = intent_pipeline.predict(test_df["message"]).tolist()
    pred_urgencies: list[str] = urgency_pipeline.predict(test_df["message"]).tolist()
    total_elapsed = time.perf_counter() - start_time
    mean_latency_s = total_elapsed / len(test_df)

    # 4. Derive routes deterministically from predicted intent & urgency
    pred_routes = [
        route(intent=i, urgency=u).value
        for i, u in zip(pred_intents, pred_urgencies)
    ]

    # Save per-message predictions
    pred_df = test_df.copy()
    pred_df["predicted_intent"] = pred_intents
    pred_df["predicted_urgency"] = pred_urgencies
    pred_df["predicted_route"] = pred_routes
    pred_df["latency_s"] = mean_latency_s

    pred_csv_path = RESULTS_DIR / "predictions_tfidf_logreg_test.csv"
    pred_df.to_csv(pred_csv_path, index=False)
    print(f"Saved baseline predictions to: {pred_csv_path}")

    # 5. Compute metrics
    y_true_intent = test_df["intent"].tolist()
    y_true_urgency = test_df["urgency"].tolist()
    y_true_route = test_df["expected_route"].tolist()

    summary_metrics: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": "tfidf_logreg",
        "prompt_version": "n/a",
        "split": "test",
        "n_samples": len(test_df),
        "intent_accuracy": round(accuracy(y_true_intent, pred_intents), 4),
        "urgency_accuracy": round(accuracy(y_true_urgency, pred_urgencies), 4),
        "route_accuracy": round(accuracy(y_true_route, pred_routes), 4),
        "under_triage_rate": round(
            under_triage_rate(y_true_urgency, pred_urgencies), 4
        ),
        "over_triage_rate": round(
            over_triage_rate(y_true_urgency, pred_urgencies), 4
        ),
        "pipeline_red_flag_recall": round(
            emergency_recall(y_true_urgency, pred_urgencies), 4
        ),
        "llm_only_emergency_recall": round(
            emergency_recall(y_true_urgency, pred_urgencies), 4
        ),
        "json_validity_rate": 1.0,  # Classical ML is guaranteed valid
        "median_latency_s": round(mean_latency_s, 6),
        "p95_latency_s": round(mean_latency_s, 6),
    }

    # 6. Append to results/summary.csv
    summary_df = pd.DataFrame([summary_metrics])
    if not SUMMARY_CSV_PATH.is_file():
        summary_df.to_csv(SUMMARY_CSV_PATH, index=False, columns=SUMMARY_COLUMNS)
    else:
        summary_df.to_csv(
            SUMMARY_CSV_PATH,
            mode="a",
            header=False,
            index=False,
            columns=SUMMARY_COLUMNS,
        )

    # 7. Print markdown table
    print("\n### Classical ML Baseline Evaluation Summary (Test Split)")
    markdown_df = pd.DataFrame([summary_metrics])
    print(format_markdown_table(markdown_df))

    return summary_metrics


if __name__ == "__main__":
    run_baseline()
