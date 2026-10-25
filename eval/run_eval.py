"""Evaluation harness for clinic triage agent local LLM models and prompts.

Executes end-to-end triage pipelines against labeled synthetic evaluation datasets,
displays real-time per-message progress, enforces a 120-second client timeout,
reports model processor info (GPU/CPU), and guarantees clean Ctrl+C termination
with persistence of partial predictions.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import httpx
import ollama
import pandas as pd

# Add repository root to pythonpath
REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.metrics import (
    accuracy,
    emergency_recall,
    format_markdown_table,
    json_validity_rate,
    median_latency,
    over_triage_rate,
    percentile_95,
    under_triage_rate,
)
from src.triage import (
    DEFAULT_OLLAMA_TIMEOUT,
    call_llm_with_meta,
    triage_message_with_meta,
)

DATASET_PATH: Final[Path] = REPO_ROOT / "data" / "synthetic_messages.csv"
RESULTS_DIR: Final[Path] = REPO_ROOT / "results"
SUMMARY_CSV_PATH: Final[Path] = RESULTS_DIR / "summary.csv"

# Enforce 120-second client request timeout to prevent hanging forever
REQUEST_TIMEOUT_S: Final[float] = 120.0

if hasattr(ollama, "_client") and hasattr(ollama._client, "_client"):
    try:
        ollama._client._client.timeout = httpx.Timeout(REQUEST_TIMEOUT_S)
    except Exception:
        pass

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


def get_model_processor_info(model_name: str) -> str:
    """Query Ollama server to determine running model processor status (GPU / CPU).

    Args:
        model_name: Name of the model to inspect.

    Returns:
        Human-readable string describing processor and VRAM utilization.
    """
    try:
        ps_resp = ollama.ps()
        models = getattr(ps_resp, "models", [])
        for m in models:
            m_name = getattr(m, "name", "") or getattr(m, "model", "")
            if model_name in m_name or m_name in model_name:
                size = getattr(m, "size", 0)
                size_vram = getattr(m, "size_vram", 0)
                if size > 0:
                    gpu_pct = int(round((size_vram / size) * 100))
                    vram_gb = size_vram / (1024**3)
                    if gpu_pct == 100:
                        return f"100% GPU ({vram_gb:.1f} GB VRAM)"
                    elif gpu_pct == 0:
                        return "100% CPU"
                    else:
                        return f"{gpu_pct}% GPU offload ({vram_gb:.1f} GB VRAM)"
    except Exception:
        pass
    return "GPU/CPU status unavailable (run `ollama ps` to verify)"


def warm_up_model(model: str, prompt_version: str = "v1") -> None:
    """Perform an untimed warm-up call to load model weights into VRAM/RAM.

    Also queries processor allocation and prints a diagnostic note.

    Args:
        model: Ollama model identifier.
        prompt_version: System prompt template version.
    """
    print(f"Warming up model '{model}' with prompt '{prompt_version}'...")
    try:
        ollama.chat(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            options={"temperature": 0},
        )
    except Exception as exc:
        print(
            f"Warning: warm-up call for '{model}' encountered: {exc}",
            file=sys.stderr,
        )

    # Print model's processor allocation
    processor_info = get_model_processor_info(model)
    print(f"Model processor status: {processor_info}")
    print(
        "Note: If latency is high, run 'ollama ps' in a separate terminal to verify GPU allocation.\n",
        flush=True,
    )


def run_evaluation_for_model(
    model: str,
    prompt_version: str,
    split: str = "dev",
    limit: int | None = None,
) -> dict[str, Any] | None:
    """Execute evaluation for a single model and prompt version combination.

    Displays real-time progress per message, enforces a 120-second timeout,
    and cleanly handles Ctrl+C by writing collected partial predictions.

    Args:
        model: Ollama model identifier (e.g. 'llama3.2:3b').
        prompt_version: System prompt template version (e.g. 'v1').
        split: Dataset split ('dev', 'test', or 'all').
        limit: Optional maximum number of messages to evaluate.

    Returns:
        Summary metrics dictionary, or None if no messages were triaged.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Synthetic dataset not found at: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    if split != "all":
        df = df[df["split"] == split].copy()

    if limit is not None and limit > 0:
        df = df.iloc[:limit].copy()

    total_count = len(df)

    # 1. Warm-up call (excluded from timing)
    warm_up_model(model=model, prompt_version=prompt_version)

    predictions: list[dict[str, Any]] = []
    interrupted = False

    print(
        f"Evaluating {total_count} messages [model={model}, prompt={prompt_version}, split={split}]...",
        flush=True,
    )

    # 2. Per-message evaluation loop with Ctrl+C interception
    try:
        for idx, (_, row) in enumerate(df.iterrows()):
            msg_id = row["id"]
            message = str(row["message"])
            expected_intent = str(row["intent"])
            expected_urgency = str(row["urgency"])
            expected_route = str(row["expected_route"])
            expected_red_flag = bool(
                str(row["red_flag"]).strip().lower() in ["true", "1", "yes"]
            )

            pred_intent: str
            pred_urgency: str
            pred_route: str
            pred_red_flag: bool
            used_fallback: bool
            attempts: int
            latency_s: float

            # Run full pipeline with timeout
            try:
                record, latency_s, meta = triage_message_with_meta(
                    message=message,
                    answers=None,
                    model=model,
                    prompt_version=prompt_version,
                    timeout=REQUEST_TIMEOUT_S,
                )
                pred_intent = record.result.intent.value
                pred_urgency = record.result.urgency.value
                pred_route = record.result.route_to.value
                pred_red_flag = record.result.red_flag.is_red_flag
                used_fallback = bool(meta.get("used_fallback", False))
                attempts = int(meta.get("attempts", 1))
            except Exception as exc:
                print(
                    f"\n[ERROR] Triage failed for {msg_id}: {exc}",
                    file=sys.stderr,
                )
                pred_intent = "other"
                pred_urgency = "urgent"
                pred_route = "on_call_clinician"
                pred_red_flag = False
                used_fallback = True
                attempts = 1
                latency_s = REQUEST_TIMEOUT_S

            # LLM-only assessment: For red-flag messages, bypass rules layer to isolate LLM reasoning
            llm_only_urgency: str
            if expected_red_flag or expected_urgency == "emergency":
                try:
                    llm_out, _ = call_llm_with_meta(
                        message=message,
                        follow_up=[],
                        model=model,
                        prompt_version=prompt_version,
                        timeout=REQUEST_TIMEOUT_S,
                    )
                    llm_only_urgency = llm_out.urgency.value
                except Exception as exc:
                    print(
                        f"\n[ERROR] LLM-only call failed for {msg_id}: {exc}",
                        file=sys.stderr,
                    )
                    llm_only_urgency = "urgent"
            else:
                llm_only_urgency = pred_urgency

            predictions.append(
                {
                    "id": msg_id,
                    "model": model,
                    "prompt_version": prompt_version,
                    "split": row["split"],
                    "message": message,
                    "expected_intent": expected_intent,
                    "predicted_intent": pred_intent,
                    "expected_urgency": expected_urgency,
                    "predicted_urgency": pred_urgency,
                    "llm_only_urgency": llm_only_urgency,
                    "expected_route": expected_route,
                    "predicted_route": pred_route,
                    "expected_red_flag": expected_red_flag,
                    "predicted_red_flag": pred_red_flag,
                    "used_fallback": used_fallback,
                    "attempts": attempts,
                    "latency_s": latency_s,
                }
            )

            # Progress line printed as each message finishes
            print(
                f"[{idx + 1:2d}/{total_count}] {msg_id} | "
                f"urgency: pred='{pred_urgency}' vs exp='{expected_urgency}' | "
                f"{latency_s:.2f}s",
                flush=True,
            )

    except KeyboardInterrupt:
        print(
            f"\n\n[INTERRUPTED] Ctrl+C received. Stopping cleanly and writing {len(predictions)} partial prediction(s)...",
            file=sys.stderr,
            flush=True,
        )
        interrupted = True

    # 3. Save predictions (partial or full)
    if not predictions:
        print("No predictions were collected to save.", file=sys.stderr)
        if interrupted:
            sys.exit(0)
        return None

    pred_df = pd.DataFrame(predictions)
    safe_model = model.replace(":", "_").replace("/", "_")
    pred_csv_filename = f"predictions_{safe_model}_{prompt_version}_{split}.csv"
    pred_csv_path = RESULTS_DIR / pred_csv_filename
    pred_df.to_csv(pred_csv_path, index=False)
    print(f"\nSaved {len(pred_df)} prediction(s) to: {pred_csv_path}")

    # 4. Compute evaluation metrics
    y_true_intent = pred_df["expected_intent"].tolist()
    y_pred_intent = pred_df["predicted_intent"].tolist()

    y_true_urgency = pred_df["expected_urgency"].tolist()
    y_pred_urgency = pred_df["predicted_urgency"].tolist()
    llm_only_urgency_list = pred_df["llm_only_urgency"].tolist()

    y_true_route = pred_df["expected_route"].tolist()
    y_pred_route = pred_df["predicted_route"].tolist()

    fallbacks = pred_df["used_fallback"].tolist()
    latencies = pred_df["latency_s"].tolist()

    summary_metrics: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "prompt_version": prompt_version,
        "split": split,
        "n_samples": len(pred_df),
        "intent_accuracy": round(accuracy(y_true_intent, y_pred_intent), 4),
        "urgency_accuracy": round(accuracy(y_true_urgency, y_pred_urgency), 4),
        "route_accuracy": round(accuracy(y_true_route, y_pred_route), 4),
        "under_triage_rate": round(
            under_triage_rate(y_true_urgency, y_pred_urgency), 4
        ),
        "over_triage_rate": round(
            over_triage_rate(y_true_urgency, y_pred_urgency), 4
        ),
        "pipeline_red_flag_recall": round(
            emergency_recall(y_true_urgency, y_pred_urgency), 4
        ),
        "llm_only_emergency_recall": round(
            emergency_recall(y_true_urgency, llm_only_urgency_list), 4
        ),
        "json_validity_rate": round(json_validity_rate(fallbacks), 4),
        "median_latency_s": round(median_latency(latencies), 4),
        "p95_latency_s": round(percentile_95(latencies), 4),
    }

    # 5. Append summary row to results/summary.csv
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

    # 6. Print Markdown summary table
    status_label = " (Partial - Interrupted by User)" if interrupted else ""
    print(f"\n### Evaluation Summary{status_label}")
    markdown_df = pd.DataFrame([summary_metrics])
    print(format_markdown_table(markdown_df))

    if interrupted:
        sys.exit(0)

    return summary_metrics


def main() -> None:
    """CLI entrypoint for running evaluation harness."""
    parser = argparse.ArgumentParser(
        description="Run LLM clinic inbox triage evaluation harness."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["llama3.2:3b"],
        help="Space-separated list of Ollama model identifiers (default: llama3.2:3b)",
    )
    parser.add_argument(
        "--prompt-versions",
        nargs="+",
        default=["v1"],
        help="Space-separated list of prompt versions (default: v1)",
    )
    parser.add_argument(
        "--split",
        choices=["dev", "test", "all"],
        default="dev",
        help="Dataset split to evaluate: dev, test, or all (default: dev)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of messages to evaluate (for quick runs)",
    )

    args = parser.parse_args()

    try:
        for model in args.models:
            for version in args.prompt_versions:
                run_evaluation_for_model(
                    model=model,
                    prompt_version=version,
                    split=args.split,
                    limit=args.limit,
                )
    except KeyboardInterrupt:
        print("\nEvaluation halted by user.", file=sys.stderr)
        sys.exit(0)


if __name__ == "__main__":
    main()
