"""Validation module for the synthetic patient triage evaluation dataset.

Provides structural, schema, and clinical routing invariant validation for
data/synthetic_messages.csv. Can be used as a standalone CLI script or
imported as a testable validation function.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

import pandas as pd

# Add repository root to pythonpath if executing directly
REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.routing import route
from src.schema import Intent, RouteTo, Urgency

DATASET_PATH: Final[Path] = REPO_ROOT / "data" / "synthetic_messages.csv"
EXPECTED_COLUMNS: Final[list[str]] = [
    "id",
    "split",
    "message",
    "intent",
    "urgency",
    "red_flag",
    "expected_route",
    "notes",
]
EXPECTED_ROW_COUNT: Final[int] = 40
EXPECTED_DEV_COUNT: Final[int] = 28
EXPECTED_TEST_COUNT: Final[int] = 12
EXPECTED_RED_FLAG_COUNT: Final[int] = 8
MIN_MESSAGES_PER_INTENT: Final[int] = 4


def validate(df: pd.DataFrame) -> list[str]:
    """Validate data integrity, schema consistency, and clinical routing rules.

    Args:
        df: Input DataFrame containing synthetic triage messages.

    Returns:
        List of human-readable error descriptions. Returns an empty list if valid.
    """
    errors: list[str] = []

    # 1. Column headers
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
    extra_cols = [col for col in df.columns if col not in EXPECTED_COLUMNS]
    if extra_cols:
        errors.append(f"Unexpected extra columns: {extra_cols}")

    # 2. Total row count
    if len(df) != EXPECTED_ROW_COUNT:
        errors.append(
            f"Row count mismatch: expected {EXPECTED_ROW_COUNT}, got {len(df)}"
        )

    # 3. Unique IDs and non-empty messages
    if "id" in df.columns:
        duplicate_ids = df[df["id"].duplicated()]["id"].tolist()
        if duplicate_ids:
            errors.append(f"Duplicate IDs found in dataset: {duplicate_ids}")

    if "message" in df.columns:
        if df["message"].isna().any():
            errors.append("Dataset contains null message values.")
        elif (df["message"].astype(str).str.strip().str.len() == 0).any():
            errors.append("Dataset contains empty message strings.")

    # 4. Split counts and names
    if "split" in df.columns:
        split_counts = df["split"].value_counts().to_dict()
        dev_count = split_counts.get("dev", 0)
        test_count = split_counts.get("test", 0)
        if dev_count != EXPECTED_DEV_COUNT:
            errors.append(
                f"Split 'dev' count mismatch: expected {EXPECTED_DEV_COUNT}, got {dev_count}"
            )
        if test_count != EXPECTED_TEST_COUNT:
            errors.append(
                f"Split 'test' count mismatch: expected {EXPECTED_TEST_COUNT}, got {test_count}"
            )
        invalid_splits = set(df["split"].unique()) - {"dev", "test"}
        if invalid_splits:
            errors.append(f"Invalid split names found: {sorted(invalid_splits)}")

    # 5. Enum values
    valid_intents = {e.value for e in Intent}
    valid_urgencies = {e.value for e in Urgency}
    valid_routes = {e.value for e in RouteTo}

    if "intent" in df.columns:
        invalid_intents = set(df["intent"].unique()) - valid_intents
        if invalid_intents:
            errors.append(f"Invalid intent values found: {sorted(invalid_intents)}")

    if "urgency" in df.columns:
        invalid_urgencies = set(df["urgency"].unique()) - valid_urgencies
        if invalid_urgencies:
            errors.append(f"Invalid urgency values found: {sorted(invalid_urgencies)}")

    if "expected_route" in df.columns:
        invalid_routes = set(df["expected_route"].unique()) - valid_routes
        if invalid_routes:
            errors.append(
                f"Invalid expected_route values found: {sorted(invalid_routes)}"
            )

    # 6. Intent representation (at least 4 messages per intent across dataset)
    if "intent" in df.columns:
        intent_counts = df["intent"].value_counts().to_dict()
        for intent in valid_intents:
            count = intent_counts.get(intent, 0)
            if count < MIN_MESSAGES_PER_INTENT:
                errors.append(
                    f"Intent '{intent}' has only {count} messages (minimum {MIN_MESSAGES_PER_INTENT} required)."
                )

    # Intent and urgency representation across splits
    if "split" in df.columns:
        for split_name in ["dev", "test"]:
            sub_df = df[df["split"] == split_name]
            if "intent" in df.columns:
                missing_intents = valid_intents - set(sub_df["intent"].unique())
                if missing_intents:
                    errors.append(
                        f"Split '{split_name}' is missing intents: {sorted(missing_intents)}"
                    )
            if "urgency" in df.columns:
                missing_urgencies = valid_urgencies - set(sub_df["urgency"].unique())
                if missing_urgencies:
                    errors.append(
                        f"Split '{split_name}' is missing urgencies: {sorted(missing_urgencies)}"
                    )

    # 7. Red flag verification
    if "red_flag" in df.columns:
        norm_red_flag = (
            df["red_flag"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["true", "1", "yes"])
        )
        red_flag_count = norm_red_flag.sum()
        if red_flag_count != EXPECTED_RED_FLAG_COUNT:
            errors.append(
                f"Red flag count mismatch: expected {EXPECTED_RED_FLAG_COUNT}, got {red_flag_count}"
            )

        if "urgency" in df.columns:
            emergency_mismatches = df[norm_red_flag & (df["urgency"] != "emergency")]
            if len(emergency_mismatches) > 0:
                errors.append(
                    f"Red flags with non-emergency urgency: {emergency_mismatches['id'].tolist()}"
                )

    # 8. Expected route consistency with src/routing.py
    if (
        "intent" in df.columns
        and "urgency" in df.columns
        and "expected_route" in df.columns
    ):
        for idx, row in df.iterrows():
            row_intent = row["intent"]
            row_urgency = row["urgency"]
            row_route = row["expected_route"]
            row_id = row.get("id", f"row_{idx}")
            if row_intent in valid_intents and row_urgency in valid_urgencies:
                try:
                    derived = route(intent=row_intent, urgency=row_urgency).value
                    if derived != row_route:
                        errors.append(
                            f"Row {row_id} routing mismatch: expected_route is '{row_route}', "
                            f"but src.routing.route('{row_intent}', '{row_urgency}') gave '{derived}'"
                        )
                except Exception as exc:
                    errors.append(f"Row {row_id} routing evaluation raised exception: {exc}")

    return errors


def validate_dataset(csv_path: Path | str = DATASET_PATH) -> pd.DataFrame:
    """Load and validate dataset from CSV, raising AssertionError on errors.

    Args:
        csv_path: Path to the synthetic dataset CSV file.

    Returns:
        The validated pandas DataFrame.

    Raises:
        AssertionError: If any data validation check fails.
        FileNotFoundError: If the CSV file does not exist.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset CSV not found at: {path}")

    df = pd.read_csv(path)
    errors = validate(df)
    if errors:
        raise AssertionError(
            f"Dataset validation failed with {len(errors)} error(s):\n"
            + "\n".join(f"- {e}" for e in errors)
        )

    return df


def print_distribution(df: pd.DataFrame) -> None:
    """Print readable dataset distribution tables."""
    print("=" * 60)
    print("CLINIC INBOX TRIAGE EVALUATION DATASET SUMMARY")
    print("=" * 60)
    print(f"Total Messages: {len(df)}")
    print(
        f"Splits: dev={len(df[df['split'] == 'dev'])}, test={len(df[df['split'] == 'test'])}"
    )
    rf_count = df["red_flag"].astype(str).str.lower().isin(["true", "1", "yes"]).sum()
    print(f"Red flags: {rf_count} true, {len(df) - rf_count} false")

    print("\n--- INTENT DISTRIBUTION ---")
    intent_split = pd.crosstab(df["intent"], df["split"], margins=True)
    print(intent_split)

    print("\n--- URGENCY DISTRIBUTION ---")
    urgency_split = pd.crosstab(df["urgency"], df["split"], margins=True)
    print(urgency_split)

    print("\n--- EXPECTED ROUTE DISTRIBUTION ---")
    route_split = pd.crosstab(df["expected_route"], df["split"], margins=True)
    print(route_split)

    print("\n--- INTENT x URGENCY CROSS-TABULATION ---")
    cross = pd.crosstab(df["intent"], df["urgency"], margins=True)
    print(cross)
    print("=" * 60)
    print("ALL VALIDATION CHECKS PASSED SUCCESSFULLY (OK)")
    print("=" * 60)


if __name__ == "__main__":
    if not DATASET_PATH.is_file():
        print(f"Error: Dataset CSV not found at {DATASET_PATH}", file=sys.stderr)
        sys.exit(1)

    loaded_df = pd.read_csv(DATASET_PATH)
    validation_errors = validate(loaded_df)
    if validation_errors:
        print(
            f"Validation failed with {len(validation_errors)} error(s):",
            file=sys.stderr,
        )
        for err in validation_errors:
            print(f"  - {err}", file=sys.stderr)
        sys.exit(1)

    print_distribution(loaded_df)
    sys.exit(0)
