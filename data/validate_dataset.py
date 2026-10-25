"""Validation script for the synthetic patient triage evaluation dataset.

Loads data/synthetic_messages.csv with pandas and verifies:
- Exactly 40 rows and correct schema columns
- Split distribution (28 dev, 12 test)
- All enum values conform to Intent, Urgency, and RouteTo
- At least 4 messages per intent
- Red-flag criteria (exactly 8 emergency red flags)
- Route consistency with src.routing.route()
- Prints summary distributions
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


def validate_dataset(csv_path: Path | str = DATASET_PATH) -> pd.DataFrame:
    """Validate data integrity, schema consistency, and clinical routing rules.

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

    # 1. Column headers
    assert list(df.columns) == EXPECTED_COLUMNS, (
        f"Columns mismatch: expected {EXPECTED_COLUMNS}, got {list(df.columns)}"
    )

    # 2. Total row count
    assert len(df) == EXPECTED_ROW_COUNT, (
        f"Row count mismatch: expected {EXPECTED_ROW_COUNT}, got {len(df)}"
    )

    # 3. Unique IDs and no missing text
    assert df["id"].is_unique, "Duplicate IDs found in dataset."
    assert df["message"].notna().all(), "Found null or empty messages."
    assert (df["message"].str.strip().str.len() > 0).all(), "Found empty messages."

    # 4. Split counts
    split_counts = df["split"].value_counts().to_dict()
    assert split_counts.get("dev") == EXPECTED_DEV_COUNT, (
        f"Dev count mismatch: expected {EXPECTED_DEV_COUNT}, got {split_counts.get('dev')}"
    )
    assert split_counts.get("test") == EXPECTED_TEST_COUNT, (
        f"Test count mismatch: expected {EXPECTED_TEST_COUNT}, got {split_counts.get('test')}"
    )
    assert set(df["split"].unique()) == {"dev", "test"}, (
        f"Unexpected splits found: {set(df['split'].unique())}"
    )

    # 5. Enum values
    valid_intents = {e.value for e in Intent}
    valid_urgencies = {e.value for e in Urgency}
    valid_routes = {e.value for e in RouteTo}

    invalid_intents = set(df["intent"].unique()) - valid_intents
    assert not invalid_intents, f"Invalid intent values found: {invalid_intents}"

    invalid_urgencies = set(df["urgency"].unique()) - valid_urgencies
    assert not invalid_urgencies, f"Invalid urgency values found: {invalid_urgencies}"

    invalid_routes = set(df["expected_route"].unique()) - valid_routes
    assert not invalid_routes, f"Invalid route values found: {invalid_routes}"

    # 6. Intent representation (at least 4 messages per intent)
    intent_counts = df["intent"].value_counts().to_dict()
    for intent in valid_intents:
        count = intent_counts.get(intent, 0)
        assert count >= MIN_MESSAGES_PER_INTENT, (
            f"Intent '{intent}' has only {count} messages, expected >= {MIN_MESSAGES_PER_INTENT}"
        )

    # Every intent and urgency present in both splits
    for split_name in ["dev", "test"]:
        sub_df = df[df["split"] == split_name]
        missing_intents = valid_intents - set(sub_df["intent"].unique())
        assert not missing_intents, (
            f"Split '{split_name}' is missing intents: {missing_intents}"
        )
        missing_urgencies = valid_urgencies - set(sub_df["urgency"].unique())
        assert not missing_urgencies, (
            f"Split '{split_name}' is missing urgencies: {missing_urgencies}"
        )

    # 7. Red flag verification
    # Convert red_flag column to standard boolean if string
    norm_red_flag = df["red_flag"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
    red_flag_count = norm_red_flag.sum()
    assert red_flag_count == EXPECTED_RED_FLAG_COUNT, (
        f"Red flag count mismatch: expected {EXPECTED_RED_FLAG_COUNT}, got {red_flag_count}"
    )

    # All red-flag messages must have urgency="emergency"
    emergency_mismatches = df[norm_red_flag & (df["urgency"] != "emergency")]
    assert len(emergency_mismatches) == 0, (
        f"Found red flags with non-emergency urgency: {emergency_mismatches[['id', 'urgency', 'message']]}"
    )

    # 8. Expected route consistency with src/routing.py
    for idx, row in df.iterrows():
        derived_route = route(intent=row["intent"], urgency=row["urgency"]).value
        assert derived_route == row["expected_route"], (
            f"Row {row['id']} routing mismatch: expected_route is '{row['expected_route']}', "
            f"but src.routing.route('{row['intent']}', '{row['urgency']}') gave '{derived_route}'"
        )

    return df


def print_distribution(df: pd.DataFrame) -> None:
    """Print readable dataset distribution tables."""
    print("=" * 60)
    print("CLINIC INBOX TRIAGE EVALUATION DATASET SUMMARY")
    print("=" * 60)
    print(f"Total Messages: {len(df)}")
    print(f"Splits: dev={len(df[df['split'] == 'dev'])}, test={len(df[df['split'] == 'test'])}")
    print(f"Red flags: {df['red_flag'].astype(str).str.lower().isin(['true', '1']).sum()} true, "
          f"{(~df['red_flag'].astype(str).str.lower().isin(['true', '1'])).sum()} false")

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
    validated_df = validate_dataset()
    print_distribution(validated_df)
