"""Unit tests for synthetic dataset validation and clinical consistency.

Verifies data/synthetic_messages.csv schema integrity, uniqueness,
clinical red-flag alignment, staff routing consistency, contact formatting,
and that validate() correctly flags broken DataFrames.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from data.validate_dataset import DATASET_PATH, validate
from src.red_flags import check_red_flags
from src.routing import route


@pytest.fixture(scope="module")
def dataset_df() -> pd.DataFrame:
    """Fixture providing the loaded synthetic messages DataFrame."""
    assert DATASET_PATH.is_file(), f"Dataset not found at {DATASET_PATH}"
    return pd.read_csv(DATASET_PATH)


def test_validate_returns_no_errors(dataset_df: pd.DataFrame):
    """Test that validate() passes with zero errors on the synthetic dataset."""
    errors = validate(dataset_df)
    assert errors == [], f"Validation errors found: {errors}"


def test_ids_and_messages_are_unique(dataset_df: pd.DataFrame):
    """Test that all message IDs and message contents are unique."""
    assert dataset_df["id"].is_unique, "Duplicate message IDs found."
    assert dataset_df["message"].is_unique, "Duplicate message contents found."


def test_expected_route_matches_src_routing(dataset_df: pd.DataFrame):
    """Test that expected_route strictly equals route(intent, urgency) for every row."""
    for idx, row in dataset_df.iterrows():
        derived_route = route(intent=row["intent"], urgency=row["urgency"]).value
        assert row["expected_route"] == derived_route, (
            f"Row {row['id']} route mismatch: expected_route is '{row['expected_route']}', "
            f"but src.routing.route('{row['intent']}', '{row['urgency']}') gave '{derived_route}'"
        )


def test_red_flag_matches_emergency_urgency(dataset_df: pd.DataFrame):
    """Test that red_flag is True exactly when urgency is 'emergency'."""
    norm_red_flag = (
        dataset_df["red_flag"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )
    is_emergency = dataset_df["urgency"] == "emergency"
    assert (norm_red_flag == is_emergency).all(), (
        "red_flag boolean does not match urgency == 'emergency' exactly."
    )


def test_check_red_flags_triggers_for_all_red_flags(dataset_df: pd.DataFrame):
    """Test that check_red_flags(message).is_red_flag is True for every red_flag row."""
    red_flag_rows = dataset_df[
        dataset_df["red_flag"].astype(str).str.lower().isin(["true", "1", "yes"])
    ]
    assert len(red_flag_rows) == 8, f"Expected 8 red flags, found {len(red_flag_rows)}"

    for _, row in red_flag_rows.iterrows():
        result = check_red_flags(row["message"])
        assert result.is_red_flag is True, (
            f"Row {row['id']} failed check_red_flags detection: '{row['message']}'"
        )


def test_phone_and_email_patterns(dataset_df: pd.DataFrame):
    """Test that all phone numbers match 555-01xx and all emails end with @example.com."""
    phone_pattern = re.compile(r"\b(?:\d{3}-)?\d{3}-\d{4}\b")
    email_pattern = re.compile(r"[\w.+-]+@[\w.-]+")

    phones_found: list[str] = []
    emails_found: list[str] = []

    for msg in dataset_df["message"]:
        phones_found.extend(phone_pattern.findall(msg))
        emails_found.extend(email_pattern.findall(msg))

    assert len(phones_found) > 0, "Expected at least one phone number in dataset."
    for phone in phones_found:
        assert re.match(r"^555-01\d{2}$", phone), (
            f"Phone '{phone}' does not match expected 555-01xx pattern."
        )

    assert len(emails_found) > 0, "Expected at least one email address in dataset."
    for email in emails_found:
        clean_email = email.rstrip(".,;!?")
        assert clean_email.endswith("@example.com"), (
            f"Email '{email}' does not end with @example.com."
        )


def test_split_row_counts(dataset_df: pd.DataFrame):
    """Test that dev split has exactly 28 rows and test split has 12 rows."""
    dev_count = len(dataset_df[dataset_df["split"] == "dev"])
    test_count = len(dataset_df[dataset_df["split"] == "test"])
    assert dev_count == 28, f"Expected 28 dev rows, got {dev_count}"
    assert test_count == 12, f"Expected 12 test rows, got {test_count}"
    assert len(dataset_df) == 40, f"Expected 40 total rows, got {len(dataset_df)}"


def test_validate_reports_errors_for_broken_dataframe(dataset_df: pd.DataFrame):
    """Test that validate() reports errors for invalid urgency, wrong route, and duplicate id."""
    broken_df = dataset_df.copy()

    # 1. Invalid urgency value
    broken_df.loc[0, "urgency"] = "critical"

    # 2. Wrong route (urgency emergency requires on_call_clinician)
    # syn_006 is emergency, change its expected_route to front_desk
    broken_df.loc[5, "expected_route"] = "front_desk"

    # 3. Duplicate ID
    broken_df.loc[2, "id"] = broken_df.loc[0, "id"]

    errors = validate(broken_df)
    assert len(errors) > 0, "Expected validate() to report errors on broken DataFrame."

    error_text = " ".join(errors).lower()
    assert "critical" in error_text or "urgency" in error_text, (
        f"Missing invalid urgency error in: {errors}"
    )
    assert "routing mismatch" in error_text or "expected_route" in error_text, (
        f"Missing routing mismatch error in: {errors}"
    )
    assert "duplicate id" in error_text, (
        f"Missing duplicate ID error in: {errors}"
    )
