"""Automated headless tests for app.py using streamlit.testing.v1.AppTest.

Covers:
- MODELS constant definitions and default model availability
- SAMPLE_MESSAGES presence (including emergency, refill, and two-symptom messages)
- ROLE_LABELS mapping
- a. App loads without exceptions and shows the disclaimer
- b. Emergency message ("I have chest pain and can't breathe") produces emergency banner without LLM call
- c. Routine message ("I have a cold since the evening") calls patched LLM once and shows summary & reply
- d. Message triggering follow-up questions (sore throat and cough) displays questions with red-flag ones first,
     and answering "Yes" to a red-flag question produces the emergency banner
- e. Saved record appears in the staff queue and approving it removes it from the pending list
- f. When the patched LLM raises an exception, the app shows a friendly error and does not crash
All app tests run against an isolated temporary SQLite database.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

from app import MODELS, ROLE_LABELS, SAMPLE_MESSAGES
from src.db import get_record, init_db, list_records, save_record
from src.intake_questions import get_follow_up_questions
from src.red_flags import check_red_flags
from src.schema import (
    Contact,
    Intent,
    LLMTriageOutput,
    RecordStatus,
    RedFlag,
    RouteTo,
    TriageRecord,
    TriageResult,
    Urgency,
)

APP_PATH: Final[str] = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture(autouse=True)
def temp_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Ensure all tests in this module execute against an isolated temporary SQLite database."""
    db_file = tmp_path / "test_app_triage.db"
    monkeypatch.setenv("TRIAGE_DB_PATH", str(db_file))
    monkeypatch.setenv("CLINIC_DB_PATH", str(db_file))
    monkeypatch.setattr("src.db.DEFAULT_DB_PATH", str(db_file))
    init_db(db_file)
    return db_file


# ============================================================================
# Static Configuration & Constants Tests
# ============================================================================


def test_models_list():
    """MODELS constant must contain llama3.2:3b as primary option."""
    assert isinstance(MODELS, list)
    assert len(MODELS) >= 3
    assert MODELS[0] == "llama3.2:3b"


def test_sample_messages_count_and_types():
    """SAMPLE_MESSAGES must include 6 sample messages with required clinical scenarios."""
    assert len(SAMPLE_MESSAGES) == 6

    # 1. Emergency message triggers rule-based red flag
    emergency_samples = [
        msg for title, msg in SAMPLE_MESSAGES.items() if "emergency" in title.lower()
    ]
    assert len(emergency_samples) == 1
    assert check_red_flags(emergency_samples[0]).is_red_flag is True

    # 2. Refill message present
    refill_samples = [
        msg for title, msg in SAMPLE_MESSAGES.items() if "refill" in title.lower()
    ]
    assert len(refill_samples) == 1
    assert "refill" in refill_samples[0].lower() or "renew" in refill_samples[0].lower()

    # 3. Two-symptom message triggers multiple adaptive questions
    two_symptom_samples = [
        msg for title, msg in SAMPLE_MESSAGES.items() if "two symptoms" in title.lower()
    ]
    assert len(two_symptom_samples) == 1
    qs = get_follow_up_questions(two_symptom_samples[0])
    assert len(qs) >= 2


def test_role_labels():
    """ROLE_LABELS maps all expected clinic queue roles."""
    expected_keys = {"all", "on_call_clinician", "pharmacist_or_ma", "front_desk"}
    assert set(ROLE_LABELS.keys()) == expected_keys


# ============================================================================
# AppTest Headless Integration Tests
# ============================================================================


def test_app_loads_and_shows_disclaimer():
    """Requirement 3a: App loads without exceptions and shows the disclaimer."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    assert not at.exception
    disclaimer_text = "Prototype. Synthetic data only. Not a medical device and not medical advice."
    all_markdown = " ".join(
        [m.value for m in at.sidebar.markdown] + [m.value for m in at.markdown]
    )
    assert disclaimer_text in all_markdown


def test_emergency_message_produces_banner_without_llm_call():
    """Requirement 3b: An emergency message produces the emergency banner and LLM/mock is NOT called."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception

    with patch("src.triage.call_llm_with_meta") as mock_llm, patch(
        "src.mock_llm.mock_call_llm"
    ) as mock_fast_llm:
        # Input acute emergency message
        at.text_area(key="msg_text_area").input("I have chest pain and can't breathe").run()
        at.button(key="btn_next").click().run()

        # Neither real LLM nor mock function should be called for emergency red flags
        assert mock_llm.call_count == 0
        assert mock_fast_llm.call_count == 0

        assert not at.exception
        error_texts = [e.value for e in at.error]
        assert any("EMERGENCY" in err for err in error_texts)
        assert any("nearest emergency" in err.lower() or "911" in err for err in error_texts)


def test_routine_message_calls_patched_llm_once_and_shows_summary():
    """Requirement 3c: Routine message calls patched LLM once and shows summary and draft reply."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception

    mock_output = LLMTriageOutput(
        intent=Intent.GENERAL_QUERY,
        urgency=Urgency.ROUTINE,
        suggested_specialty=None,
        contact=Contact(name="Alice"),
        summary="Patient Alice reports mild cold symptoms starting this evening.",
        draft_reply="Hello Alice, for common cold symptoms stay hydrated and get plenty of rest.",
    )

    with patch("src.triage.call_llm_with_meta") as mock_llm:
        mock_llm.return_value = (mock_output, {"attempts": 1, "latency_s": 0.35})

        at.text_area(key="msg_text_area").input("I have a cold since the evening").run()
        at.button(key="btn_next").click().run()

        assert not at.exception
        assert mock_llm.call_count == 1

        # Verify clinical summary and draft reply are visible in UI output
        all_markdowns = " ".join([m.value for m in at.markdown])
        all_infos = " ".join([i.value for i in at.info])
        assert "Patient Alice reports mild cold symptoms" in all_markdowns
        assert "Hello Alice, for common cold symptoms stay hydrated" in all_infos


def test_sore_throat_and_cough_follow_up_and_red_flag_answer():
    """Requirement 3d: Message triggering follow-up questions displays red-flag questions first,

    and answering 'Yes' to a red-flag question produces the emergency banner.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception

    # Message with sore throat and cough triggers intake screening questions
    at.text_area(key="msg_text_area").input("I have a sore throat and cough").run()
    at.button(key="btn_next").click().run()

    assert not at.exception
    # Step 2: Adaptive follow-up questions displayed
    radios = list(at.radio)
    assert 1 <= len(radios) <= 3

    # Answer "Yes" to the first red-flag screening question
    radios[0].set_value("Yes").run()

    # Submit for triage
    at.button(key="btn_submit_triage").click().run()

    assert not at.exception
    # Emergency banner should trigger due to the affirmative red-flag intake answer
    error_texts = [e.value for e in at.error]
    assert any("EMERGENCY" in err for err in error_texts)


def test_saved_record_appears_in_staff_queue_and_approval_removes_it(temp_database: Path):
    """Requirement 3e: Saved record appears in staff queue and approving it removes it from pending."""
    # Pre-populate temp database with a pending triage record
    sample_result = TriageResult(
        intent=Intent.REFILL,
        urgency=Urgency.ROUTINE,
        route_to=RouteTo.PHARMACIST_OR_MA,
        suggested_specialty=None,
        red_flag=RedFlag(is_red_flag=False, reason=None),
        follow_up=[],
        contact=Contact(name="Arthur Davis", phone="555-0144"),
        summary="Patient Arthur Davis requests a prescription refill.",
        draft_reply="Hello Arthur, your refill has been approved and forwarded to the pharmacy.",
    )
    rec = TriageRecord(
        original_message="Need refill of metformin 500mg.",
        result=sample_result,
        status=RecordStatus.PENDING,
    )
    save_record(rec, "mock", "v1", 0.1, db_path=temp_database)

    # Initial assertion: record is pending in DB
    pending_before = list_records(status="pending", db_path=temp_database)
    assert len(pending_before) == 1
    assert pending_before[0].id == rec.id

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception

    # Find the Approve button for this record and click it
    btn_approve = at.button(key=f"btn_app_{rec.id}")
    assert btn_approve is not None
    btn_approve.click().run()

    assert not at.exception

    # Post-approval assertion: pending list in DB is now empty
    pending_after = list_records(status="pending", db_path=temp_database)
    assert len(pending_after) == 0

    # Record status is now approved
    updated_rec = get_record(rec.id, db_path=temp_database)
    assert updated_rec is not None
    assert updated_rec.status == RecordStatus.APPROVED


def test_patched_llm_exception_shows_friendly_error():
    """Requirement 3f: When patched LLM raises an exception, app shows friendly error and does not crash."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception

    with patch(
        "src.triage.call_llm_with_meta",
        side_effect=RuntimeError("Connection to local Ollama server refused"),
    ):
        at.text_area(key="msg_text_area").input("Need to book an annual checkup").run()
        at.button(key="btn_next").click().run()

        # The app should NOT crash with an uncaught exception
        assert not at.exception
        # Friendly error message should be displayed
        error_texts = [e.value for e in at.error]
        assert any(
            "Triage Failed" in err or "Connection to local Ollama server refused" in err
            for err in error_texts
        )


def test_sample_message_selector_populates_text_area():
    """Selecting a sample message populates the text area widget directly."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception
    assert at.text_area(key="msg_text_area").value == ""

    # Select the first sample message
    sample_key = list(SAMPLE_MESSAGES.keys())[0]
    expected_text = SAMPLE_MESSAGES[sample_key]
    at.sidebar.selectbox(key="sample_selector").select(sample_key).run()

    assert not at.exception
    assert at.text_area(key="msg_text_area").value == expected_text
