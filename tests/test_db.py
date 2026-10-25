"""Tests for src/db.py SQLite persistence layer.

Covers:
- Database initialization and idempotency (tables, indexes, pragma check)
- TriageRecord round-trip persistence and equality (save_record, get_record)
- Handling of non-existent record IDs
- list_records priority ordering (emergency > urgent > soon > routine, oldest first)
- list_records filtering by status, route_to, and combined
- update_status success with approved and rejected
- update_status with custom final_reply
- update_status on unknown ID returns False
- update_status with invalid status values raises ValueError
- get_stats aggregation (empty database and populated counts, red flags, avg latency)
- DEFAULT_DB_PATH constant
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from src.db import (
    DEFAULT_DB_PATH,
    get_record,
    get_stats,
    init_db,
    list_records,
    save_record,
    update_status,
)
from src.schema import (
    Contact,
    FollowUpAnswer,
    Intent,
    RecordStatus,
    RedFlag,
    RouteTo,
    TriageRecord,
    TriageResult,
    Urgency,
)


def _make_sample_record(
    intent: Intent = Intent.APPOINTMENT,
    urgency: Urgency = Urgency.ROUTINE,
    route_to: RouteTo = RouteTo.FRONT_DESK,
    is_red_flag: bool = False,
    red_flag_reason: str | None = None,
    received_at: datetime | None = None,
    status: RecordStatus = RecordStatus.PENDING,
    final_reply: str | None = None,
    original_message: str = "Test message for triage.",
) -> TriageRecord:
    """Helper to construct a complete, valid TriageRecord instance."""
    reason = red_flag_reason if is_red_flag else None
    result = TriageResult(
        intent=intent,
        urgency=urgency,
        route_to=route_to,
        suggested_specialty="Family Medicine" if intent == Intent.APPOINTMENT else None,
        red_flag=RedFlag(is_red_flag=is_red_flag, reason=reason),
        follow_up=[
            FollowUpAnswer(
                question="How long have you had symptoms?",
                answer="3 days.",
            )
        ],
        contact=Contact(
            name="John Doe",
            phone="555-0199",
            email="johndoe@example.com",
        ),
        summary="Patient reports mild symptoms and requests an appointment.",
        draft_reply="Hello John, we will gladly assist you with an appointment.",
    )
    return TriageRecord(
        id=str(uuid4()),
        received_at=received_at or datetime.now(timezone.utc),
        original_message=original_message,
        result=result,
        status=status,
        final_reply=final_reply,
    )


# ============================================================================
# 1. Database Initialization and Idempotency
# ============================================================================


def test_default_db_path_constant():
    """DEFAULT_DB_PATH must equal 'data/triage.db'."""
    assert DEFAULT_DB_PATH == "data/triage.db"


def test_init_db_creates_tables_and_indexes(tmp_path: Path):
    """init_db creates table triage_records and expected indexes."""
    db_file = tmp_path / "test_init.db"
    assert not db_file.exists()

    init_db(db_file)
    assert db_file.exists()

    conn = sqlite3.connect(str(db_file))
    try:
        # Check table presence
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='triage_records'"
        ).fetchall()
        assert len(tables) == 1

        # Check indexes presence
        indexes = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        assert "idx_triage_records_status" in indexes
        assert "idx_triage_records_route_to" in indexes

        # Check column names in schema
        columns = [
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(triage_records)"
            ).fetchall()
        ]
        expected_columns = [
            "id",
            "received_at",
            "original_message",
            "status",
            "final_reply",
            "model",
            "prompt_version",
            "latency_s",
            "intent",
            "urgency",
            "route_to",
            "is_red_flag",
            "result_json",
        ]
        assert columns == expected_columns
    finally:
        conn.close()


def test_init_db_is_idempotent(tmp_path: Path):
    """Calling init_db multiple times on an existing database succeeds without error."""
    db_file = tmp_path / "test_idempotent.db"
    init_db(db_file)
    # Subsequent calls should succeed silently without altering existing state
    init_db(db_file)
    init_db(db_file)

    conn = sqlite3.connect(str(db_file))
    try:
        count = conn.execute(
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='triage_records'"
        ).fetchone()[0]
        assert count == 1
    finally:
        conn.close()


# ============================================================================
# 2. Round-trip Persistence: save_record and get_record
# ============================================================================


def test_save_and_get_record_round_trip(tmp_path: Path):
    """save_record followed by get_record returns an equal TriageRecord."""
    db_file = tmp_path / "test_roundtrip.db"
    init_db(db_file)

    record = _make_sample_record(
        intent=Intent.REFILL,
        urgency=Urgency.SOON,
        route_to=RouteTo.PHARMACIST_OR_MA,
        is_red_flag=False,
        original_message="Please refill my lisinopril 10mg.",
    )

    save_record(
        record=record,
        model="llama3.2:3b",
        prompt_version="v1",
        latency_s=0.4512,
        db_path=db_file,
    )

    retrieved = get_record(record.id, db_path=db_file)
    assert retrieved is not None
    assert retrieved == record
    assert retrieved.id == record.id
    assert retrieved.received_at == record.received_at
    assert retrieved.original_message == record.original_message
    assert retrieved.status == record.status
    assert retrieved.final_reply == record.final_reply
    assert retrieved.result == record.result


def test_save_and_get_record_with_red_flag(tmp_path: Path):
    """save_record round-trips emergency records with red flag reason."""
    db_file = tmp_path / "test_redflag_roundtrip.db"
    init_db(db_file)

    record = _make_sample_record(
        intent=Intent.GENERAL_QUERY,
        urgency=Urgency.EMERGENCY,
        route_to=RouteTo.ON_CALL_CLINICIAN,
        is_red_flag=True,
        red_flag_reason="cardiac: crushing chest pain radiating to left arm",
        original_message="Crushing chest pain and hard to breathe.",
    )

    save_record(
        record=record,
        model="llama3.2:3b",
        prompt_version="v1",
        latency_s=0.12,
        db_path=db_file,
    )

    retrieved = get_record(record.id, db_path=db_file)
    assert retrieved is not None
    assert retrieved == record
    assert retrieved.result.red_flag.is_red_flag is True
    assert retrieved.result.red_flag.reason == "cardiac: crushing chest pain radiating to left arm"


def test_get_record_nonexistent_returns_none(tmp_path: Path):
    """get_record returns None when record_id does not exist."""
    db_file = tmp_path / "test_nonexistent.db"
    init_db(db_file)

    retrieved = get_record("unknown-id-000", db_path=db_file)
    assert retrieved is None


def test_save_record_replaces_existing_id(tmp_path: Path):
    """save_record with an existing ID replaces the record."""
    db_file = tmp_path / "test_replace.db"
    init_db(db_file)

    record = _make_sample_record(original_message="First version.")
    save_record(record, "llama3.2:3b", "v1", 0.5, db_path=db_file)

    # Modify message and save again
    updated_record = record.model_copy(update={"original_message": "Updated version."})
    save_record(updated_record, "llama3.2:3b", "v1", 0.6, db_path=db_file)

    retrieved = get_record(record.id, db_path=db_file)
    assert retrieved is not None
    assert retrieved.original_message == "Updated version."


# ============================================================================
# 3. List Records Ordering and Filtering
# ============================================================================


def test_list_records_clinical_priority_and_fifo_ordering(tmp_path: Path):
    """list_records orders by emergency > urgent > soon > routine, then oldest first."""
    db_file = tmp_path / "test_priority.db"
    init_db(db_file)

    base_time = datetime(2026, 10, 25, 10, 0, 0, tzinfo=timezone.utc)

    # Create records with intentional disorder across urgency and arrival time:
    # 2 emergencies (one earlier, one later)
    # 2 urgents (one earlier, one later)
    # 2 soon (one earlier, one later)
    # 2 routines (one earlier, one later)
    r_routine_old = _make_sample_record(
        urgency=Urgency.ROUTINE,
        received_at=base_time + timedelta(minutes=1),
        original_message="Routine Old",
    )
    r_soon_new = _make_sample_record(
        urgency=Urgency.SOON,
        received_at=base_time + timedelta(minutes=10),
        original_message="Soon New",
    )
    r_urgent_new = _make_sample_record(
        urgency=Urgency.URGENT,
        received_at=base_time + timedelta(minutes=8),
        original_message="Urgent New",
    )
    r_emergency_new = _make_sample_record(
        urgency=Urgency.EMERGENCY,
        is_red_flag=True,
        red_flag_reason="cardiac: chest pain",
        received_at=base_time + timedelta(minutes=6),
        original_message="Emergency New",
    )
    r_routine_new = _make_sample_record(
        urgency=Urgency.ROUTINE,
        received_at=base_time + timedelta(minutes=15),
        original_message="Routine New",
    )
    r_soon_old = _make_sample_record(
        urgency=Urgency.SOON,
        received_at=base_time + timedelta(minutes=3),
        original_message="Soon Old",
    )
    r_emergency_old = _make_sample_record(
        urgency=Urgency.EMERGENCY,
        is_red_flag=True,
        red_flag_reason="respiratory: cannot breathe",
        received_at=base_time + timedelta(minutes=2),
        original_message="Emergency Old",
    )
    r_urgent_old = _make_sample_record(
        urgency=Urgency.URGENT,
        received_at=base_time + timedelta(minutes=4),
        original_message="Urgent Old",
    )

    records = [
        r_routine_old,
        r_soon_new,
        r_urgent_new,
        r_emergency_new,
        r_routine_new,
        r_soon_old,
        r_emergency_old,
        r_urgent_old,
    ]
    for r in records:
        save_record(r, "llama3.2:3b", "v1", 0.3, db_path=db_file)

    results = list_records(db_path=db_file)
    assert len(results) == 8

    expected_ids = [
        r_emergency_old.id,  # Emergency, min=2
        r_emergency_new.id,  # Emergency, min=6
        r_urgent_old.id,     # Urgent, min=4
        r_urgent_new.id,     # Urgent, min=8
        r_soon_old.id,       # Soon, min=3
        r_soon_new.id,       # Soon, min=10
        r_routine_old.id,    # Routine, min=1
        r_routine_new.id,    # Routine, min=15
    ]
    assert [r.id for r in results] == expected_ids


def test_list_records_filtering(tmp_path: Path):
    """list_records supports filtering by status, route_to, and combined criteria."""
    db_file = tmp_path / "test_filter.db"
    init_db(db_file)

    r1 = _make_sample_record(
        route_to=RouteTo.FRONT_DESK,
        status=RecordStatus.PENDING,
        original_message="Msg 1",
    )
    r2 = _make_sample_record(
        route_to=RouteTo.ON_CALL_CLINICIAN,
        status=RecordStatus.PENDING,
        original_message="Msg 2",
    )
    r3 = _make_sample_record(
        route_to=RouteTo.PHARMACIST_OR_MA,
        status=RecordStatus.APPROVED,
        final_reply="Approved refill.",
        original_message="Msg 3",
    )
    r4 = _make_sample_record(
        route_to=RouteTo.FRONT_DESK,
        status=RecordStatus.REJECTED,
        original_message="Msg 4",
    )

    for r in (r1, r2, r3, r4):
        save_record(r, "llama3.2:3b", "v1", 0.25, db_path=db_file)

    # Filter by status string
    pending_records = list_records(status="pending", db_path=db_file)
    assert {r.id for r in pending_records} == {r1.id, r2.id}

    approved_records = list_records(status="approved", db_path=db_file)
    assert {r.id for r in approved_records} == {r3.id}

    rejected_records = list_records(status="rejected", db_path=db_file)
    assert {r.id for r in rejected_records} == {r4.id}

    # Filter by status Enum
    pending_enum_records = list_records(status=RecordStatus.PENDING, db_path=db_file)
    assert {r.id for r in pending_enum_records} == {r1.id, r2.id}

    # Filter by route_to string
    clinician_records = list_records(route_to="on_call_clinician", db_path=db_file)
    assert {r.id for r in clinician_records} == {r2.id}

    front_desk_records = list_records(route_to="front_desk", db_path=db_file)
    assert {r.id for r in front_desk_records} == {r1.id, r4.id}

    # Filter by route_to Enum
    pharmacy_records = list_records(route_to=RouteTo.PHARMACIST_OR_MA, db_path=db_file)
    assert {r.id for r in pharmacy_records} == {r3.id}

    # Combined filter
    combined = list_records(status="pending", route_to="front_desk", db_path=db_file)
    assert {r.id for r in combined} == {r1.id}

    # Non-matching filter returns empty list
    empty = list_records(status="approved", route_to="front_desk", db_path=db_file)
    assert empty == []


# ============================================================================
# 4. update_status: Success, Unknown ID, and Validation
# ============================================================================


def test_update_status_success_approved_with_reply(tmp_path: Path):
    """update_status updates status to approved and sets final_reply."""
    db_file = tmp_path / "test_update_approved.db"
    init_db(db_file)

    record = _make_sample_record(status=RecordStatus.PENDING, final_reply=None)
    save_record(record, "llama3.2:3b", "v1", 0.4, db_path=db_file)

    success = update_status(
        record_id=record.id,
        status="approved",
        final_reply="Your appointment is confirmed for Thursday at 2:00 PM.",
        db_path=db_file,
    )
    assert success is True

    updated = get_record(record.id, db_path=db_file)
    assert updated is not None
    assert updated.status == RecordStatus.APPROVED
    assert updated.final_reply == "Your appointment is confirmed for Thursday at 2:00 PM."


def test_update_status_success_rejected_without_reply(tmp_path: Path):
    """update_status updates status to rejected without altering null reply."""
    db_file = tmp_path / "test_update_rejected.db"
    init_db(db_file)

    record = _make_sample_record(status=RecordStatus.PENDING, final_reply=None)
    save_record(record, "llama3.2:3b", "v1", 0.4, db_path=db_file)

    success = update_status(
        record_id=record.id,
        status="rejected",
        db_path=db_file,
    )
    assert success is True

    updated = get_record(record.id, db_path=db_file)
    assert updated is not None
    assert updated.status == RecordStatus.REJECTED
    assert updated.final_reply is None


def test_update_status_unknown_id_returns_false(tmp_path: Path):
    """update_status returns False when record_id does not exist."""
    db_file = tmp_path / "test_update_unknown.db"
    init_db(db_file)

    assert update_status("nonexistent-uuid-1234", "approved", db_path=db_file) is False
    assert update_status("nonexistent-uuid-5678", "rejected", db_path=db_file) is False


def test_update_status_invalid_value_raises_value_error(tmp_path: Path):
    """update_status raises ValueError for invalid status values (including pending)."""
    db_file = tmp_path / "test_update_invalid.db"
    init_db(db_file)

    record = _make_sample_record()
    save_record(record, "llama3.2:3b", "v1", 0.4, db_path=db_file)

    for invalid_val in ["pending", "in_progress", "canceled", "unknown", "", "APPROVED_EXTRA"]:
        with pytest.raises(ValueError, match="Status must be 'approved' or 'rejected'"):
            update_status(record.id, invalid_val, db_path=db_file)

    # RecordStatus.PENDING enum should also be rejected by update_status
    with pytest.raises(ValueError, match="Status must be 'approved' or 'rejected'"):
        update_status(record.id, RecordStatus.PENDING, db_path=db_file)


# ============================================================================
# 5. get_stats Aggregation
# ============================================================================


def test_get_stats_empty_database(tmp_path: Path):
    """get_stats on an empty database returns empty dicts and zero totals."""
    db_file = tmp_path / "test_stats_empty.db"
    init_db(db_file)

    stats = get_stats(db_path=db_file)
    assert stats["by_status"] == {}
    assert stats["by_route_to"] == {}
    assert stats["by_urgency"] == {}
    assert stats["red_flag_count"] == 0
    assert stats["average_latency"] == 0.0


def test_get_stats_populated_counts(tmp_path: Path):
    """get_stats accurately calculates counts by status, route_to, urgency, red flags, and avg latency."""
    db_file = tmp_path / "test_stats_populated.db"
    init_db(db_file)

    # 1. Pending, Front Desk, Routine, No red flag, lat=1.0
    r1 = _make_sample_record(
        status=RecordStatus.PENDING,
        route_to=RouteTo.FRONT_DESK,
        urgency=Urgency.ROUTINE,
        is_red_flag=False,
    )
    save_record(r1, "llama3.2:3b", "v1", 1.0, db_path=db_file)

    # 2. Pending, On-call, Emergency, Red flag, lat=2.0
    r2 = _make_sample_record(
        status=RecordStatus.PENDING,
        route_to=RouteTo.ON_CALL_CLINICIAN,
        urgency=Urgency.EMERGENCY,
        is_red_flag=True,
        red_flag_reason="cardiac: chest pressure",
    )
    save_record(r2, "llama3.2:3b", "v1", 2.0, db_path=db_file)

    # 3. Approved, On-call, Emergency, Red flag, lat=3.0
    r3 = _make_sample_record(
        status=RecordStatus.APPROVED,
        route_to=RouteTo.ON_CALL_CLINICIAN,
        urgency=Urgency.EMERGENCY,
        is_red_flag=True,
        red_flag_reason="respiratory: shortness of breath",
        final_reply="Sent to ED.",
    )
    save_record(r3, "llama3.2:3b", "v1", 3.0, db_path=db_file)

    # 4. Rejected, Pharmacist, Urgent, No red flag, lat=4.0
    r4 = _make_sample_record(
        status=RecordStatus.REJECTED,
        route_to=RouteTo.PHARMACIST_OR_MA,
        urgency=Urgency.URGENT,
        is_red_flag=False,
    )
    save_record(r4, "llama3.2:3b", "v1", 4.0, db_path=db_file)

    stats = get_stats(db_path=db_file)

    # Verify status breakdown
    assert stats["by_status"] == {
        "pending": 2,
        "approved": 1,
        "rejected": 1,
    }

    # Verify route_to breakdown
    assert stats["by_route_to"] == {
        "front_desk": 1,
        "on_call_clinician": 2,
        "pharmacist_or_ma": 1,
    }

    # Verify urgency breakdown
    assert stats["by_urgency"] == {
        "routine": 1,
        "emergency": 2,
        "urgent": 1,
    }

    # Verify red flag count and average latency ((1.0 + 2.0 + 3.0 + 4.0) / 4 = 2.5)
    assert stats["red_flag_count"] == 2
    assert stats["average_latency"] == 2.5
