"""SQLite persistence layer for clinic triage records.

Stores TriageRecord models, provides querying with clinical priority ordering
(emergency > urgent > soon > routine, oldest first), staff review status
updates, and aggregate triage queue statistics using standard library sqlite3.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Optional

from src.schema import (
    RecordStatus,
    RouteTo,
    TriageRecord,
    TriageResult,
    Urgency,
)

DEFAULT_DB_PATH: Final[str] = "data/triage.db"


def _get_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Create and return an SQLite connection with Row factory enabled.

    Args:
        db_path: Path to database file or ':memory:' (defaults to DEFAULT_DB_PATH).

    Returns:
        Configured sqlite3.Connection instance.
    """
    path_str = str(db_path or DEFAULT_DB_PATH)
    if path_str != ":memory:" and not path_str.startswith("file:"):
        target_path = Path(path_str)
        if target_path.parent and str(target_path.parent) != ".":
            target_path.parent.mkdir(parents=True, exist_ok=True)

    uri = path_str.startswith("file:")
    conn = sqlite3.connect(path_str, uri=uri)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_table(conn: sqlite3.Connection) -> None:
    """Ensure triage_records table and indexes exist on the given connection."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS triage_records (
            id TEXT PRIMARY KEY,
            received_at TEXT,
            original_message TEXT,
            status TEXT,
            final_reply TEXT NULL,
            model TEXT,
            prompt_version TEXT,
            latency_s REAL,
            intent TEXT,
            urgency TEXT,
            route_to TEXT,
            is_red_flag INTEGER,
            result_json TEXT
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_triage_records_status ON triage_records(status);"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_triage_records_route_to ON triage_records(route_to);"
    )
    conn.commit()


def init_db(db_path: Path | str | None = None) -> None:
    """Initialize SQLite database and create triage_records table and indexes if missing.

    Idempotent: safe to invoke repeatedly without error.

    Args:
        db_path: Path to database file (defaults to DEFAULT_DB_PATH).
    """
    conn = _get_connection(db_path)
    try:
        _ensure_table(conn)
    finally:
        conn.close()


def save_record(
    record: TriageRecord,
    model: str,
    prompt_version: str,
    latency_s: float,
    db_path: Path | str | None = None,
) -> None:
    """Persist a TriageRecord instance to SQLite.

    Args:
        record: TriageRecord instance to store.
        model: Model identifier used for classification (e.g. 'llama3.2:3b').
        prompt_version: System prompt template version (e.g. 'v1').
        latency_s: Elapsed inference latency in seconds.
        db_path: Optional path to SQLite database.
    """
    conn = _get_connection(db_path)
    try:
        _ensure_table(conn)
        status_val = (
            record.status.value
            if hasattr(record.status, "value")
            else str(record.status)
        )
        intent_val = (
            record.result.intent.value
            if hasattr(record.result.intent, "value")
            else str(record.result.intent)
        )
        urgency_val = (
            record.result.urgency.value
            if hasattr(record.result.urgency, "value")
            else str(record.result.urgency)
        )
        route_val = (
            record.result.route_to.value
            if hasattr(record.result.route_to, "value")
            else str(record.result.route_to)
        )
        is_red_flag_val = 1 if record.result.red_flag.is_red_flag else 0
        result_json_val = record.result.model_dump_json()

        conn.execute(
            """
            INSERT OR REPLACE INTO triage_records (
                id, received_at, original_message, status, final_reply,
                model, prompt_version, latency_s, intent, urgency,
                route_to, is_red_flag, result_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(record.id),
                record.received_at.isoformat(),
                str(record.original_message),
                status_val,
                record.final_reply,
                str(model),
                str(prompt_version),
                float(latency_s),
                intent_val,
                urgency_val,
                route_val,
                is_red_flag_val,
                result_json_val,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_record(
    record_id: str,
    db_path: Path | str | None = None,
) -> TriageRecord | None:
    """Retrieve a TriageRecord from SQLite by ID, rebuilding from stored result_json.

    Args:
        record_id: Unique record ID string.
        db_path: Optional path to SQLite database.

    Returns:
        Rebuilt TriageRecord instance, or None if not found.
    """
    conn = _get_connection(db_path)
    try:
        _ensure_table(conn)
        cursor = conn.execute(
            """
            SELECT id, received_at, original_message, status, final_reply, result_json
            FROM triage_records
            WHERE id = ?
            """,
            (str(record_id),),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        result = TriageResult.model_validate_json(row["result_json"])
        return TriageRecord(
            id=str(row["id"]),
            received_at=datetime.fromisoformat(row["received_at"]),
            original_message=str(row["original_message"]),
            result=result,
            status=RecordStatus(row["status"]),
            final_reply=row["final_reply"],
        )
    finally:
        conn.close()


def list_records(
    status: str | RecordStatus | None = None,
    route_to: str | RouteTo | None = None,
    db_path: Path | str | None = None,
) -> list[TriageRecord]:
    """List triaged records from SQLite with optional filtering and priority ordering.

    Ordered with urgency emergency > urgent > soon > routine first, then oldest first.

    Args:
        status: Optional review status filter ('pending', 'approved', 'rejected').
        route_to: Optional queue filter ('on_call_clinician', 'pharmacist_or_ma', 'front_desk').
        db_path: Optional path to SQLite database.

    Returns:
        List of matching TriageRecord instances.
    """
    conn = _get_connection(db_path)
    try:
        _ensure_table(conn)
        conditions = []
        params = []

        if status is not None:
            status_val = (
                status.value
                if hasattr(status, "value")
                else str(status).lower().strip()
            )
            conditions.append("status = ?")
            params.append(status_val)

        if route_to is not None:
            route_val = (
                route_to.value
                if hasattr(route_to, "value")
                else str(route_to).lower().strip()
            )
            conditions.append("route_to = ?")
            params.append(route_val)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"""
            SELECT id, received_at, original_message, status, final_reply, result_json
            FROM triage_records
            {where_clause}
            ORDER BY
                CASE urgency
                    WHEN 'emergency' THEN 1
                    WHEN 'urgent' THEN 2
                    WHEN 'soon' THEN 3
                    WHEN 'routine' THEN 4
                    ELSE 5
                END ASC,
                received_at ASC
        """
        cursor = conn.execute(query, params)
        records: list[TriageRecord] = []
        for row in cursor.fetchall():
            result = TriageResult.model_validate_json(row["result_json"])
            records.append(
                TriageRecord(
                    id=str(row["id"]),
                    received_at=datetime.fromisoformat(row["received_at"]),
                    original_message=str(row["original_message"]),
                    result=result,
                    status=RecordStatus(row["status"]),
                    final_reply=row["final_reply"],
                )
            )
        return records
    finally:
        conn.close()


def update_status(
    record_id: str,
    status: str | RecordStatus,
    final_reply: Optional[str] = None,
    db_path: Path | str | None = None,
) -> bool:
    """Update review status and optional final reply for an existing triage record.

    Args:
        record_id: Unique record ID string.
        status: Target review status ('approved' or 'rejected').
        final_reply: Optional approved or edited reply text sent to the patient.
        db_path: Optional path to SQLite database.

    Returns:
        True if record was found and updated, False if record_id does not exist.

    Raises:
        ValueError: If status is not 'approved' or 'rejected'.
    """
    status_str = (
        status.value if hasattr(status, "value") else str(status).lower().strip()
    )
    if status_str not in ("approved", "rejected"):
        raise ValueError(
            f"Invalid status '{status}'. Status must be 'approved' or 'rejected'."
        )

    conn = _get_connection(db_path)
    try:
        _ensure_table(conn)
        if final_reply is not None:
            cursor = conn.execute(
                "UPDATE triage_records SET status = ?, final_reply = ? WHERE id = ?",
                (status_str, final_reply, str(record_id)),
            )
        else:
            cursor = conn.execute(
                "UPDATE triage_records SET status = ? WHERE id = ?",
                (status_str, str(record_id)),
            )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def get_stats(db_path: Path | str | None = None) -> dict[str, Any]:
    """Compute aggregate triage metrics and counts across stored records.

    Args:
        db_path: Optional path to SQLite database.

    Returns:
        Dict containing counts by status, by route_to, by urgency, red flag count,
        and average latency.
    """
    conn = _get_connection(db_path)
    try:
        _ensure_table(conn)
        # Status counts
        status_cursor = conn.execute(
            "SELECT status, COUNT(*) as cnt FROM triage_records GROUP BY status"
        )
        by_status = {row["status"]: int(row["cnt"]) for row in status_cursor.fetchall()}

        # Route_to counts
        route_cursor = conn.execute(
            "SELECT route_to, COUNT(*) as cnt FROM triage_records GROUP BY route_to"
        )
        by_route_to = {row["route_to"]: int(row["cnt"]) for row in route_cursor.fetchall()}

        # Urgency counts
        urgency_cursor = conn.execute(
            "SELECT urgency, COUNT(*) as cnt FROM triage_records GROUP BY urgency"
        )
        by_urgency = {row["urgency"]: int(row["cnt"]) for row in urgency_cursor.fetchall()}

        # Red flag count
        rf_cursor = conn.execute(
            "SELECT COUNT(*) as cnt FROM triage_records WHERE is_red_flag = 1"
        )
        rf_row = rf_cursor.fetchone()
        red_flag_count = int(rf_row["cnt"]) if rf_row else 0

        # Latency & total
        lat_cursor = conn.execute(
            "SELECT AVG(latency_s) as avg_lat, COUNT(*) as total FROM triage_records"
        )
        lat_row = lat_cursor.fetchone()
        avg_lat = float(lat_row["avg_lat"]) if (lat_row and lat_row["avg_lat"] is not None) else 0.0
        total_count = int(lat_row["total"]) if lat_row else 0

        return {
            "by_status": by_status,
            "by_route_to": by_route_to,
            "by_urgency": by_urgency,
            "red_flag_count": red_flag_count,
            "average_latency": round(avg_lat, 4),
            "average_latency_s": round(avg_lat, 4),
            "avg_latency": round(avg_lat, 4),
            "status": by_status,
            "route_to": by_route_to,
            "urgency": by_urgency,
            "counts_by_status": by_status,
            "counts_by_route_to": by_route_to,
            "counts_by_urgency": by_urgency,
            "red_flags": red_flag_count,
            "total_records": total_count,
        }
    finally:
        conn.close()


__all__ = [
    "DEFAULT_DB_PATH",
    "get_record",
    "get_stats",
    "init_db",
    "list_records",
    "save_record",
    "update_status",
]
