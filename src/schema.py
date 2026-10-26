"""Pydantic v2 schemas for the clinic triage pipeline.

Defines core data models, enums, and utility functions for message
classification, urgency assessment, staff routing, red-flag detection,
triage outcomes, and persistent triage records.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class Intent(str, Enum):
    """Categorized primary intent of an incoming patient message."""

    APPOINTMENT = "appointment"
    REFILL = "refill"
    TEST_RESULTS = "test_results"
    ADMINISTRATIVE = "administrative"
    GENERAL_QUERY = "general_query"
    OTHER = "other"


class Urgency(str, Enum):
    """Clinical or operational urgency level for triage prioritization."""

    ROUTINE = "routine"
    SOON = "soon"
    URGENT = "urgent"
    EMERGENCY = "emergency"


class RouteTo(str, Enum):
    """Staff role or clinical queue designated to resolve the message."""

    ON_CALL_CLINICIAN = "on_call_clinician"
    PHARMACIST_OR_MA = "pharmacist_or_ma"
    FRONT_DESK = "front_desk"


class RecordStatus(str, Enum):
    """Review and approval status of a triage record in the clinic workflow."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class RedFlag(BaseModel):
    """Assessment of emergency symptoms or safety-critical red flags."""

    is_red_flag: bool = Field(
        ...,
        description="Whether the patient message meets safety or emergency red-flag criteria.",
    )
    reason: Optional[str] = Field(
        default=None,
        description="Clinical rationale explaining why red-flag criteria were triggered, if applicable.",
    )

    @model_validator(mode="after")
    def validate_reason_if_red_flag(self) -> RedFlag:
        """Validate that a non-empty reason is provided if is_red_flag is True."""
        if self.is_red_flag:
            if not self.reason or not self.reason.strip():
                raise ValueError(
                    "A non-empty reason must be provided when is_red_flag is True."
                )
        return self


class Contact(BaseModel):
    """Patient contact information extracted from intake or message content."""

    name: Optional[str] = Field(
        default=None,
        description="Patient's full or preferred name if provided.",
    )
    phone: Optional[str] = Field(
        default=None,
        description="Patient's telephone contact number if provided.",
    )
    email: Optional[str] = Field(
        default=None,
        description="Patient's email address if provided.",
    )
    preferred_method: Optional[str] = Field(
        default=None,
        description="Patient's preferred contact method (phone, email, no preference) if provided.",
    )


class FollowUpAnswer(BaseModel):
    """A single follow-up intake question and patient response pair."""

    question: str = Field(
        ...,
        description="The follow-up question presented to the patient during intake.",
    )
    answer: str = Field(
        ...,
        description="The patient's response to the follow-up question.",
    )


class LLMTriageOutput(BaseModel):
    """Structured extraction output produced directly by the triage LLM.

    Restricted to only the decisions made by the language model. Rule-based
    assessments (red flags, routing, intake follow-ups) are merged separately.
    Disallows unknown or extra fields to catch LLM hallucinations.
    """

    model_config = ConfigDict(extra="forbid")

    intent: Intent = Field(
        ...,
        description="Primary categorized intent of the incoming patient message.",
    )
    urgency: Urgency = Field(
        ...,
        description="Assessed clinical or operational urgency level.",
    )
    suggested_specialty: Optional[str] = Field(
        default=None,
        description="Suggested clinical specialty (e.g., Cardiology, Dermatology) if specialist review is warranted.",
    )
    contact: Contact = Field(
        ...,
        description="Extracted patient contact details.",
    )
    summary: str = Field(
        ...,
        min_length=1,
        description="One-sentence clinical or operational summary of the patient message.",
    )
    draft_reply: str = Field(
        ...,
        min_length=1,
        description="Preliminary draft response prepared for staff review and approval.",
    )


class TriageResult(BaseModel):
    """Complete triage outcome and action plan for a patient inbox message."""

    intent: Intent = Field(
        ...,
        description="Primary categorized intent of the incoming patient message.",
    )
    urgency: Urgency = Field(
        ...,
        description="Assessed clinical or operational urgency level.",
    )
    route_to: RouteTo = Field(
        ...,
        description="Clinic staff queue or department designated to handle the message.",
    )
    suggested_specialty: Optional[str] = Field(
        default=None,
        description="Suggested clinical specialty (e.g., Cardiology, Dermatology) if specialist review is warranted.",
    )
    red_flag: RedFlag = Field(
        ...,
        description="Red flag safety assessment containing flag status and justification.",
    )
    follow_up: list[FollowUpAnswer] = Field(
        ...,
        description="List of intake follow-up questions and patient answers.",
    )
    contact: Contact = Field(
        ...,
        description="Extracted patient contact details.",
    )
    summary: str = Field(
        ...,
        min_length=1,
        description="One-sentence clinical or operational summary of the patient message.",
    )
    draft_reply: str = Field(
        ...,
        min_length=1,
        description="Preliminary draft response prepared for staff review and approval.",
    )


class TriageRecord(BaseModel):
    """Persistent audit and review record for a triaged message."""

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier (UUID string) for the triage record.",
    )
    received_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the message was received in UTC.",
    )
    original_message: str = Field(
        ...,
        min_length=1,
        description="Original raw patient message text.",
    )
    result: TriageResult = Field(
        ...,
        description="Complete triage outcome and action plan.",
    )
    status: RecordStatus = Field(
        default=RecordStatus.PENDING,
        description="Review and approval status (pending, approved, rejected).",
    )
    final_reply: Optional[str] = Field(
        default=None,
        description="Final approved or edited response message sent to patient.",
    )

    @field_validator("received_at", mode="after")
    def ensure_utc(cls, v: datetime) -> datetime:
        """Ensure received_at timestamp is timezone-aware in UTC."""
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)


def build_triage_result(
    llm_output: LLMTriageOutput,
    route_to: RouteTo,
    red_flag: RedFlag,
    follow_up: list[FollowUpAnswer],
    contact: Optional[Contact] = None,
) -> TriageResult:
    """Merge LLM extraction output with rule-based triage components.

    Args:
        llm_output: The structured output extracted by the LLM.
        route_to: Staff queue determined by deterministic routing rules.
        red_flag: Safety assessment determined by rule-based keyword detection.
        follow_up: List of intake follow-up questions and patient answers.
        contact: Optional merged or validated Contact instance to override LLM contact.

    Returns:
        A consolidated TriageResult instance.
    """
    return TriageResult(
        intent=llm_output.intent,
        urgency=llm_output.urgency,
        route_to=route_to,
        suggested_specialty=llm_output.suggested_specialty,
        red_flag=red_flag,
        follow_up=follow_up,
        contact=contact if contact is not None else llm_output.contact,
        summary=llm_output.summary,
        draft_reply=llm_output.draft_reply,
    )


__all__ = [
    "Contact",
    "FollowUpAnswer",
    "Intent",
    "LLMTriageOutput",
    "RecordStatus",
    "RedFlag",
    "RouteTo",
    "TriageRecord",
    "TriageResult",
    "Urgency",
    "build_triage_result",
]
