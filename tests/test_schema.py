"""Tests for src/schema.py Pydantic v2 models, enums, and helpers.

Covers:
- Valid model creation and data serialization
- Invalid enum values rejection
- Missing required fields validation
- Optional field defaults
- LLMTriageOutput extra field rejection (extra="forbid")
- build_triage_result integration
- RedFlag validator (non-empty reason required when is_red_flag is True)
- min_length=1 validation on summary and draft_reply
- TriageRecord creation, defaults (UUID, UTC received_at, status), and validations
"""

from datetime import datetime, timezone
from uuid import UUID

import pytest
from pydantic import ValidationError

from src.schema import (
    Contact,
    FollowUpAnswer,
    Intent,
    LLMTriageOutput,
    RecordStatus,
    RedFlag,
    RouteTo,
    TriageRecord,
    TriageResult,
    Urgency,
    build_triage_result,
)


@pytest.fixture
def valid_triage_data() -> dict:
    """Provides a complete, valid dictionary to construct TriageResult."""
    return {
        "intent": "appointment",
        "urgency": "routine",
        "route_to": "front_desk",
        "suggested_specialty": "Internal Medicine",
        "red_flag": {
            "is_red_flag": False,
            "reason": None,
        },
        "follow_up": [
            {
                "question": "Is this a follow-up or new issue?",
                "answer": "Routine annual checkup.",
            }
        ],
        "contact": {
            "name": "Jane Doe",
            "phone": "555-0142",
            "email": "jane.doe@example.com",
        },
        "summary": "Patient requests an appointment for a routine annual physical.",
        "draft_reply": "Hello Jane, we would be pleased to schedule your annual physical.",
    }


@pytest.fixture
def valid_llm_output_data() -> dict:
    """Provides valid data for LLMTriageOutput."""
    return {
        "intent": "refill",
        "urgency": "soon",
        "suggested_specialty": None,
        "contact": {
            "name": "Alice Green",
            "phone": "555-9876",
            "email": "alice@example.com",
        },
        "summary": "Patient requests blood pressure prescription refill.",
        "draft_reply": "Hello Alice, we have forwarded your refill request to pharmacy.",
    }


# =====================================================================
# 1. Valid Input Tests
# =====================================================================


def test_valid_triage_result_full(valid_triage_data: dict) -> None:
    """Verify full TriageResult instantiation with all fields populated."""
    result = TriageResult.model_validate(valid_triage_data)

    assert result.intent == Intent.APPOINTMENT
    assert result.urgency == Urgency.ROUTINE
    assert result.route_to == RouteTo.FRONT_DESK
    assert result.suggested_specialty == "Internal Medicine"
    assert result.red_flag.is_red_flag is False
    assert result.red_flag.reason is None
    assert len(result.follow_up) == 1
    assert result.follow_up[0].question == "Is this a follow-up or new issue?"
    assert result.follow_up[0].answer == "Routine annual checkup."
    assert result.contact.name == "Jane Doe"
    assert result.contact.phone == "555-0142"
    assert result.contact.email == "jane.doe@example.com"
    assert (
        result.summary
        == "Patient requests an appointment for a routine annual physical."
    )
    assert "schedule your annual physical" in result.draft_reply


def test_valid_triage_result_with_enum_instances() -> None:
    """Verify instantiating TriageResult directly using Enum instances."""
    result = TriageResult(
        intent=Intent.REFILL,
        urgency=Urgency.SOON,
        route_to=RouteTo.PHARMACIST_OR_MA,
        suggested_specialty=None,
        red_flag=RedFlag(is_red_flag=False),
        follow_up=[],
        contact=Contact(name="Bob Smith"),
        summary="Patient needs a blood pressure medication refill.",
        draft_reply="Hello Bob, your refill request has been sent to our pharmacy team.",
    )

    assert result.intent == Intent.REFILL
    assert result.urgency == Urgency.SOON
    assert result.route_to == RouteTo.PHARMACIST_OR_MA
    assert result.suggested_specialty is None
    assert result.follow_up == []
    assert result.contact.name == "Bob Smith"


def test_valid_triage_result_minimal_optional_fields(
    valid_triage_data: dict,
) -> None:
    """Verify TriageResult when optional fields are omitted or None."""
    data = valid_triage_data.copy()
    data.pop("suggested_specialty")
    data["follow_up"] = []
    data["contact"] = {}
    data["red_flag"] = {"is_red_flag": False}

    result = TriageResult.model_validate(data)
    assert result.suggested_specialty is None
    assert result.follow_up == []
    assert result.contact.name is None
    assert result.contact.phone is None
    assert result.contact.email is None
    assert result.red_flag.reason is None


def test_valid_contact_all_optional() -> None:
    """Verify Contact fields are all independently optional."""
    empty_contact = Contact()
    assert empty_contact.name is None
    assert empty_contact.phone is None
    assert empty_contact.email is None

    partial_contact = Contact(email="test@example.com")
    assert partial_contact.name is None
    assert partial_contact.phone is None
    assert partial_contact.email == "test@example.com"

    full_contact = Contact(
        name="Alice", phone="123-4567", email="alice@example.com"
    )
    assert full_contact.name == "Alice"
    assert full_contact.phone == "123-4567"
    assert full_contact.email == "alice@example.com"


def test_valid_red_flag() -> None:
    """Verify RedFlag handles both flagged (with reason) and non-flagged cases."""
    no_flag = RedFlag(is_red_flag=False)
    assert no_flag.is_red_flag is False
    assert no_flag.reason is None

    flagged = RedFlag(
        is_red_flag=True,
        reason="Patient reported severe acute chest pain radiating to jaw.",
    )
    assert flagged.is_red_flag is True
    assert "chest pain" in flagged.reason


def test_valid_follow_up_answer() -> None:
    """Verify FollowUpAnswer accepts valid question and answer strings."""
    qa = FollowUpAnswer(
        question="Are you experiencing shortness of breath?",
        answer="No, just mild coughing.",
    )
    assert qa.question == "Are you experiencing shortness of breath?"
    assert qa.answer == "No, just mild coughing."


def test_model_serialization(valid_triage_data: dict) -> None:
    """Verify round-trip serialization and deserialization via JSON."""
    result = TriageResult.model_validate(valid_triage_data)
    json_str = result.model_dump_json()
    reloaded = TriageResult.model_validate_json(json_str)

    assert reloaded == result
    assert reloaded.intent == Intent.APPOINTMENT


# =====================================================================
# 2. Invalid Enum Values Tests
# =====================================================================


@pytest.mark.parametrize(
    "invalid_intent",
    ["appointment_request", "billing", "emergency", "prescription", "", "UNKNOWN"],
)
def test_invalid_intent_values(
    valid_triage_data: dict, invalid_intent: str
) -> None:
    """Verify invalid intent values raise ValidationError."""
    data = valid_triage_data.copy()
    data["intent"] = invalid_intent
    with pytest.raises(ValidationError) as exc_info:
        TriageResult.model_validate(data)

    errors = exc_info.value.errors()
    assert any(err["loc"] == ("intent",) for err in errors)


@pytest.mark.parametrize(
    "invalid_urgency",
    ["high", "low", "critical", "medium", "stat", "asap", ""],
)
def test_invalid_urgency_values(
    valid_triage_data: dict, invalid_urgency: str
) -> None:
    """Verify invalid urgency values raise ValidationError."""
    data = valid_triage_data.copy()
    data["urgency"] = invalid_urgency
    with pytest.raises(ValidationError) as exc_info:
        TriageResult.model_validate(data)

    errors = exc_info.value.errors()
    assert any(err["loc"] == ("urgency",) for err in errors)


@pytest.mark.parametrize(
    "invalid_route",
    ["doctor", "nurse", "billing_dept", "triage_desk", "clinician", ""],
)
def test_invalid_route_to_values(
    valid_triage_data: dict, invalid_route: str
) -> None:
    """Verify invalid route_to values raise ValidationError."""
    data = valid_triage_data.copy()
    data["route_to"] = invalid_route
    with pytest.raises(ValidationError) as exc_info:
        TriageResult.model_validate(data)

    errors = exc_info.value.errors()
    assert any(err["loc"] == ("route_to",) for err in errors)


def test_direct_enum_instantiation_raises_value_error() -> None:
    """Verify direct instantiation of Enums with invalid strings raises ValueError."""
    with pytest.raises(ValueError):
        Intent("invalid_intent")

    with pytest.raises(ValueError):
        Urgency("invalid_urgency")

    with pytest.raises(ValueError):
        RouteTo("invalid_route")

    with pytest.raises(ValueError):
        RecordStatus("invalid_status")


# =====================================================================
# 3. Missing Required Fields Tests
# =====================================================================


@pytest.mark.parametrize(
    "missing_field",
    [
        "intent",
        "urgency",
        "route_to",
        "red_flag",
        "follow_up",
        "contact",
        "summary",
        "draft_reply",
    ],
)
def test_triage_result_missing_required_fields(
    valid_triage_data: dict, missing_field: str
) -> None:
    """Verify omitting any required field from TriageResult raises ValidationError."""
    data = valid_triage_data.copy()
    del data[missing_field]

    with pytest.raises(ValidationError) as exc_info:
        TriageResult.model_validate(data)

    errors = exc_info.value.errors()
    assert any(
        err["loc"] == (missing_field,) and err["type"] == "missing"
        for err in errors
    )


def test_red_flag_missing_required_is_red_flag() -> None:
    """Verify RedFlag requires is_red_flag boolean."""
    with pytest.raises(ValidationError) as exc_info:
        RedFlag.model_validate({})

    errors = exc_info.value.errors()
    assert any(
        err["loc"] == ("is_red_flag",) and err["type"] == "missing"
        for err in errors
    )


@pytest.mark.parametrize("missing_field", ["question", "answer"])
def test_follow_up_answer_missing_required_fields(missing_field: str) -> None:
    """Verify FollowUpAnswer requires both question and answer."""
    data = {
        "question": "What medications are you currently taking?",
        "answer": "Metformin 500mg daily.",
    }
    del data[missing_field]

    with pytest.raises(ValidationError) as exc_info:
        FollowUpAnswer.model_validate(data)

    errors = exc_info.value.errors()
    assert any(
        err["loc"] == (missing_field,) and err["type"] == "missing"
        for err in errors
    )


def test_optional_fields_do_not_raise(valid_triage_data: dict) -> None:
    """Verify optional fields (suggested_specialty, reason, contact details) can be omitted."""
    data = valid_triage_data.copy()
    data.pop("suggested_specialty", None)
    data["red_flag"] = {"is_red_flag": False}
    data["contact"] = {}

    result = TriageResult.model_validate(data)
    assert result.suggested_specialty is None
    assert result.red_flag.reason is None
    assert result.contact.name is None
    assert result.contact.phone is None
    assert result.contact.email is None


# =====================================================================
# 4. RedFlag Validator Tests (is_red_flag == True requires reason)
# =====================================================================


@pytest.mark.parametrize(
    "invalid_reason",
    [None, "", "   ", "\t\n"],
)
def test_red_flag_true_requires_non_empty_reason(
    invalid_reason: str | None,
) -> None:
    """Verify that when is_red_flag is True, reason must be a non-empty string."""
    with pytest.raises(ValidationError) as exc_info:
        RedFlag(is_red_flag=True, reason=invalid_reason)

    assert "non-empty reason" in str(exc_info.value)


def test_red_flag_true_omitted_reason_raises() -> None:
    """Verify omitting reason when is_red_flag is True raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        RedFlag(is_red_flag=True)

    assert "non-empty reason" in str(exc_info.value)


def test_red_flag_false_allows_empty_or_none_reason() -> None:
    """Verify that when is_red_flag is False, reason can be None or omitted."""
    rf_none = RedFlag(is_red_flag=False, reason=None)
    assert rf_none.is_red_flag is False
    assert rf_none.reason is None

    rf_omitted = RedFlag(is_red_flag=False)
    assert rf_omitted.is_red_flag is False
    assert rf_omitted.reason is None


# =====================================================================
# 5. String min_length=1 Constraints Tests (summary and draft_reply)
# =====================================================================


@pytest.mark.parametrize(
    "field_name",
    ["summary", "draft_reply"],
)
def test_triage_result_empty_string_fields_raise(
    valid_triage_data: dict, field_name: str
) -> None:
    """Verify summary and draft_reply cannot be empty strings on TriageResult."""
    data = valid_triage_data.copy()
    data[field_name] = ""

    with pytest.raises(ValidationError) as exc_info:
        TriageResult.model_validate(data)

    errors = exc_info.value.errors()
    assert any(
        err["loc"] == (field_name,) and "string_too_short" in err["type"]
        for err in errors
    )


@pytest.mark.parametrize(
    "field_name",
    ["summary", "draft_reply"],
)
def test_llm_triage_output_empty_string_fields_raise(
    valid_llm_output_data: dict, field_name: str
) -> None:
    """Verify summary and draft_reply cannot be empty strings on LLMTriageOutput."""
    data = valid_llm_output_data.copy()
    data[field_name] = ""

    with pytest.raises(ValidationError) as exc_info:
        LLMTriageOutput.model_validate(data)

    errors = exc_info.value.errors()
    assert any(
        err["loc"] == (field_name,) and "string_too_short" in err["type"]
        for err in errors
    )


# =====================================================================
# 6. LLMTriageOutput and extra="forbid" Tests
# =====================================================================


def test_valid_llm_triage_output(valid_llm_output_data: dict) -> None:
    """Verify LLMTriageOutput instantiates with valid data."""
    output = LLMTriageOutput.model_validate(valid_llm_output_data)

    assert output.intent == Intent.REFILL
    assert output.urgency == Urgency.SOON
    assert output.suggested_specialty is None
    assert output.contact.name == "Alice Green"
    assert "refill" in output.summary
    assert "pharmacy" in output.draft_reply


def test_llm_triage_output_rejects_extra_fields(
    valid_llm_output_data: dict,
) -> None:
    """Verify extra='forbid' on LLMTriageOutput catches unexpected fields/hallucinations."""
    data = valid_llm_output_data.copy()
    data["extra_analysis"] = "LLM generated extra thoughts"
    data["confidence_score"] = 0.95

    with pytest.raises(ValidationError) as exc_info:
        LLMTriageOutput.model_validate(data)

    errors = exc_info.value.errors()
    extra_field_errors = [
        err["loc"][0] for err in errors if err["type"] == "extra_forbidden"
    ]
    assert "extra_analysis" in extra_field_errors
    assert "confidence_score" in extra_field_errors


def test_other_models_do_not_forbid_extra(valid_triage_data: dict) -> None:
    """Verify extra='forbid' is applied strictly to LLMTriageOutput only."""
    data = valid_triage_data.copy()
    data["extra_metadata"] = "tolerated_in_triage_result"

    # Should not raise
    result = TriageResult.model_validate(data)
    assert result.intent == Intent.APPOINTMENT


@pytest.mark.parametrize(
    "missing_field",
    ["intent", "urgency", "contact", "summary", "draft_reply"],
)
def test_llm_triage_output_missing_required_fields(
    valid_llm_output_data: dict, missing_field: str
) -> None:
    """Verify required fields for LLMTriageOutput raise ValidationError if omitted."""
    data = valid_llm_output_data.copy()
    del data[missing_field]

    with pytest.raises(ValidationError) as exc_info:
        LLMTriageOutput.model_validate(data)

    errors = exc_info.value.errors()
    assert any(
        err["loc"] == (missing_field,) and err["type"] == "missing"
        for err in errors
    )


# =====================================================================
# 7. build_triage_result Helper Tests
# =====================================================================


def test_build_triage_result_success(valid_llm_output_data: dict) -> None:
    """Verify build_triage_result merges LLM decisions with rule-based fields."""
    llm_output = LLMTriageOutput.model_validate(valid_llm_output_data)
    route_to = RouteTo.PHARMACIST_OR_MA
    red_flag = RedFlag(is_red_flag=False, reason=None)
    follow_up = [
        FollowUpAnswer(
            question="Which pharmacy do you prefer?",
            answer="CVS on Main Street.",
        )
    ]

    triage_result = build_triage_result(
        llm_output=llm_output,
        route_to=route_to,
        red_flag=red_flag,
        follow_up=follow_up,
    )

    assert isinstance(triage_result, TriageResult)
    assert triage_result.intent == llm_output.intent
    assert triage_result.urgency == llm_output.urgency
    assert triage_result.route_to == route_to
    assert (
        triage_result.suggested_specialty == llm_output.suggested_specialty
    )
    assert triage_result.red_flag == red_flag
    assert triage_result.follow_up == follow_up
    assert triage_result.contact == llm_output.contact
    assert triage_result.summary == llm_output.summary
    assert triage_result.draft_reply == llm_output.draft_reply


# =====================================================================
# 8. TriageRecord Model Tests
# =====================================================================


def test_triage_record_defaults(valid_triage_data: dict) -> None:
    """Verify TriageRecord default values: generated UUID, UTC timestamp, and pending status."""
    result = TriageResult.model_validate(valid_triage_data)
    record = TriageRecord(
        original_message="Hello, I need an appointment please.",
        result=result,
    )

    # id should be a valid UUID string
    assert isinstance(record.id, str)
    UUID(record.id)  # Does not raise

    # received_at should be timezone-aware UTC datetime
    assert isinstance(record.received_at, datetime)
    assert record.received_at.tzinfo == timezone.utc

    # status defaults to RecordStatus.PENDING
    assert record.status == RecordStatus.PENDING
    assert record.status == "pending"

    # final_reply defaults to None
    assert record.final_reply is None
    assert record.original_message == "Hello, I need an appointment please."
    assert record.result == result


def test_triage_record_custom_fields(valid_triage_data: dict) -> None:
    """Verify TriageRecord allows custom id, received_at, status, and final_reply."""
    result = TriageResult.model_validate(valid_triage_data)
    custom_dt = datetime(2026, 10, 25, 8, 30, 0, tzinfo=timezone.utc)
    custom_id = "550e8400-e29b-41d4-a716-446655440000"

    record = TriageRecord(
        id=custom_id,
        received_at=custom_dt,
        original_message="Please renew my medication.",
        result=result,
        status=RecordStatus.APPROVED,
        final_reply="Your medication has been approved and sent to CVS.",
    )

    assert record.id == custom_id
    assert record.received_at == custom_dt
    assert record.status == RecordStatus.APPROVED
    assert record.status == "approved"
    assert (
        record.final_reply
        == "Your medication has been approved and sent to CVS."
    )


def test_triage_record_naive_datetime_converted_to_utc(
    valid_triage_data: dict,
) -> None:
    """Verify naive datetime passed to received_at is assigned UTC timezone."""
    result = TriageResult.model_validate(valid_triage_data)
    naive_dt = datetime(2026, 5, 1, 10, 0, 0)

    record = TriageRecord(
        original_message="Test message",
        result=result,
        received_at=naive_dt,
    )
    assert record.received_at.tzinfo == timezone.utc
    assert record.received_at.year == 2026
    assert record.received_at.hour == 10


def test_triage_record_invalid_status_raises(
    valid_triage_data: dict,
) -> None:
    """Verify passing an unrecognized status string raises ValidationError."""
    result = TriageResult.model_validate(valid_triage_data)

    with pytest.raises(ValidationError) as exc_info:
        TriageRecord(
            original_message="Test message",
            result=result,
            status="completed",  # Invalid
        )

    errors = exc_info.value.errors()
    assert any(err["loc"] == ("status",) for err in errors)


def test_triage_record_empty_message_raises(
    valid_triage_data: dict,
) -> None:
    """Verify original_message cannot be empty string."""
    result = TriageResult.model_validate(valid_triage_data)

    with pytest.raises(ValidationError) as exc_info:
        TriageRecord(
            original_message="",
            result=result,
        )

    errors = exc_info.value.errors()
    assert any(
        err["loc"] == ("original_message",)
        and "string_too_short" in err["type"]
        for err in errors
    )


def test_triage_record_missing_required_result() -> None:
    """Verify result is a required field of TriageRecord."""
    with pytest.raises(ValidationError) as exc_info:
        TriageRecord(original_message="Some message")

    errors = exc_info.value.errors()
    assert any(
        err["loc"] == ("result",) and err["type"] == "missing"
        for err in errors
    )
