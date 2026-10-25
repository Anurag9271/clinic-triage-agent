"""Unit tests for the LLM triage pipeline (src/triage.py).

Verifies LLM calling with ollama.chat mocking, Pydantic v2 validation retries,
safe fallback behavior, and emergency red-flag short-circuiting.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.intake_questions import Question
from src.prompts import PROMPT_V1
from src.red_flags import EMERGENCY_MESSAGE
from src.schema import (
    Contact,
    FollowUpAnswer,
    Intent,
    LLMTriageOutput,
    RouteTo,
    TriageRecord,
    Urgency,
)
from src.triage import (
    MAX_VALIDATION_RETRIES,
    call_llm,
    triage_message,
)


@pytest.fixture
def valid_llm_dict() -> dict:
    """Fixture providing a dictionary conforming to LLMTriageOutput schema."""
    return {
        "intent": "appointment",
        "urgency": "routine",
        "suggested_specialty": "Dermatology",
        "contact": {
            "name": "Jane Doe",
            "phone": "555-0199",
            "email": "jane@example.com",
        },
        "summary": "Patient Jane Doe requests a routine dermatology skin check appointment.",
        "draft_reply": "Hello Jane, we would be happy to schedule your skin check. What day works best?",
    }


@pytest.fixture
def valid_llm_json(valid_llm_dict: dict) -> str:
    """Fixture providing a JSON string conforming to LLMTriageOutput schema."""
    return json.dumps(valid_llm_dict)


def _make_chat_response(content: str) -> MagicMock:
    """Helper to create a mock ChatResponse object mimicking ollama.chat return."""
    mock_resp = MagicMock()
    mock_resp.message.content = content
    return mock_resp


# =====================================================================
# Tests for call_llm
# =====================================================================


def test_call_llm_valid_json(valid_llm_json: str):
    """Test call_llm successfully parses valid JSON on the first attempt."""
    with patch("ollama.chat") as mock_chat:
        mock_chat.return_value = _make_chat_response(valid_llm_json)

        output = call_llm(
            message="I would like to schedule a skin check next Tuesday.",
            follow_up=[],
            model="llama3.2:3b",
            prompt_version="v1",
        )

        assert mock_chat.call_count == 1
        assert isinstance(output, LLMTriageOutput)
        assert output.intent == Intent.APPOINTMENT
        assert output.urgency == Urgency.ROUTINE
        assert output.suggested_specialty == "Dermatology"
        assert output.contact.name == "Jane Doe"
        assert output.contact.phone == "555-0199"

        # Verify ollama.chat arguments
        call_kwargs = mock_chat.call_args.kwargs
        assert call_kwargs["model"] == "llama3.2:3b"
        assert call_kwargs["options"]["temperature"] == 0
        assert call_kwargs["options"]["num_predict"] == 300
        assert call_kwargs.get("keep_alive") == "30m" or call_kwargs["options"].get("keep_alive") == "30m"
        assert call_kwargs["format"] == LLMTriageOutput.model_json_schema()
        messages = call_kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == PROMPT_V1
        assert messages[1]["role"] == "user"
        assert "skin check" in messages[1]["content"]


def test_call_llm_with_dict_response(valid_llm_json: str):
    """Test call_llm correctly handles dict-like response from ollama.chat."""
    with patch("ollama.chat") as mock_chat:
        mock_chat.return_value = {"message": {"content": valid_llm_json}}

        output = call_llm(
            message="I need an appointment.",
            follow_up=[],
            model="llama3.2:3b",
        )

        assert mock_chat.call_count == 1
        assert output.intent == Intent.APPOINTMENT


def test_call_llm_includes_follow_up_in_user_prompt(valid_llm_json: str):
    """Test that follow-up Q&A pairs are formatted into user content for the LLM."""
    follow_up = [
        FollowUpAnswer(
            question="Do you have shortness of breath?",
            answer="No, breathing is fine.",
        ),
        FollowUpAnswer(
            question="How many days have you had the cough?",
            answer="About 4 days.",
        ),
    ]

    with patch("ollama.chat") as mock_chat:
        mock_chat.return_value = _make_chat_response(valid_llm_json)

        call_llm(
            message="I have had a lingering cough.",
            follow_up=follow_up,
            model="llama3.2:3b",
        )

        call_kwargs = mock_chat.call_args.kwargs
        user_content = call_kwargs["messages"][1]["content"]
        assert "I have had a lingering cough." in user_content
        assert "Do you have shortness of breath?" in user_content
        assert "No, breathing is fine." in user_content
        assert "About 4 days." in user_content


def test_call_llm_invalid_json_then_valid_on_retry(valid_llm_json: str):
    """Test call_llm retries upon encountering invalid JSON and succeeds on attempt 2."""
    malformed_json = '{"intent": "appointment", "urgency": "invalid_urgency"}'

    with patch("ollama.chat") as mock_chat:
        mock_chat.side_effect = [
            _make_chat_response(malformed_json),
            _make_chat_response(valid_llm_json),
        ]

        output = call_llm(
            message="Need an appointment.",
            follow_up=[],
            model="llama3.2:3b",
        )

        assert mock_chat.call_count == 2
        assert output.intent == Intent.APPOINTMENT
        assert output.urgency == Urgency.ROUTINE


def test_call_llm_syntax_error_json_then_valid_on_retry(valid_llm_json: str):
    """Test call_llm retries upon unparseable JSON text and succeeds on retry."""
    broken_syntax = "Not valid JSON at all!"

    with patch("ollama.chat") as mock_chat:
        mock_chat.side_effect = [
            _make_chat_response(broken_syntax),
            _make_chat_response(valid_llm_json),
        ]

        output = call_llm(
            message="Need an appointment.",
            follow_up=[],
            model="llama3.2:3b",
        )

        assert mock_chat.call_count == 2
        assert output.intent == Intent.APPOINTMENT


def test_call_llm_repeated_failure_returns_safe_fallback():
    """Test call_llm exhausts all retries and returns safe fallback output."""
    invalid_json = '{"broken": "missing fields"}'

    with patch("ollama.chat") as mock_chat:
        mock_chat.return_value = _make_chat_response(invalid_json)

        output = call_llm(
            message="Help with my bill.",
            follow_up=[],
            model="llama3.2:3b",
        )

        # 1 initial try + 2 retries = 3 calls total
        assert mock_chat.call_count == MAX_VALIDATION_RETRIES + 1
        assert output.intent == Intent.OTHER
        assert output.urgency == Urgency.URGENT
        assert output.summary == "Automatic triage failed; manual review required."
        assert len(output.draft_reply) > 0
        assert output.contact.name is None


def test_call_llm_exception_in_chat_exhausts_retries_and_falls_back():
    """Test call_llm handles exceptions raised directly by ollama.chat."""
    with patch("ollama.chat") as mock_chat:
        mock_chat.side_effect = RuntimeError("Connection to Ollama daemon refused")

        output = call_llm(
            message="Check my prescription.",
            follow_up=[],
            model="llama3.2:3b",
        )

        assert mock_chat.call_count == MAX_VALIDATION_RETRIES + 1
        assert output.intent == Intent.OTHER
        assert output.urgency == Urgency.URGENT
        assert output.summary == "Automatic triage failed; manual review required."


def test_call_llm_unknown_prompt_version():
    """Test call_llm raises ValueError when an unknown prompt version is specified."""
    with pytest.raises(ValueError, match="Prompt version 'v999' not recognized"):
        call_llm(
            message="Hello",
            follow_up=[],
            model="llama3.2:3b",
            prompt_version="v999",
        )


# =====================================================================
# Tests for triage_message
# =====================================================================


def test_triage_message_standard_flow(valid_llm_json: str):
    """Test standard end-to-end flow with valid message and no red flags."""
    with patch("ollama.chat") as mock_chat:
        mock_chat.return_value = _make_chat_response(valid_llm_json)

        message = "Hi, I would like to schedule an annual dermatology checkup."
        record, elapsed = triage_message(message=message)

        assert mock_chat.call_count == 1
        assert isinstance(record, TriageRecord)
        assert isinstance(elapsed, float)
        assert elapsed >= 0.0

        assert record.original_message == message
        assert record.result.intent == Intent.APPOINTMENT
        assert record.result.urgency == Urgency.ROUTINE
        assert record.result.route_to == RouteTo.FRONT_DESK
        assert record.result.red_flag.is_red_flag is False
        assert record.result.red_flag.reason is None
        assert record.result.suggested_specialty == "Dermatology"
        assert record.result.contact.name == "Jane Doe"


def test_triage_message_red_flag_message_skips_llm():
    """Test that an emergency message red flag skips the LLM call entirely."""
    with patch("ollama.chat") as mock_chat:
        emergency_msg = "Help, my chest feels heavy and I have crushing chest pain!"
        record, elapsed = triage_message(message=emergency_msg)

        # Critical clinical safety assertion: LLM must NOT be called
        mock_chat.assert_not_called()

        assert isinstance(record, TriageRecord)
        assert isinstance(elapsed, float)
        assert record.original_message == emergency_msg
        assert record.result.urgency == Urgency.EMERGENCY
        assert record.result.route_to == RouteTo.ON_CALL_CLINICIAN
        assert record.result.draft_reply == EMERGENCY_MESSAGE
        assert record.result.red_flag.is_red_flag is True
        assert "cardiac" in (record.result.red_flag.reason or "").lower()


def test_triage_message_red_flag_answer_skips_llm():
    """Test that a red-flag intake answer skips the LLM call entirely."""
    cough_blood_q = Question(
        id="cough_2",
        group="cough",
        text="Are you coughing up blood or pink frothy mucus?",
        answer_type="yes_no",
        red_flag_if="yes",
        reason="Concern for massive hemoptysis or pulmonary hemorrhage.",
    )

    answers = [(cough_blood_q, "Yes")]

    with patch("ollama.chat") as mock_chat:
        benign_msg = "I have had a cough for 3 days."
        record, elapsed = triage_message(message=benign_msg, answers=answers)

        # Critical clinical safety assertion: LLM must NOT be called
        mock_chat.assert_not_called()

        assert isinstance(record, TriageRecord)
        assert record.original_message == benign_msg
        assert record.result.urgency == Urgency.EMERGENCY
        assert record.result.route_to == RouteTo.ON_CALL_CLINICIAN
        assert record.result.draft_reply == EMERGENCY_MESSAGE
        assert record.result.red_flag.is_red_flag is True
        assert "cough_2" in (record.result.red_flag.reason or "")
        assert len(record.result.follow_up) == 1
        assert record.result.follow_up[0].answer == "Yes"


def test_triage_message_both_message_and_answer_red_flags():
    """Test emergency bypass when both the message and intake answers are red flags."""
    sob_q = Question(
        id="resp_1",
        group="cough",
        text="Are you having severe shortness of breath?",
        answer_type="yes_no",
        red_flag_if="yes",
        reason="Severe respiratory distress.",
    )
    answers = [(sob_q, "yes")]

    with patch("ollama.chat") as mock_chat:
        severe_msg = "My face is drooping and arm is numb."
        record, elapsed = triage_message(message=severe_msg, answers=answers)

        mock_chat.assert_not_called()
        assert record.result.urgency == Urgency.EMERGENCY
        assert record.result.route_to == RouteTo.ON_CALL_CLINICIAN
        assert record.result.draft_reply == EMERGENCY_MESSAGE
        assert record.result.red_flag.is_red_flag is True
        reason = record.result.red_flag.reason or ""
        assert "neurological" in reason
        assert "resp_1" in reason


def test_triage_message_non_red_flag_answers_proceeds_to_llm(valid_llm_json: str):
    """Test that benign intake answers pass through to the LLM normally."""
    cough_q = Question(
        id="cough_1",
        group="cough",
        text="Do you have shortness of breath or difficulty breathing?",
        answer_type="yes_no",
        red_flag_if="yes",
        reason="Shortness of breath.",
    )
    # Patient answered "no" - not a red flag
    answers = [(cough_q, "No, breathing is normal")]

    with patch("ollama.chat") as mock_chat:
        mock_chat.return_value = _make_chat_response(valid_llm_json)

        message = "Just a mild cough since yesterday."
        record, elapsed = triage_message(message=message, answers=answers)

        assert mock_chat.call_count == 1
        assert record.result.red_flag.is_red_flag is False
        assert len(record.result.follow_up) == 1
        assert record.result.follow_up[0].question == cough_q.text
        assert record.result.follow_up[0].answer == "No, breathing is normal"


def test_triage_message_with_follow_up_answer_models(valid_llm_json: str):
    """Test triage_message accepts pre-converted FollowUpAnswer models."""
    answers = [
        FollowUpAnswer(
            question="What pharmacy do you prefer?",
            answer="Main St Pharmacy",
        )
    ]

    with patch("ollama.chat") as mock_chat:
        mock_chat.return_value = _make_chat_response(valid_llm_json)

        record, elapsed = triage_message(
            message="Please refill my lisinopril.",
            answers=answers,
        )

        assert mock_chat.call_count == 1
        assert len(record.result.follow_up) == 1
        assert record.result.follow_up[0].question == "What pharmacy do you prefer?"


def test_triage_message_llm_failure_routes_fallback_to_on_call():
    """Test that an LLM failure falls back to urgent priority routed to on-call clinician."""
    with patch("ollama.chat") as mock_chat:
        mock_chat.return_value = _make_chat_response("not valid json")

        record, elapsed = triage_message(message="General inquiry about clinic hours.")

        assert mock_chat.call_count == 3
        assert record.result.intent == Intent.OTHER
        assert record.result.urgency == Urgency.URGENT
        # Rule: urgency urgent -> ON_CALL_CLINICIAN
        assert record.result.route_to == RouteTo.ON_CALL_CLINICIAN
        assert "Automatic triage failed" in record.result.summary
