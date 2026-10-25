"""Unit tests for src/mock_llm.py deterministic keyword-based mock LLM.

Covers:
- Keyword intent classification:
  - refill -> refill
  - appointment, book, reschedule -> appointment
  - result, test -> test_results
  - bill, paperwork, sick note -> administrative
  - fallback -> general_query
- Keyword urgency classification:
  - fever, pain, worse -> soon
  - fallback -> routine
- LLMTriageOutput validity and schema compliance
- Metadata generation (latency, is_mock flag)
- Integration with triage_message(use_mock=True)
"""

from __future__ import annotations

import pytest

from src.mock_llm import mock_call_llm, mock_classify_message
from src.schema import Intent, LLMTriageOutput, Urgency
from src.triage import triage_message


@pytest.mark.parametrize(
    "message,expected_intent,expected_urgency",
    [
        ("Need a refill of my blood pressure medication.", Intent.REFILL, Urgency.ROUTINE),
        ("Can I book a visit with Dr. Gomez?", Intent.APPOINTMENT, Urgency.ROUTINE),
        ("Please reschedule my dermatology checkup.", Intent.APPOINTMENT, Urgency.ROUTINE),
        ("I need to make an appointment.", Intent.APPOINTMENT, Urgency.ROUTINE),
        ("Checking in on my lab test results.", Intent.TEST_RESULTS, Urgency.ROUTINE),
        ("Has the radiologist looked at my x-ray result?", Intent.TEST_RESULTS, Urgency.ROUTINE),
        ("I need an itemized bill for my insurance.", Intent.ADMINISTRATIVE, Urgency.ROUTINE),
        ("Please complete this disability paperwork.", Intent.ADMINISTRATIVE, Urgency.ROUTINE),
        ("I need a doctor's sick note for work.", Intent.ADMINISTRATIVE, Urgency.ROUTINE),
        ("What are your weekend hours?", Intent.GENERAL_QUERY, Urgency.ROUTINE),
        ("My back pain is getting worse.", Intent.GENERAL_QUERY, Urgency.SOON),
        ("I have a high fever of 102.", Intent.GENERAL_QUERY, Urgency.SOON),
        ("Need to book a visit for severe ear pain.", Intent.APPOINTMENT, Urgency.SOON),
        ("Need a refill because symptoms got worse.", Intent.REFILL, Urgency.SOON),
    ],
)
def test_mock_classify_message(message: str, expected_intent: Intent, expected_urgency: Urgency):
    """Test deterministic keyword classification for intent and urgency."""
    intent, urgency = mock_classify_message(message)
    assert intent == expected_intent
    assert urgency == expected_urgency


def test_mock_call_llm_output_validity():
    """mock_call_llm returns a valid LLMTriageOutput and proper metadata."""
    msg = "I need to reschedule my physical. Contact: John Smith, 555-0144, john@example.com."
    output, meta = mock_call_llm(msg)

    assert isinstance(output, LLMTriageOutput)
    assert output.intent == Intent.APPOINTMENT
    assert output.urgency == Urgency.ROUTINE
    assert output.contact.name == "John Smith"
    assert output.contact.phone == "555-0144"
    assert output.contact.email == "john@example.com"
    assert len(output.summary) > 0
    assert len(output.draft_reply) > 0

    assert meta["is_mock"] is True
    assert meta["attempts"] == 1
    assert meta["latency_s"] >= 0.0


def test_triage_message_with_use_mock_integration():
    """triage_message with use_mock=True produces a persistent TriageRecord without calling Ollama."""
    msg = "I need a refill for metformin. Contact: Arthur Davis."
    record, latency_s = triage_message(message=msg, use_mock=True)

    assert record.result.intent == Intent.REFILL
    assert record.result.urgency == Urgency.ROUTINE
    assert record.result.route_to.value == "pharmacist_or_ma"
    assert latency_s >= 0.0
