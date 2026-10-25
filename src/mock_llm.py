"""Deterministic mock LLM triage classification for fast demo mode.

Provides instant, keyword-based LLMTriageOutput generation without invoking
local Ollama models:
- refill -> intent refill
- appointment/book/reschedule -> appointment
- result/test -> test_results
- bill/paperwork/sick note -> administrative
- fever/pain/worse -> urgency soon
- otherwise general_query with urgency routine
"""

from __future__ import annotations

import re
import time
from typing import Any, Final, Optional

from src.schema import (
    Contact,
    FollowUpAnswer,
    Intent,
    LLMTriageOutput,
    Urgency,
)


def mock_classify_message(message: str) -> tuple[Intent, Urgency]:
    """Classify intent and urgency from message text using deterministic keyword rules.

    Keyword rules:
    - refill -> Intent.REFILL
    - appointment, book, reschedule -> Intent.APPOINTMENT
    - result, test -> Intent.TEST_RESULTS
    - bill, paperwork, sick note -> Intent.ADMINISTRATIVE
    - otherwise -> Intent.GENERAL_QUERY

    Urgency rules:
    - fever, pain, worse -> Urgency.SOON
    - otherwise -> Urgency.ROUTINE

    Args:
        message: Raw or normalized patient message text.

    Returns:
        A tuple of (Intent, Urgency).
    """
    text = message.lower()

    # Intent classification
    if "refill" in text:
        intent = Intent.REFILL
    elif "appointment" in text or "book" in text or "reschedule" in text:
        intent = Intent.APPOINTMENT
    elif "result" in text or "test" in text:
        intent = Intent.TEST_RESULTS
    elif "bill" in text or "paperwork" in text or "sick note" in text:
        intent = Intent.ADMINISTRATIVE
    else:
        intent = Intent.GENERAL_QUERY

    # Urgency classification
    if "fever" in text or "pain" in text or "worse" in text:
        urgency = Urgency.SOON
    else:
        urgency = Urgency.ROUTINE

    return intent, urgency


def mock_call_llm(
    message: str,
    follow_up: Optional[list[FollowUpAnswer]] = None,
    model: str = "mock",
    prompt_version: str = "v1",
) -> tuple[LLMTriageOutput, dict[str, Any]]:
    """Generate a deterministic mock LLMTriageOutput conforming to schema.

    Args:
        message: Raw patient message text.
        follow_up: Optional follow-up question-answer pairs.
        model: Model identifier (ignored in mock).
        prompt_version: Prompt version (ignored in mock).

    Returns:
        Tuple of (LLMTriageOutput, meta dict).
    """
    start_time = time.perf_counter()
    intent, urgency = mock_classify_message(message)

    # Basic contact extraction (regex matching for phone/email if present)
    name = None
    phone = None
    email = None

    phone_match = re.search(r"\b(\d{3}-\d{4}|\d{3}-\d{3}-\d{4})\b", message)
    if phone_match:
        phone = phone_match.group(1)

    email_match = re.search(r"\b([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)\b", message)
    if email_match:
        email = email_match.group(1)

    name_match = re.search(
        r"(?:Contact|Name|Thanks|Regards)[,:]\s*([A-Z][a-z]+ [A-Z][a-z]+)", message
    )
    if name_match:
        name = name_match.group(1)

    contact = Contact(name=name, phone=phone, email=email)

    summary = f"Patient submitted a {intent.value} inquiry with {urgency.value} urgency."
    draft_reply = (
        f"Thank you for contacting the clinic regarding your {intent.value.replace('_', ' ')}. "
        f"Our team has received your message and will review it promptly."
    )

    output = LLMTriageOutput(
        intent=intent,
        urgency=urgency,
        suggested_specialty=None,
        contact=contact,
        summary=summary,
        draft_reply=draft_reply,
    )
    elapsed = time.perf_counter() - start_time
    meta = {
        "attempts": 1,
        "used_fallback": False,
        "latency_s": elapsed,
        "is_mock": True,
    }
    return output, meta


__all__ = [
    "mock_call_llm",
    "mock_classify_message",
]
