"""LLM triage execution and clinical decision orchestration module.

Integrates rule-based red-flag screening, adaptive intake answer evaluation,
and local LLM inference via Ollama to classify incoming clinic inbox messages,
assign clinical urgency, route to appropriate staff queues, and produce
auditable triage records.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Final, Optional

import ollama
from pydantic import ValidationError

from src.intake_questions import (
    Question,
    evaluate_answers,
    to_follow_up_answers,
)
from src.prompts import PROMPTS
from src.red_flags import EMERGENCY_MESSAGE, check_red_flags
from src.routing import route
from src.schema import (
    Contact,
    FollowUpAnswer,
    Intent,
    LLMTriageOutput,
    RedFlag,
    RouteTo,
    TriageRecord,
    TriageResult,
    Urgency,
    build_triage_result,
)

logger = logging.getLogger(__name__)

MAX_VALIDATION_RETRIES: Final[int] = 2
DEFAULT_OLLAMA_TIMEOUT: Final[float] = 120.0

# Configure default Ollama HTTP client timeout to prevent indefinite hangs
if hasattr(ollama, "_client") and hasattr(ollama._client, "_client"):
    try:
        import httpx

        ollama._client._client.timeout = httpx.Timeout(DEFAULT_OLLAMA_TIMEOUT)
    except Exception:
        pass

SAFE_FALLBACK_DRAFT_REPLY: Final[str] = (
    "Thank you for contacting our clinic. We have received your message and queued "
    "it for review by our clinical staff. If you are experiencing worsening symptoms, "
    "please call our clinic directly or seek immediate medical care."
)


def _extract_response_content(response: Any) -> str:
    """Extract message content string from an Ollama chat response object or dict.

    Supports ChatResponse dataclasses, dict representations, and mocked objects.

    Args:
        response: Response returned from ollama.chat().

    Returns:
        Raw string content from the assistant message.
    """
    if isinstance(response, str):
        return response

    message_attr = getattr(response, "message", None)
    if message_attr is not None:
        content = getattr(message_attr, "content", None)
        if content is not None:
            return str(content)
        if isinstance(message_attr, dict):
            return str(message_attr.get("content", ""))

    if isinstance(response, dict):
        msg = response.get("message")
        if isinstance(msg, dict):
            return str(msg.get("content", ""))
        if msg is not None:
            content = getattr(msg, "content", None)
            if content is not None:
                return str(content)
        content = response.get("content")
        if content is not None:
            return str(content)

    return str(response or "")


def _format_user_content(message: str, follow_up: list[FollowUpAnswer]) -> str:
    """Format the patient message and any intake follow-up Q&A into prompt user content.

    Args:
        message: Raw incoming patient message text.
        follow_up: List of follow-up intake question-answer pairs.

    Returns:
        Structured string containing patient message and any follow-up Q&A.
    """
    if not follow_up:
        return message

    qa_lines = [f"Q: {item.question}\nA: {item.answer}" for item in follow_up]
    return f"{message}\n\nFollow-up Q&A:\n" + "\n\n".join(qa_lines)


def call_llm_with_meta(
    message: str,
    follow_up: list[FollowUpAnswer],
    model: str,
    prompt_version: str = "v1",
    client: Optional[Any] = None,
    timeout: float = DEFAULT_OLLAMA_TIMEOUT,
) -> tuple[LLMTriageOutput, dict[str, Any]]:
    """Call Ollama LLM to classify message and extract structured triage decisions with metadata.

    Sends the versioned system prompt and combined user message/intake answers
    to ollama.chat using temperature 0 and structured JSON schema formatting.
    Validates output with LLMTriageOutput.model_validate_json and retries up to
    2 times on validation errors. If all retries fail, returns a safe fallback
    LLMTriageOutput with urgent priority for manual clinical review.

    Args:
        message: Raw patient message text.
        follow_up: List of follow-up intake question-answer pairs.
        model: Ollama model identifier (e.g. 'llama3.2:3b').
        prompt_version: Version key for system prompt template (default: 'v1').
        client: Optional custom Ollama Client instance with configured timeout.
        timeout: Request timeout in seconds (default: 120.0).

    Returns:
        A tuple of (LLMTriageOutput, meta_dict) where meta_dict contains:
            - attempts (int): Total number of attempts executed (1 to 3).
            - used_fallback (bool): True if all retries failed and safe fallback was used.
            - latency_s (float): Total elapsed time in seconds for the call.

    Raises:
        ValueError: If prompt_version is not recognized in PROMPTS.
    """
    if prompt_version not in PROMPTS:
        raise ValueError(
            f"Prompt version '{prompt_version}' not recognized. Available: {list(PROMPTS.keys())}"
        )

    start_time = time.perf_counter()
    system_prompt = PROMPTS[prompt_version]
    user_content = _format_user_content(message, follow_up)
    chat_func = getattr(client, "chat", ollama.chat) if client is not None else ollama.chat

    max_retries = MAX_VALIDATION_RETRIES
    for attempt in range(max_retries + 1):
        attempts_count = attempt + 1
        try:
            response = chat_func(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                format=LLMTriageOutput.model_json_schema(),
                options={"temperature": 0},
            )
            raw_content = _extract_response_content(response)
            output = LLMTriageOutput.model_validate_json(raw_content)
            latency_s = time.perf_counter() - start_time
            meta = {
                "attempts": attempts_count,
                "used_fallback": False,
                "latency_s": latency_s,
            }
            return output, meta
        except (ValidationError, Exception) as err:
            logger.warning(
                "LLM triage validation failed on attempt %d of %d: %s",
                attempt + 1,
                max_retries + 1,
                err,
            )
            if attempt == max_retries:
                logger.error(
                    "All %d LLM triage attempts failed. Returning safe fallback triage output.",
                    max_retries + 1,
                )
                fallback_output = LLMTriageOutput(
                    intent=Intent.OTHER,
                    urgency=Urgency.URGENT,
                    suggested_specialty=None,
                    contact=Contact(),
                    summary="Automatic triage failed; manual review required.",
                    draft_reply=SAFE_FALLBACK_DRAFT_REPLY,
                )
                latency_s = time.perf_counter() - start_time
                meta = {
                    "attempts": max_retries + 1,
                    "used_fallback": True,
                    "latency_s": latency_s,
                }
                return fallback_output, meta

    # Defensive return for static analyzers
    latency_s = time.perf_counter() - start_time
    return (
        LLMTriageOutput(
            intent=Intent.OTHER,
            urgency=Urgency.URGENT,
            suggested_specialty=None,
            contact=Contact(),
            summary="Automatic triage failed; manual review required.",
            draft_reply=SAFE_FALLBACK_DRAFT_REPLY,
        ),
        {"attempts": max_retries + 1, "used_fallback": True, "latency_s": latency_s},
    )


def call_llm(
    message: str,
    follow_up: list[FollowUpAnswer],
    model: str,
    prompt_version: str = "v1",
) -> LLMTriageOutput:
    """Call Ollama LLM to classify message and extract structured triage decisions.

    Delegates to call_llm_with_meta and returns the validated LLMTriageOutput.

    Args:
        message: Raw patient message text.
        follow_up: List of follow-up intake question-answer pairs.
        model: Ollama model identifier (e.g. 'llama3.2:3b').
        prompt_version: Version key for system prompt template (default: 'v1').

    Returns:
        Validated LLMTriageOutput model instance.
    """
    output, _ = call_llm_with_meta(
        message=message,
        follow_up=follow_up,
        model=model,
        prompt_version=prompt_version,
    )
    return output


def triage_message_with_meta(
    message: str,
    answers: Optional[list[tuple[Question, str]] | list[FollowUpAnswer]] = None,
    model: str = "llama3.2:3b",
    prompt_version: str = "v1",
    client: Optional[Any] = None,
    timeout: float = DEFAULT_OLLAMA_TIMEOUT,
) -> tuple[TriageRecord, float, dict[str, Any]]:
    """Execute end-to-end clinical triage returning the record, elapsed time, and execution metadata.

    Performs deterministic red-flag screening on the raw message and intake
    answers. If any red flag triggers, bypasses LLM inference entirely and
    immediately constructs an emergency triage record. Otherwise invokes the LLM.

    Args:
        message: Raw incoming patient message text.
        answers: Optional intake questions and answers.
        model: Ollama model identifier to use for LLM classification.
        prompt_version: Version identifier for system prompt template.
        client: Optional custom Ollama Client instance.
        timeout: Request timeout in seconds.

    Returns:
        A tuple of (TriageRecord, elapsed_seconds: float, meta: dict).
    """
    start_time = time.perf_counter()

    # 1. Evaluate answers and convert to FollowUpAnswer list
    follow_up_list: list[FollowUpAnswer] = []
    answers_red_flag = RedFlag(is_red_flag=False, reason=None)

    if answers:
        if isinstance(answers[0], tuple):
            answers_red_flag = evaluate_answers(answers)  # type: ignore[arg-type]
            follow_up_list = to_follow_up_answers(answers)  # type: ignore[arg-type]
        elif isinstance(answers[0], FollowUpAnswer):
            follow_up_list = list(answers)  # type: ignore[arg-type]
        else:
            answers_red_flag = evaluate_answers(answers)  # type: ignore[arg-type]
            follow_up_list = to_follow_up_answers(answers)  # type: ignore[arg-type]

    # 2. Check message for emergency red flags
    message_red_flag = check_red_flags(message)

    # 3. Emergency bypass if either check triggered a red flag
    if message_red_flag.is_red_flag or answers_red_flag.is_red_flag:
        if message_red_flag.is_red_flag and answers_red_flag.is_red_flag:
            combined_reason = f"{message_red_flag.reason}; {answers_red_flag.reason}"
            active_red_flag = RedFlag(is_red_flag=True, reason=combined_reason)
        elif message_red_flag.is_red_flag:
            active_red_flag = message_red_flag
        else:
            active_red_flag = answers_red_flag

        emergency_result = TriageResult(
            intent=Intent.OTHER,
            urgency=Urgency.EMERGENCY,
            route_to=RouteTo.ON_CALL_CLINICIAN,
            suggested_specialty=None,
            red_flag=active_red_flag,
            follow_up=follow_up_list,
            contact=Contact(),
            summary=f"Emergency red flag detected: {active_red_flag.reason}",
            draft_reply=EMERGENCY_MESSAGE,
        )
        elapsed = time.perf_counter() - start_time
        record = TriageRecord(
            original_message=message,
            result=emergency_result,
        )
        meta = {
            "attempts": 0,
            "used_fallback": False,
            "latency_s": elapsed,
        }
        return record, elapsed, meta

    # 4. Standard path: Call LLM, route, and build consolidated triage result
    active_red_flag = RedFlag(is_red_flag=False, reason=None)
    llm_output, meta = call_llm_with_meta(
        message=message,
        follow_up=follow_up_list,
        model=model,
        prompt_version=prompt_version,
        client=client,
        timeout=timeout,
    )
    routed_to = route(intent=llm_output.intent, urgency=llm_output.urgency)
    triage_result = build_triage_result(
        llm_output=llm_output,
        route_to=routed_to,
        red_flag=active_red_flag,
        follow_up=follow_up_list,
    )

    elapsed = time.perf_counter() - start_time
    record = TriageRecord(
        original_message=message,
        result=triage_result,
    )
    return record, elapsed, meta


def triage_message(
    message: str,
    answers: Optional[list[tuple[Question, str]] | list[FollowUpAnswer]] = None,
    model: str = "llama3.2:3b",
    prompt_version: str = "v1",
    client: Optional[Any] = None,
    timeout: float = DEFAULT_OLLAMA_TIMEOUT,
) -> tuple[TriageRecord, float]:
    """Execute end-to-end clinical triage for an incoming patient message.

    Performs deterministic red-flag screening on the raw message and intake
    answers. If any red flag triggers, bypasses LLM inference entirely and
    immediately constructs an emergency triage record routed to the on-call
    clinician with standard emergency guidance.

    If no red flags are found, invokes the LLM triage extraction, applies
    staff routing rules, and creates a consolidated TriageRecord.

    Args:
        message: Raw incoming patient message text.
        answers: Optional intake questions and answers, either as (Question, str)
            tuples or validated FollowUpAnswer models.
        model: Ollama model identifier to use for LLM classification.
        prompt_version: Version identifier for system prompt template.
        client: Optional custom Ollama Client instance.
        timeout: Request timeout in seconds.

    Returns:
        A tuple of (TriageRecord, elapsed_seconds: float).
    """
    record, elapsed, _ = triage_message_with_meta(
        message=message,
        answers=answers,
        model=model,
        prompt_version=prompt_version,
        client=client,
        timeout=timeout,
    )
    return record, elapsed


__all__ = [
    "DEFAULT_OLLAMA_TIMEOUT",
    "MAX_VALIDATION_RETRIES",
    "SAFE_FALLBACK_DRAFT_REPLY",
    "call_llm",
    "call_llm_with_meta",
    "triage_message",
    "triage_message_with_meta",
]
