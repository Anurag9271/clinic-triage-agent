"""Deterministic message routing module for the clinic triage pipeline.

Maps message intent and clinical urgency to the appropriate staff queue
based on deterministic clinical workflow priority rules.
"""

from __future__ import annotations

from src.schema import Intent, RouteTo, Urgency


def route(intent: Intent | str, urgency: Urgency | str) -> RouteTo:
    """Deterministically route a message to the appropriate staff queue.

    Rules applied in priority order:
    1. Urgency is emergency or urgent -> ON_CALL_CLINICIAN
    2. Intent is refill -> PHARMACIST_OR_MA
    3. Intent is administrative -> FRONT_DESK
    4. Intent is appointment, test_results, general_query, or other:
       - routine -> FRONT_DESK
       - soon -> ON_CALL_CLINICIAN

    Args:
        intent: Primary categorized intent of the message.
        urgency: Assessed clinical or operational urgency level.

    Returns:
        The designated RouteTo staff queue enum.
    """
    if isinstance(intent, str) and not isinstance(intent, Intent):
        intent = Intent(intent)
    if isinstance(urgency, str) and not isinstance(urgency, Urgency):
        urgency = Urgency(urgency)

    # Rule 1: Emergency or urgent clinical priority
    if urgency in (Urgency.EMERGENCY, Urgency.URGENT):
        return RouteTo.ON_CALL_CLINICIAN

    # Rule 2: Prescription refills
    if intent == Intent.REFILL:
        return RouteTo.PHARMACIST_OR_MA

    # Rule 3: Administrative paperwork, sick notes, billing
    if intent == Intent.ADMINISTRATIVE:
        return RouteTo.FRONT_DESK

    # Rule 4: Appointment, test results, general query, or other
    if urgency == Urgency.SOON:
        return RouteTo.ON_CALL_CLINICIAN

    return RouteTo.FRONT_DESK


__all__ = ["route"]
