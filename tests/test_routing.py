"""Tests for src/routing.py deterministic clinic triage message routing.

Covers:
- All 24 Intent x Urgency combinations with an explicit expected table
- String-based arguments acceptance
- Invalid intent/urgency rejection
"""

import pytest

from src.routing import route
from src.schema import Intent, RouteTo, Urgency


# Explicit table defining the expected RouteTo destination for all 24
# intent (6) x urgency (4) combinations.
ROUTING_COMBINATIONS = [
    # ------------------------------------------------------------------
    # 1. Appointment
    # ------------------------------------------------------------------
    (Intent.APPOINTMENT, Urgency.ROUTINE, RouteTo.FRONT_DESK),
    (Intent.APPOINTMENT, Urgency.SOON, RouteTo.ON_CALL_CLINICIAN),
    (Intent.APPOINTMENT, Urgency.URGENT, RouteTo.ON_CALL_CLINICIAN),
    (Intent.APPOINTMENT, Urgency.EMERGENCY, RouteTo.ON_CALL_CLINICIAN),
    # ------------------------------------------------------------------
    # 2. Refill
    # ------------------------------------------------------------------
    (Intent.REFILL, Urgency.ROUTINE, RouteTo.PHARMACIST_OR_MA),
    (Intent.REFILL, Urgency.SOON, RouteTo.PHARMACIST_OR_MA),
    (Intent.REFILL, Urgency.URGENT, RouteTo.ON_CALL_CLINICIAN),
    (Intent.REFILL, Urgency.EMERGENCY, RouteTo.ON_CALL_CLINICIAN),
    # ------------------------------------------------------------------
    # 3. Test Results
    # ------------------------------------------------------------------
    (Intent.TEST_RESULTS, Urgency.ROUTINE, RouteTo.FRONT_DESK),
    (Intent.TEST_RESULTS, Urgency.SOON, RouteTo.ON_CALL_CLINICIAN),
    (Intent.TEST_RESULTS, Urgency.URGENT, RouteTo.ON_CALL_CLINICIAN),
    (Intent.TEST_RESULTS, Urgency.EMERGENCY, RouteTo.ON_CALL_CLINICIAN),
    # ------------------------------------------------------------------
    # 4. Administrative
    # ------------------------------------------------------------------
    (Intent.ADMINISTRATIVE, Urgency.ROUTINE, RouteTo.FRONT_DESK),
    (Intent.ADMINISTRATIVE, Urgency.SOON, RouteTo.FRONT_DESK),
    (Intent.ADMINISTRATIVE, Urgency.URGENT, RouteTo.ON_CALL_CLINICIAN),
    (Intent.ADMINISTRATIVE, Urgency.EMERGENCY, RouteTo.ON_CALL_CLINICIAN),
    # ------------------------------------------------------------------
    # 5. General Query
    # ------------------------------------------------------------------
    (Intent.GENERAL_QUERY, Urgency.ROUTINE, RouteTo.FRONT_DESK),
    (Intent.GENERAL_QUERY, Urgency.SOON, RouteTo.ON_CALL_CLINICIAN),
    (Intent.GENERAL_QUERY, Urgency.URGENT, RouteTo.ON_CALL_CLINICIAN),
    (Intent.GENERAL_QUERY, Urgency.EMERGENCY, RouteTo.ON_CALL_CLINICIAN),
    # ------------------------------------------------------------------
    # 6. Other
    # ------------------------------------------------------------------
    (Intent.OTHER, Urgency.ROUTINE, RouteTo.FRONT_DESK),
    (Intent.OTHER, Urgency.SOON, RouteTo.ON_CALL_CLINICIAN),
    (Intent.OTHER, Urgency.URGENT, RouteTo.ON_CALL_CLINICIAN),
    (Intent.OTHER, Urgency.EMERGENCY, RouteTo.ON_CALL_CLINICIAN),
]


def test_routing_table_completeness() -> None:
    """Verify that ROUTING_COMBINATIONS covers every possible (Intent, Urgency) pair."""
    all_pairs = {(intent, urgency) for intent in Intent for urgency in Urgency}
    tested_pairs = {(intent, urgency) for intent, urgency, _ in ROUTING_COMBINATIONS}

    assert len(ROUTING_COMBINATIONS) == 24
    assert tested_pairs == all_pairs


@pytest.mark.parametrize(("intent", "urgency", "expected_route"), ROUTING_COMBINATIONS)
def test_all_24_routing_combinations(
    intent: Intent, urgency: Urgency, expected_route: RouteTo
) -> None:
    """Verify deterministic routing for all 24 intent x urgency combinations."""
    actual_route = route(intent, urgency)
    assert actual_route == expected_route


@pytest.mark.parametrize(("intent", "urgency", "expected_route"), ROUTING_COMBINATIONS)
def test_routing_with_string_arguments(
    intent: Intent, urgency: Urgency, expected_route: RouteTo
) -> None:
    """Verify route function accepts raw string enum values."""
    actual_route = route(intent.value, urgency.value)
    assert actual_route == expected_route


def test_invalid_arguments_raise_value_error() -> None:
    """Verify passing invalid intent or urgency strings raises ValueError."""
    with pytest.raises(ValueError):
        route("invalid_intent", Urgency.ROUTINE)

    with pytest.raises(ValueError):
        route(Intent.REFILL, "invalid_urgency")
