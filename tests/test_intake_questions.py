"""Tests for src/intake_questions.py adaptive symptom follow-up screening.

Covers:
- All 10 symptom groups detected by at least 2 distinct phrasings
- Enforced question capping (default 3, custom limits)
- Red-flag screening questions ordered first
- No-match and non-clinical messages returning empty lists
- All affirmative 'yes' variants triggering red flags
- Negative ('no') and unclear answers not triggering red flags
- to_follow_up_answers output producing valid FollowUpAnswer models
- Question model validation and SYMPTOM_QUESTIONS structure
"""

import pytest
from pydantic import ValidationError

from src.intake_questions import (
    SYMPTOM_PATTERNS,
    SYMPTOM_QUESTIONS,
    YES_VARIANTS,
    Question,
    evaluate_answers,
    get_follow_up_questions,
    to_follow_up_answers,
)
from src.schema import FollowUpAnswer, RedFlag


# =====================================================================
# 1. Symptom Group Detection (At least 2 phrasings per group)
# =====================================================================

GROUP_PHRASES = [
    ("sore_throat", "I have had a severe sore throat since yesterday."),
    ("sore_throat", "My throat hurts terribly whenever I try to swallow."),
    ("cough", "I cannot stop coughing today and my chest feels irritated."),
    ("cough", "Suffering from a dry hacking cough for four days."),
    ("fever", "I woke up running a high temperature with severe chills."),
    ("fever", "Feeling very feverish and burning up all night."),
    ("headache", "Terrible throbbing headache right behind my eyes."),
    ("headache", "My head hurts constantly and won't go away with aspirin."),
    ("abdominal_pain", "Experiencing a severe stomach ache and sharp cramps."),
    ("abdominal_pain", "I have bad tummy pain and belly cramps since this morning."),
    ("rash", "Developed a red itchy rash all over my arms."),
    ("rash", "Woke up with hives and sudden skin breakout on my neck."),
    ("back_pain", "Suffering from severe lower back pain after lifting a box."),
    ("back_pain", "My back hurts so badly I can barely sit down."),
    ("dizziness", "I feel dizzy every time I get out of bed."),
    ("dizziness", "Experiencing lightheadedness and feeling faint when walking."),
    ("injury", "I twisted my ankle playing soccer this afternoon."),
    ("injury", "I had an accident at home and hurt my knee badly."),
    ("medication_refill", "I need a refill for my blood pressure pills."),
    ("medication_refill", "Running out of medication and need a prescription renewal."),
]


@pytest.mark.parametrize(("expected_group", "phrase"), GROUP_PHRASES)
def test_group_detected_by_phrasings(expected_group: str, phrase: str) -> None:
    """Verify each of the 10 symptom groups is detected by at least 2 distinct phrasings."""
    questions = get_follow_up_questions(phrase)
    assert len(questions) > 0, f"No questions returned for phrase: '{phrase}'"
    # Ensure at least one returned question belongs to the expected group
    assert any(q.group == expected_group for q in questions), (
        f"Expected group '{expected_group}' not found in returned questions: "
        f"{[q.group for q in questions]}"
    )


def test_all_ten_groups_represented_in_patterns() -> None:
    """Verify that all 10 required clinical groups exist in SYMPTOM_PATTERNS and SYMPTOM_QUESTIONS."""
    expected_groups = {
        "sore_throat",
        "cough",
        "fever",
        "headache",
        "abdominal_pain",
        "rash",
        "back_pain",
        "dizziness",
        "injury",
        "medication_refill",
    }
    assert set(SYMPTOM_PATTERNS.keys()) == expected_groups
    assert set(SYMPTOM_QUESTIONS.keys()) == expected_groups

    for group, questions in SYMPTOM_QUESTIONS.items():
        assert 2 <= len(questions) <= 3, (
            f"Group '{group}' must have 2 to 3 questions, found {len(questions)}"
        )


# =====================================================================
# 2. Question Capping (Max Questions)
# =====================================================================


def test_question_capping_default_three() -> None:
    """Verify that when multiple symptom groups match, questions are capped at default 3."""
    # Matches sore_throat, cough, and fever (3 groups with ~9 total questions)
    multi_symptom_message = (
        "I have a sore throat, bad hacking cough, and high feverish temperature."
    )
    questions = get_follow_up_questions(multi_symptom_message)
    assert len(questions) == 3


@pytest.mark.parametrize("cap", [1, 2, 4, 5])
def test_question_capping_custom_limits(cap: int) -> None:
    """Verify custom max_questions limits are strictly honored."""
    multi_symptom_message = (
        "I have a sore throat, bad hacking cough, and high feverish temperature."
    )
    questions = get_follow_up_questions(multi_symptom_message, max_questions=cap)
    assert len(questions) == cap


def test_question_capping_zero_or_negative() -> None:
    """Verify max_questions <= 0 returns an empty list."""
    msg = "I have a sore throat and bad cough."
    assert get_follow_up_questions(msg, max_questions=0) == []
    assert get_follow_up_questions(msg, max_questions=-1) == []


# =====================================================================
# 3. Question Ordering: Red-Flag Questions First
# =====================================================================


def test_red_flag_questions_ordered_first() -> None:
    """Verify questions with red_flag_if is not None are placed before non-red-flag questions."""
    multi_symptom_message = (
        "I have a sore throat, bad hacking cough, and high feverish temperature."
    )
    # Request up to 6 questions to observe the boundary between red-flag and non-red-flag questions
    questions = get_follow_up_questions(multi_symptom_message, max_questions=6)

    seen_non_red_flag = False
    for q in questions:
        if q.red_flag_if is None:
            seen_non_red_flag = True
        else:
            # If we see a red-flag question AFTER a non-red-flag question, ordering is broken
            assert not seen_non_red_flag, (
                f"Red-flag question '{q.id}' appeared after a non-red-flag question."
            )


# =====================================================================
# 4. No Match and Non-Clinical Messages
# =====================================================================


@pytest.mark.parametrize(
    "message",
    [
        "Can you send me my clinic billing statement?",
        "Thank you for the note, everything looks good.",
        "What are your office hours this weekend?",
        "Where is patient parking located?",
        "",
        "   \t\n  ",
        "!!! ??? ...",
    ],
)
def test_no_match_returns_empty_list(message: str) -> None:
    """Verify that messages without clinical symptom groups return an empty list."""
    assert get_follow_up_questions(message) == []


# =====================================================================
# 5. Answer Evaluation: Affirmative 'yes' Variants Trigger
# =====================================================================

AFFIRMATIVE_VARIANTS = [
    "yes",
    "y",
    "yeah",
    "yep",
    "true",
    "yes i do",
    "YES",
    "Y",
    "Yeah!",
    "Yep.",
    "True",
    "Yes, I do",
]


@pytest.mark.parametrize("affirmative", AFFIRMATIVE_VARIANTS)
def test_yes_variants_trigger_red_flag(affirmative: str) -> None:
    """Verify all recognized affirmative answer variations trigger an emergency RedFlag."""
    test_question = Question(
        id="cough_blood_check",
        group="cough",
        text="Are you coughing up blood?",
        answer_type="yes_no",
        red_flag_if="yes",
        reason="Active hemoptysis detected.",
    )

    result = evaluate_answers([(test_question, affirmative)])
    assert isinstance(result, RedFlag)
    assert result.is_red_flag is True
    assert result.reason is not None
    assert "cough_blood_check" in result.reason
    assert "hemoptysis" in result.reason.lower()


def test_multiple_answers_evaluation_triggers_on_first_red_flag() -> None:
    """Verify evaluate_answers returns is_red_flag=True if ANY answer matches red_flag_if."""
    q_safe = Question(
        id="fever_duration",
        group="fever",
        text="How many days?",
        answer_type="text",
        red_flag_if=None,
    )
    q_flag = Question(
        id="fever_stiff_neck",
        group="fever",
        text="Stiff neck?",
        answer_type="yes_no",
        red_flag_if="yes",
        reason="Meningeal signs.",
    )

    answers = [(q_safe, "3 days"), (q_flag, "yes")]
    result = evaluate_answers(answers)
    assert result.is_red_flag is True
    assert "fever_stiff_neck" in result.reason


# =====================================================================
# 6. Answer Evaluation: Negative and Unclear Answers Do Not Trigger
# =====================================================================

NEGATIVE_AND_UNCLEAR_ANSWERS = [
    "no",
    "nope",
    "nah",
    "not really",
    "false",
    "NO",
    "No.",
    "maybe",
    "not sure",
    "I don't know",
    "sometimes",
    "possibly",
    "what do you mean?",
    "unknown",
    "",
    "   ",
]


@pytest.mark.parametrize("non_affirmative", NEGATIVE_AND_UNCLEAR_ANSWERS)
def test_negative_and_unclear_answers_do_not_trigger(
    non_affirmative: str,
) -> None:
    """Verify 'no' and unclear/uncertain answers do not trigger a red flag."""
    test_question = Question(
        id="chest_pain_check",
        group="cardiac",
        text="Do you have crushing chest pain?",
        answer_type="yes_no",
        red_flag_if="yes",
        reason="Cardiogenic emergency.",
    )

    result = evaluate_answers([(test_question, non_affirmative)])
    assert result.is_red_flag is False
    assert result.reason is None


def test_evaluate_empty_answers_returns_false() -> None:
    """Verify evaluating an empty answers list returns unflagged RedFlag."""
    result = evaluate_answers([])
    assert result.is_red_flag is False
    assert result.reason is None


# =====================================================================
# 7. to_follow_up_answers Schema Conversion
# =====================================================================


def test_to_follow_up_answers_conversion() -> None:
    """Verify to_follow_up_answers correctly constructs valid FollowUpAnswer models."""
    q1 = Question(
        id="st_1",
        group="sore_throat",
        text="Are you having difficulty breathing?",
        answer_type="yes_no",
        red_flag_if="yes",
    )
    q2 = Question(
        id="st_3",
        group="sore_throat",
        text="How many days have you had this sore throat?",
        answer_type="text",
    )

    pairs = [(q1, "No"), (q2, "About 3 days")]
    converted = to_follow_up_answers(pairs)

    assert len(converted) == 2
    for item in converted:
        assert isinstance(item, FollowUpAnswer)

    assert converted[0].question == "Are you having difficulty breathing?"
    assert converted[0].answer == "No"
    assert (
        converted[1].question == "How many days have you had this sore throat?"
    )
    assert converted[1].answer == "About 3 days"


# =====================================================================
# 8. Model Validation Tests
# =====================================================================


def test_question_model_validations() -> None:
    """Verify Question model requires proper fields and accepts valid answer_type literals."""
    q_valid = Question(
        id="test_id",
        group="fever",
        text="What is your temperature?",
        answer_type="text",
    )
    assert q_valid.id == "test_id"
    assert q_valid.answer_type == "text"
    assert q_valid.red_flag_if is None

    # Invalid answer_type raises ValidationError
    with pytest.raises(ValidationError):
        Question(
            id="test_bad",
            group="fever",
            text="Invalid type?",
            answer_type="multiple_choice",  # Not in ("yes_no", "text")
        )
