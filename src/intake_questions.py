"""Adaptive symptom-specific intake questions and answer evaluation.

Provides deterministic follow-up screening questions for common symptoms,
prioritizes red-flag questions, and evaluates patient responses without
calling any LLM.
"""

from __future__ import annotations

import re
from typing import Final, Literal, Optional

from pydantic import BaseModel, Field

from src.red_flags import normalize_text
from src.schema import FollowUpAnswer, RedFlag


class Question(BaseModel):
    """An intake follow-up question for adaptive patient symptom screening."""

    id: str = Field(..., description="Unique identifier for the question.")
    group: str = Field(
        ..., description="Clinical symptom group (e.g., sore_throat, cough)."
    )
    text: str = Field(..., description="The question text presented to the patient.")
    answer_type: Literal["yes_no", "text"] = Field(
        ...,
        description="Expected answer format: 'yes_no' for boolean screening or 'text' for free response.",
    )
    red_flag_if: Optional[str] = Field(
        default=None,
        description="Answer value that triggers an emergency red flag (e.g. 'yes').",
    )
    reason: Optional[str] = Field(
        default=None,
        description="Clinical rationale shown when red_flag_if is triggered.",
    )


# Standard affirmative answers for yes_no red flag checks
YES_VARIANTS: Final[set[str]] = {"yes", "y", "yeah", "yep", "true", "yes i do"}


SYMPTOM_QUESTIONS: Final[dict[str, list[Question]]] = {
    "sore_throat": [
        Question(
            id="st_1",
            group="sore_throat",
            text="Are you having difficulty breathing, inability to swallow your own saliva, or drooling?",
            answer_type="yes_no",
            red_flag_if="yes",
            reason="Potential airway obstruction or acute epiglottitis/peritonsillar abscess.",
        ),
        Question(
            id="st_2",
            group="sore_throat",
            text="Do you have a fever, swollen neck glands, or white patches on your tonsils?",
            answer_type="yes_no",
            red_flag_if=None,
            reason=None,
        ),
        Question(
            id="st_3",
            group="sore_throat",
            text="How many days have you had this sore throat and have you been exposed to strep throat?",
            answer_type="text",
            red_flag_if=None,
            reason=None,
        ),
    ],
    "cough": [
        Question(
            id="cough_1",
            group="cough",
            text="Are you coughing up blood or experiencing severe shortness of breath?",
            answer_type="yes_no",
            red_flag_if="yes",
            reason="Acute hemoptysis or severe respiratory distress.",
        ),
        Question(
            id="cough_2",
            group="cough",
            text="Is your cough accompanied by high fever, shaking chills, or chest pain when inhaling?",
            answer_type="yes_no",
            red_flag_if="yes",
            reason="Signs of acute pneumonia or pleuritic pulmonary infection.",
        ),
        Question(
            id="cough_3",
            group="cough",
            text="How many days have you been coughing and are you producing green or yellow mucus?",
            answer_type="text",
            red_flag_if=None,
            reason=None,
        ),
    ],
    "fever": [
        Question(
            id="fever_1",
            group="fever",
            text="Is your fever accompanied by a stiff neck, confusion, lethargy, or a new non-blanching rash?",
            answer_type="yes_no",
            red_flag_if="yes",
            reason="Suspected central nervous system infection (meningitis) or acute sepsis.",
        ),
        Question(
            id="fever_2",
            group="fever",
            text="Has your temperature reached over 103°F (39.4°C) without responding to fever-reducing medication?",
            answer_type="yes_no",
            red_flag_if="yes",
            reason="Sustained high hyperpyrexia requiring urgent medical evaluation.",
        ),
        Question(
            id="fever_3",
            group="fever",
            text="What was your highest recorded temperature and how many days has the fever lasted?",
            answer_type="text",
            red_flag_if=None,
            reason=None,
        ),
    ],
    "headache": [
        Question(
            id="ha_1",
            group="headache",
            text="Did this headache start suddenly with explosive severity ('worst headache of my life')?",
            answer_type="yes_no",
            red_flag_if="yes",
            reason="Thunderclap headache concerning for subarachnoid hemorrhage.",
        ),
        Question(
            id="ha_2",
            group="headache",
            text="Do you have a stiff neck, sudden confusion, one-sided numbness, or vision loss?",
            answer_type="yes_no",
            red_flag_if="yes",
            reason="Focal neurological deficit or central intracranial pathology.",
        ),
        Question(
            id="ha_3",
            group="headache",
            text="Where is the headache pain located and does it throb or feel like constant pressure?",
            answer_type="text",
            red_flag_if=None,
            reason=None,
        ),
    ],
    "abdominal_pain": [
        Question(
            id="abd_1",
            group="abdominal_pain",
            text="Are you having severe constant pain, vomiting blood, or passing black or bloody stools?",
            answer_type="yes_no",
            red_flag_if="yes",
            reason="Acute surgical abdomen or acute gastrointestinal hemorrhage.",
        ),
        Question(
            id="abd_2",
            group="abdominal_pain",
            text="Is your abdomen rigid or rock-hard to touch, or are you unable to keep down fluids?",
            answer_type="yes_no",
            red_flag_if="yes",
            reason="Peritoneal irritation or severe dehydration requiring urgent intervention.",
        ),
        Question(
            id="abd_3",
            group="abdominal_pain",
            text="Where in your stomach or abdomen is the pain located and when did it begin?",
            answer_type="text",
            red_flag_if=None,
            reason=None,
        ),
    ],
    "rash": [
        Question(
            id="rash_1",
            group="rash",
            text="Is the rash accompanied by swelling of your face, lips, or tongue, or difficulty breathing?",
            answer_type="yes_no",
            red_flag_if="yes",
            reason="Anaphylaxis or acute angioedema.",
        ),
        Question(
            id="rash_2",
            group="rash",
            text="Is the rash blistering, peeling, painful, or spreading very rapidly with fever?",
            answer_type="yes_no",
            red_flag_if="yes",
            reason="Concern for acute blistering dermatologic emergency or severe drug reaction.",
        ),
        Question(
            id="rash_3",
            group="rash",
            text="Where on your body is the rash located and have you started any new medications or soaps?",
            answer_type="text",
            red_flag_if=None,
            reason=None,
        ),
    ],
    "back_pain": [
        Question(
            id="back_1",
            group="back_pain",
            text="Do you have numbness in your groin/saddle area, weakness in your legs, or loss of bladder/bowel control?",
            answer_type="yes_no",
            red_flag_if="yes",
            reason="Suspected cauda equina syndrome requiring emergency neurosurgical decompression.",
        ),
        Question(
            id="back_2",
            group="back_pain",
            text="Did the back pain start after a major fall or trauma, or is it paired with high fever?",
            answer_type="yes_no",
            red_flag_if="yes",
            reason="Concern for spinal fracture or spinal epidural abscess.",
        ),
        Question(
            id="back_3",
            group="back_pain",
            text="How long have you had this back pain and does it radiate down into your legs or feet?",
            answer_type="text",
            red_flag_if=None,
            reason=None,
        ),
    ],
    "dizziness": [
        Question(
            id="dizzy_1",
            group="dizziness",
            text="Did your dizziness cause you to faint, or is it accompanied by chest pain or heart palpitations?",
            answer_type="yes_no",
            red_flag_if="yes",
            reason="Cardiogenic syncope or hemodynamically unstable arrhythmia.",
        ),
        Question(
            id="dizzy_2",
            group="dizziness",
            text="Do you have slurred speech, facial drooping, double vision, or weakness on one side?",
            answer_type="yes_no",
            red_flag_if="yes",
            reason="Acute posterior circulation stroke or neurological deficit.",
        ),
        Question(
            id="dizzy_3",
            group="dizziness",
            text="Does the room feel like it is spinning around you (vertigo), or do you feel lightheaded when standing?",
            answer_type="text",
            red_flag_if=None,
            reason=None,
        ),
    ],
    "injury": [
        Question(
            id="inj_1",
            group="injury",
            text="Is there a bone piercing through the skin, severe visible deformity, or total inability to bear weight?",
            answer_type="yes_no",
            red_flag_if="yes",
            reason="Open or displaced fracture requiring immediate orthopedic evaluation.",
        ),
        Question(
            id="inj_2",
            group="injury",
            text="Did you suffer a direct blow to the head followed by loss of consciousness, confusion, or repeated vomiting?",
            answer_type="yes_no",
            red_flag_if="yes",
            reason="Traumatic brain injury or acute intracranial hematoma.",
        ),
        Question(
            id="inj_3",
            group="injury",
            text="When and how did the injury happen, and what treatments have you tried so far?",
            answer_type="text",
            red_flag_if=None,
            reason=None,
        ),
    ],
    "medication_refill": [
        Question(
            id="refill_1",
            group="medication_refill",
            text="How many days of medication do you have remaining before running out completely?",
            answer_type="text",
            red_flag_if=None,
            reason=None,
        ),
        Question(
            id="refill_2",
            group="medication_refill",
            text="Are you having any new, unexpected, or concerning side effects from this prescription?",
            answer_type="yes_no",
            red_flag_if=None,
            reason=None,
        ),
        Question(
            id="refill_3",
            group="medication_refill",
            text="What is the exact medication name, dosage, and your preferred pharmacy location?",
            answer_type="text",
            red_flag_if=None,
            reason=None,
        ),
    ],
}


SYMPTOM_PATTERNS: Final[dict[str, list[re.Pattern]]] = {
    "sore_throat": [
        re.compile(
            r"\b(sore\s+throat|throat\s+(hurts?|is\s+sore|aching)|scratchy\s+throat|pain\s+swallowing|hurts?\s+to\s+swallow|swollen\s+tonsils?|strep(\s+throat)?)\b"
        ),
    ],
    "cough": [
        re.compile(
            r"\b(cough(ing|s)?|hacking\s+cough|dry\s+cough|wet\s+cough|productive\s+cough|chest\s+cold)\b"
        ),
    ],
    "fever": [
        re.compile(
            r"\b(fever(ish)?|high\s+temperature|elevated\s+temp(erature)?|chills|burning\s+up|night\s+sweats)\b"
        ),
    ],
    "headache": [
        re.compile(
            r"\b(headaches?|head\s+(hurts?|aches?|aching)|migraines?|pain\s+in\s+(my\s+)?head|throbbing\s+head)\b"
        ),
    ],
    "abdominal_pain": [
        re.compile(
            r"\b(abdominal\s+pain|stomach\s+(ache|pain|hurts?|cramps?)|belly\s+(ache|pain|hurts?)|tummy\s+(ache|pain|hurts?)|gut\s+pain|pain\s+in\s+(my\s+)?(stomach|belly|abdomen))\b"
        ),
    ],
    "rash": [
        re.compile(
            r"\b(rash(es)?|hives|skin\s+breakout|itchy\s+skin|red\s+bumps|skin\s+spots|skin\s+eruption|skin\s+irritation)\b"
        ),
    ],
    "back_pain": [
        re.compile(
            r"\b(back\s+pain|lower\s+back\s+pain|backache|back\s+hurts?|pain\s+in\s+(my\s+)?back|lumbar\s+pain|sciatica)\b"
        ),
    ],
    "dizziness": [
        re.compile(
            r"\b(dizzy|dizziness|feel\s+dizzy|lightheaded(ness)?|feeling\s+faint|vertigo|room\s+is\s+spinning|unsteady\s+on\s+(my\s+)?feet)\b"
        ),
    ],
    "injury": [
        re.compile(
            r"\b(injur(y|ed|ies)|fell\s+down|fall|sprain(ed)?|twisted\s+(my\s+)?(ankle|wrist|knee)|broke\s+(my\s+)?|fracture|hurt\s+my\s+(ankle|wrist|knee|leg|arm|foot|shoulder)|cut\s+my\s+|accident)\b"
        ),
    ],
    "medication_refill": [
        re.compile(
            r"\b(need\s+a\s+refill|refill|renew(al)?|prescription\s+renewal|run(ning)?\s+out\s+of\s+(pills|medication)|medication\s+refill|renew\s+(my\s+)?prescription)\b"
        ),
    ],
}


def get_follow_up_questions(message: str, max_questions: int = 3) -> list[Question]:
    """Identify symptom groups from a message and collect relevant follow-up questions.

    Red-flag screening questions are ordered first, duplicate questions are
    removed, and the total list is capped at max_questions.

    Args:
        message: Patient intake message or symptom complaint.
        max_questions: Maximum number of questions to return (default: 3).

    Returns:
        List of relevant Question instances, or empty list if no groups match.
    """
    if not message or max_questions <= 0:
        return []

    normalized = normalize_text(message)
    if not normalized:
        return []

    matched_groups: list[str] = []
    for group, patterns in SYMPTOM_PATTERNS.items():
        if any(pat.search(normalized) for pat in patterns):
            matched_groups.append(group)

    if not matched_groups:
        return []

    collected_questions: list[Question] = []
    for group in matched_groups:
        collected_questions.extend(SYMPTOM_QUESTIONS.get(group, []))

    # Stable sort placing red-flag questions (red_flag_if is not None) first
    sorted_questions = sorted(
        collected_questions, key=lambda q: 0 if q.red_flag_if else 1
    )

    # Deduplicate by question ID while preserving sorted order
    seen_ids: set[str] = set()
    unique_questions: list[Question] = []
    for q in sorted_questions:
        if q.id not in seen_ids:
            seen_ids.add(q.id)
            unique_questions.append(q)

    return unique_questions[:max_questions]


def _normalize_answer(answer: str) -> str:
    """Helper to lowercase, remove punctuation, and collapse whitespace in user answers."""
    cleaned = re.sub(r"[^\w\s]", " ", answer.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def evaluate_answers(answers: list[tuple[Question, str]]) -> RedFlag:
    """Evaluate patient answers to screening questions for clinical red flags.

    Recognizes standard affirmative variants (yes, y, yeah, yep, true, yes i do)
    in a case-insensitive manner. Negative or unclear answers do not trigger a flag.

    Args:
        answers: List of (Question, patient_answer_str) pairs.

    Returns:
        RedFlag instance with is_red_flag=True if any red-flag condition is met,
        or is_red_flag=False otherwise.
    """
    for question, answer in answers:
        if not question.red_flag_if:
            continue

        norm_answer = _normalize_answer(answer)
        target = _normalize_answer(question.red_flag_if)

        is_match = False
        if target in YES_VARIANTS:
            is_match = norm_answer in YES_VARIANTS
        else:
            is_match = norm_answer == target

        if is_match:
            detail = question.reason or question.text
            reason = f"Intake red flag triggered on question '{question.id}': {detail}"
            return RedFlag(is_red_flag=True, reason=reason)

    return RedFlag(is_red_flag=False, reason=None)


def to_follow_up_answers(
    answers: list[tuple[Question, str]],
) -> list[FollowUpAnswer]:
    """Convert question-answer pairs into the schema's FollowUpAnswer models.

    Args:
        answers: List of (Question, patient_answer_str) pairs.

    Returns:
        List of validated FollowUpAnswer models.
    """
    return [
        FollowUpAnswer(question=question.text, answer=str(answer).strip())
        for question, answer in answers
    ]


__all__ = [
    "Question",
    "SYMPTOM_PATTERNS",
    "SYMPTOM_QUESTIONS",
    "YES_VARIANTS",
    "evaluate_answers",
    "get_follow_up_questions",
    "to_follow_up_answers",
]
