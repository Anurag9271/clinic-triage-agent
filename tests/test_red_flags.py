"""Tests for src/red_flags.py rule-based emergency detection.

Covers:
- Positive triggers for all 7 categories (at least 8 messages per category)
- Audited emergency phrasings (chest heaviness, worst ever headache, seizure, etc.)
- Negative triggers for ordinary clinic inbox messages (refills, scheduling, billing)
- Paraphrase cases ("can't catch my breath", "my chest feels heavy", etc.)
- Negation over-triage policy (negations are intentionally not suppressed)
- Edge cases (empty strings, whitespace, punctuation, uppercase)
- EMERGENCY_MESSAGE constant verification
- RED_FLAG_PATTERNS structure and completeness
"""

import re
import pytest

from src.red_flags import (
    EMERGENCY_MESSAGE,
    RED_FLAG_PATTERNS,
    check_red_flags,
    normalize_text,
)


# =====================================================================
# 1. Positive Triggers: Core emergency messages across all 7 categories
# =====================================================================

CARDIAC_MESSAGES = [
    "I am having sudden severe chest pain that started 20 minutes ago.",
    "There is a terrible tightness in my chest like an elephant sitting on it.",
    "I feel a crushing chest pressure and pain radiating to my jaw.",
    "Sharp pain down my left arm and across my chest.",
    "I think my husband is having a heart attack right now, please help.",
    "Patient reports acute angina and chest discomfort since this morning.",
    "My heart is racing and dizzy, I feel like I might pass out any second.",
    "Heart pounding uncontrollably with fainting spells and near collapse.",
    # Additional heaviness and pressure phrasings
    "My chest feels heavy since waking up.",
    "I have had a heavy chest and discomfort all morning.",
    "It feels like a crushing weight on my chest.",
    "There is intense pressure in my chest when I lie down.",
    "Experiencing a squeezing or crushing chest feeling right now.",
    "My chest feels tight and it is getting worse.",
    "Sharp shooting pain in my left arm or jaw.",
]

RESPIRATORY_MESSAGES = [
    "I have severe shortness of breath and cannot catch my breath.",
    "I can't breathe and my rescue inhaler is not working at all.",
    "Help, I am struggling to breathe and gasping for air.",
    "It feels like I'm suffocating and not getting enough air into my lungs.",
    "My lips are turning blue and I am very short of breath.",
    "My child has severe asthma attack and blue lips.",
    "Patient in acute respiratory distress, struggling to breathe.",
    "He is choking on food and cannot breathe right now!",
    "The patient is not breathing or turning blue, call an ambulance!",
]

NEUROLOGICAL_MESSAGES = [
    "My mother's face is drooping on the left side suddenly.",
    "He has slurred speech and can't speak clearly since waking up.",
    "One side of my body is numb and weak since 10 minutes ago.",
    "Sudden weakness in my arm and leg, I can't move my arm.",
    "I think my grandfather is having a stroke, please send someone.",
    "Experiencing the worst headache of my life, sudden thunderclap pain.",
    "My roommate is having a seizure on the floor right now.",
    "Patient had sudden loss of consciousness and remains unresponsive.",
    "I have a sudden severe or worst ever headache right now.",
    "My sister had a seizure or convulsion and won't wake up.",
    "Patient passed out or unconscious on the floor.",
    "I woke up and can't feel my arm or leg on my left side.",
    "Sudden vision loss in my right eye this morning.",
]

SEVERE_BLEEDING_MESSAGES = [
    "Cut my hand with a kitchen knife and the bleeding won't stop.",
    "I can't stop the bleeding from a deep wound on my thigh.",
    "Blood is spurting from an arterial wound on my forearm.",
    "I have soaked through several towels with heavy bleeding.",
    "I am coughing up blood in large amounts this morning.",
    "My father is vomiting blood that looks like coffee grounds.",
    "Passing massive rectal bleeding with large blood clots.",
    "Patient has arterial bleeding and severe blood loss after injury.",
    "I started coughing up blood about an hour ago.",
]

ANAPHYLAXIS_MESSAGES = [
    "I was stung by a bee and my throat is closing up fast.",
    "Feels like my throat is closing and I have hives all over my body.",
    "My tongue is swelling and lips are swollen after eating peanuts.",
    "Patient is in anaphylactic shock after taking penicillin.",
    "I just used my EpiPen because my throat was swelling shut.",
    "Severe allergic reaction with trouble breathing and hives.",
    "Throat is swelling closed and I can barely swallow my saliva.",
    "Hives and throat tight after eating seafood, need immediate help.",
]

OVERDOSE_POISONING_MESSAGES = [
    "My teenager took too many pills and is becoming very drowsy.",
    "I accidentally swallowed a whole bottle of prescription blood pressure pills.",
    "Suspected opioid overdose, patient is barely breathing.",
    "My toddler swallowed bleach from under the kitchen sink.",
    "Accidentally drank cleaner that was stored in a water bottle.",
    "Possible carbon monoxide poisoning, the whole family is dizzy and nauseated.",
    "We gave him Narcan after finding him unresponsive with empty pill bottles.",
    "I took an overdose of Tylenol about an hour ago.",
]

MENTAL_HEALTH_CRISIS_MESSAGES = [
    "I am thinking of ending my life and don't know what to do.",
    "I want to kill myself and I have a plan ready for tonight.",
    "Experiencing overwhelming suicidal thoughts all day today.",
    "I don't want to live anymore, there is no reason to live.",
    "I have a plan to end myself and wrote a suicide note.",
    "I have a strong urge to hurt myself right now.",
    "I cannot keep myself safe tonight, I need immediate crisis help.",
    "Hearing voices telling me to harm myself and kill myself.",
    "I am having persistent thoughts of hurting myself or suicide.",
]


@pytest.mark.parametrize("message", CARDIAC_MESSAGES)
def test_cardiac_triggers(message: str) -> None:
    """Verify cardiac emergency messages trigger RedFlag with cardiac category."""
    result = check_red_flags(message)
    assert result.is_red_flag is True
    assert result.reason is not None
    assert "cardiac" in result.reason.lower()


@pytest.mark.parametrize("message", RESPIRATORY_MESSAGES)
def test_respiratory_triggers(message: str) -> None:
    """Verify respiratory emergency messages trigger RedFlag with respiratory category."""
    result = check_red_flags(message)
    assert result.is_red_flag is True
    assert result.reason is not None
    assert "respiratory" in result.reason.lower()


@pytest.mark.parametrize("message", NEUROLOGICAL_MESSAGES)
def test_neurological_triggers(message: str) -> None:
    """Verify neurological emergency messages trigger RedFlag with neurological category."""
    result = check_red_flags(message)
    assert result.is_red_flag is True
    assert result.reason is not None
    assert "neurological" in result.reason.lower()


@pytest.mark.parametrize("message", SEVERE_BLEEDING_MESSAGES)
def test_severe_bleeding_triggers(message: str) -> None:
    """Verify severe bleeding messages trigger RedFlag with severe_bleeding category."""
    result = check_red_flags(message)
    assert result.is_red_flag is True
    assert result.reason is not None
    assert "severe_bleeding" in result.reason.lower()


@pytest.mark.parametrize("message", ANAPHYLAXIS_MESSAGES)
def test_anaphylaxis_triggers(message: str) -> None:
    """Verify anaphylaxis messages trigger RedFlag with anaphylaxis category."""
    result = check_red_flags(message)
    assert result.is_red_flag is True
    assert result.reason is not None
    assert "anaphylaxis" in result.reason.lower()


@pytest.mark.parametrize("message", OVERDOSE_POISONING_MESSAGES)
def test_overdose_poisoning_triggers(message: str) -> None:
    """Verify overdose/poisoning messages trigger RedFlag with overdose_poisoning category."""
    result = check_red_flags(message)
    assert result.is_red_flag is True
    assert result.reason is not None
    assert "overdose_poisoning" in result.reason.lower()


@pytest.mark.parametrize("message", MENTAL_HEALTH_CRISIS_MESSAGES)
def test_mental_health_crisis_triggers(message: str) -> None:
    """Verify mental health crisis messages trigger RedFlag with mental_health_crisis category."""
    result = check_red_flags(message)
    assert result.is_red_flag is True
    assert result.reason is not None
    assert "mental_health_crisis" in result.reason.lower()


# =====================================================================
# 2. Audited Phrasings: Explicit must-trigger cases
# =====================================================================


@pytest.mark.parametrize(
    ("phrase", "expected_category"),
    [
        # Cardiac heaviness and pressure phrasings
        ("my chest feels heavy", "cardiac"),
        ("heavy chest", "cardiac"),
        ("chest feels heavy", "cardiac"),
        ("weight on my chest", "cardiac"),
        ("pressure in my chest", "cardiac"),
        ("squeezing or crushing chest", "cardiac"),
        ("chest feels tight", "cardiac"),
        ("pain in my left arm or jaw", "cardiac"),
        # Neurological
        ("sudden severe headache", "neurological"),
        ("worst ever headache", "neurological"),
        ("sudden severe or worst ever headache", "neurological"),
        ("passed out or unconscious", "neurological"),
        ("seizure or convulsion", "neurological"),
        ("can't feel my arm or leg", "neurological"),
        ("cant feel my arm or leg", "neurological"),
        ("sudden vision loss", "neurological"),
        # Severe bleeding
        ("coughing up blood", "severe_bleeding"),
        # Respiratory
        ("not breathing or turning blue", "respiratory"),
        # Mental health crisis
        ("thoughts of hurting myself or suicide", "mental_health_crisis"),
    ],
)
def test_audited_must_trigger_phrasings(phrase: str, expected_category: str) -> None:
    """Verify all audited emergency phrasings trigger their respective red-flag category."""
    result = check_red_flags(phrase)
    assert result.is_red_flag is True, f"Failed to trigger red flag for: '{phrase}'"
    assert result.reason is not None
    assert expected_category in result.reason.lower()


# =====================================================================
# 3. Negative Triggers: Ordinary clinic inbox messages (must not trigger)
# =====================================================================

ORDINARY_CLINIC_MESSAGES = [
    # Explicit must-not-trigger phrasings from prompt
    "need a refill",
    "I need a refill for my regular blood pressure medication.",
    "reschedule my appointment",
    "Need to reschedule my appointment for next Tuesday.",
    "question about my bill",
    "I have a question about my bill from last month.",
    # Routine refills
    "Hi, I would like to request a 90-day refill of my Lisinopril 20mg prescription. My preferred pharmacy is Walgreens on Maple Ave.",
    "Could Dr. Martinez send in a refill for my daily fluticasone nasal spray to CVS Pharmacy on Elm Street?",
    "Need a renewal on my atorvastatin prescription, I have about 5 days left in my current bottle.",
    # Billing inquiries
    "Hello, I received an invoice for my office visit on Tuesday. Can someone explain the $30 copay charge on my account?",
    "I have a billing question regarding statement #49102. Does this reflect the payment made through my HSA account?",
    # Appointment scheduling
    "Good morning, I would like to schedule my routine annual wellness exam with Dr. Carter for sometime next month.",
    "Can I book a 15-minute follow-up appointment next Wednesday afternoon to review my mild knee ache?",
    "Looking to schedule a pediatric well-child check for my 4-year-old daughter with Dr. Liu.",
    # Lab and diagnostic test results
    "Can I get my blood test results?",
    "Hi, I had routine fasting blood work done last Thursday. Are the cholesterol and metabolic panel results ready yet?",
    "Checking in to see if my routine screening mammogram report has arrived from the imaging center.",
    # Administrative & paperwork requests
    "Hello front desk, I was out sick with a mild head cold on Monday and my employer requires a standard return-to-work note.",
    "Please send a copy of my child's updated immunization and vaccination record to the school nurse email.",
    "I recently changed my primary insurance provider from Blue Cross to Aetna. How can I upload my new insurance card?",
    # Mild non-emergency query
    "I am out of breath after climbing stairs, wondering about a checkup",
]


@pytest.mark.parametrize("message", ORDINARY_CLINIC_MESSAGES)
def test_ordinary_messages_do_not_trigger(message: str) -> None:
    """Verify ordinary non-emergency clinic messages return is_red_flag=False and reason=None."""
    result = check_red_flags(message)
    assert result.is_red_flag is False, f"False positive triggered on: '{message}'"
    assert result.reason is None


# =====================================================================
# 4. Paraphrase Cases
# =====================================================================


@pytest.mark.parametrize(
    ("phrase", "expected_category"),
    [
        ("I can't catch my breath after walking up two stairs.", "respiratory"),
        ("There is an unbearable tightness in my chest today.", "cardiac"),
        ("My husband noticed my face is drooping when I woke up.", "neurological"),
        ("I have been thinking of ending my life lately.", "mental_health_crisis"),
        ("Feels like my throat is closing and my tongue is heavy.", "anaphylaxis"),
        ("Cut my thumb slicing a bagel and the bleeding won't stop.", "severe_bleeding"),
        ("She took too many pills from the medicine cabinet.", "overdose_poisoning"),
    ],
)
def test_paraphrase_cases(phrase: str, expected_category: str) -> None:
    """Verify specific realistic patient paraphrases trigger their expected category."""
    result = check_red_flags(phrase)
    assert result.is_red_flag is True
    assert result.reason is not None
    assert expected_category in result.reason.lower()


# =====================================================================
# 5. Over-Triage Policy: Negations in free text are not suppressed
# =====================================================================


@pytest.mark.parametrize(
    "message_with_negation",
    [
        "Patient denies chest pain but reports fatigue.",
        "Doctor said to come in if I get shortness of breath.",
        "I am not having a heart attack but wanted to check my pulse reading.",
        "I have no thoughts of suicide, just feeling sad today.",
        "I have no chest pain, just need a sick note",
    ],
)
def test_negations_are_not_suppressed(message_with_negation: str) -> None:
    """Verify that negations are not suppressed, enforcing the clinical over-triage safety policy."""
    result = check_red_flags(message_with_negation)
    assert result.is_red_flag is True
    assert result.reason is not None


# =====================================================================
# 6. Normalization and Edge Cases
# =====================================================================


def test_empty_string_returns_false() -> None:
    """Verify empty input returns unflagged RedFlag."""
    result = check_red_flags("")
    assert result.is_red_flag is False
    assert result.reason is None


def test_whitespace_and_punctuation_only_returns_false() -> None:
    """Verify whitespace or punctuation only does not trigger red flags."""
    assert check_red_flags("   \n\t   ").is_red_flag is False
    assert check_red_flags("... ??? !!!").is_red_flag is False


def test_uppercase_and_heavy_punctuation_triggers() -> None:
    """Verify uppercase text with intense punctuation is correctly normalized and matched."""
    result = check_red_flags("HELP! CANNOT BREATHE AT ALL!!!!!")
    assert result.is_red_flag is True
    assert "respiratory" in result.reason.lower()


def test_normalize_text_function() -> None:
    """Verify normalize_text collapses whitespace, strips punctuation, and lowercases."""
    raw = "  Severe,  CHEST-PAIN...  radiating to arm!  "
    normalized = normalize_text(raw)
    assert normalized == "severe chest pain radiating to arm"


# =====================================================================
# 7. Module Structure and EMERGENCY_MESSAGE Constants
# =====================================================================


def test_emergency_message_constant() -> None:
    """Verify EMERGENCY_MESSAGE constant is non-empty and contains emergency instructions."""
    assert isinstance(EMERGENCY_MESSAGE, str)
    assert len(EMERGENCY_MESSAGE) > 20
    assert "911" in EMERGENCY_MESSAGE or "emergency" in EMERGENCY_MESSAGE.lower()


def test_red_flag_patterns_dict_completeness() -> None:
    """Verify RED_FLAG_PATTERNS contains all 7 required categories with compiled regexes."""
    expected_categories = {
        "cardiac",
        "respiratory",
        "neurological",
        "severe_bleeding",
        "anaphylaxis",
        "overdose_poisoning",
        "mental_health_crisis",
    }
    assert set(RED_FLAG_PATTERNS.keys()) == expected_categories

    for category, patterns in RED_FLAG_PATTERNS.items():
        assert isinstance(patterns, list)
        assert len(patterns) >= 1
        for pat in patterns:
            assert isinstance(pat, re.Pattern)
