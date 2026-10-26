"""Rule-based emergency red-flag detector for patient messages.

This module provides deterministic, rule-based screening for life-threatening
clinical conditions prior to any LLM processing. It uses compiled regular
expressions to detect symptoms across critical emergency categories.
"""

from __future__ import annotations

import re
from typing import Final

from src.schema import RedFlag

EMERGENCY_MESSAGE: Final[str] = (
    "If you are experiencing a life-threatening medical emergency, please call 911 "
    "(or your local emergency services) or go to the nearest emergency room immediately."
)


def normalize_text(text: str) -> str:
    """Normalize input text by lowercasing, stripping punctuation, and collapsing whitespace.

    Args:
        text: Raw patient message.

    Returns:
        Normalized string suitable for regex matching.
    """
    if not text:
        return ""
    # Lowercase
    cleaned = text.lower()
    # Replace punctuation characters with spaces so words separated by punctuation
    # (e.g. "pain.sudden" or "chest,hurts") do not get concatenated into single tokens.
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    # Collapse consecutive whitespace characters into a single space
    return re.sub(r"\s+", " ", cleaned).strip()


# Pattern groups ordered to prioritize high-specificity systemic emergencies
# (overdose/poisoning and anaphylaxis) before organ-specific symptom categories.
RED_FLAG_PATTERNS: Final[dict[str, list[re.Pattern]]] = {
    "overdose_poisoning": [
        re.compile(r"\b(narcan\s+(given|administered|used)|gave\s+(him\s+|her\s+|them\s+)?narcan)\b"),
        re.compile(r"\b(overdos(e|ed|ing)|accidental\s+overdose|drug\s+overdose|(opioid|fentanyl)\s+overdose)\b"),
        re.compile(r"\b(took\s+too\s+many\s+(pills|medications?|tablets?)|swallowed\s+(a\s+whole\s+|an\s+entire\s+)?bottle\s+of)\b"),
        re.compile(r"\b(took\s+(a\s+whole\s+|an\s+entire\s+|all\s+my\s+)bottle\s+of)\b"),
        re.compile(r"\b(took\s+(too\s+much|an\s+overdose\s+of)\s+(insulin|tylenol|aspirin|medication|pills))\b"),
        re.compile(r"\b(swallowed\s+(poison|bleach|detergent|chemicals?|batter(y|ies)|toxic(\s+substance)?))\b"),
        re.compile(r"\b(drank\s+(bleach|cleaner|antifreeze|poison|chemicals?))\b"),
        re.compile(r"\b(carbon\s+monoxide\s+poisoning|toxic\s+ingestion)\b"),
    ],
    "anaphylaxis": [
        re.compile(r"\b(throat\s+(is\s+)?closing(\s+up)?|feels?\s+like\s+(my\s+)?throat\s+is\s+closing)\b"),
        re.compile(r"\b(throat\s+(is\s+)?swelling(\s+(shut|closed))?|swelling\s+in\s+(my\s+)?throat)\b"),
        re.compile(r"\b(tongue\s+(is\s+)?swell(ing|en)|swollen\s+tongue|lips?\s+and\s+tongue\s+swelling|swelling\s+of\s+(the\s+)?tongue)\b"),
        re.compile(r"\b(anaphylax(is|ic)|anaphylactic\s+shock)\b"),
        re.compile(r"\b((used|need|injected|administered)\s+(an?|my|the)?\s*epi\s*pen)\b"),
        re.compile(r"\b(allergic\s+reaction\s+(and|with)\s+(can\s*t\s+breathe|trouble\s+breathing|throat\s+(tight|closing)|swelling))\b"),
        re.compile(r"\b(severe\s+allergic\s+reaction)\b"),
        re.compile(r"\b(hives\s+(and|with)\s+(trouble\s+breathing|throat\s+(closing|tight)|shortness\s+of\s+breath|wheezing|swelling))\b"),
    ],
    "cardiac": [
        re.compile(r"\b(chest\s+(pain|pressure|tightness|heaviness|squeezing|hurts?|aching|discomfort))\b"),
        re.compile(r"\b(heart\s+(pain|pressure|tightness|heaviness|squeezing|hurts?|aching|discomfort))\b"),
        re.compile(r"\b(tightness|pressure|heaviness|squeezing|crushing|elephant|weight)\s+(in|on|across)\s+(my\s+)?(chest|heart)\b"),
        re.compile(r"\b(crushing|severe|acute|sharp|stabbing|heavy)\s+(pain\s+in\s+(my\s+)?(chest|heart)|chest\s+pain|heart\s+pain|chest|heart)\b"),
        re.compile(r"\b(chest|heart)\s+(feels?|is)\s+(heavy|tight|crushed|squeezed|painful|aching)\b"),
        re.compile(r"\b(heavy\s+chest|chest\s+feels?\s+heavy|chest\s+feels?\s+tight|heavy\s+heart(\s+pain)?)\b"),
        re.compile(r"\bpain\s+(in|across|around|near)\s+(my\s+)?(chest|heart)\b"),
        re.compile(r"\b(squeezing|crushing)\s+(in\s+|on\s+)?(my\s+)?(chest|heart)\b"),
        re.compile(r"\b(radiating|spreading|shooting)\s+to\s+(my\s+)?(left\s+)?(arm|jaw|neck|back|shoulder)\b"),
        re.compile(r"\bpain\s+(down|in)\s+(my\s+)?(left\s+arm|jaw)(\s+or\s+(my\s+)?(left\s+arm|jaw))?\b"),
        re.compile(r"\b(heart\s+attack|cardiac\s+arrest|myocardial\s+infarction|angina(\s+pectoris)?)\b"),
        re.compile(r"\b(heart\s+(is\s+)?(racing|fluttering|pounding|skipping|beating\s+erratically)|palpitations)\s+(uncontrollably\s+)?(and|with)\s+(dizzy|dizziness|faint|fainting|passed\s+out|passing\s+out|blackout|blacking\s+out|collapsed?)\b"),
    ],
    "respiratory": [
        re.compile(r"\b(shortness\s+of\s+breath|short\s+of\s+breath|dyspnea)\b"),
        re.compile(r"\b(can\s*t|cannot|unable\s+to|hard\s+to|struggling\s+to|trouble|difficulty)\s+(catch\s+(my\s+)?breath|breathe|breathing)\b"),
        re.compile(r"\b(gasping\s+for\s+(air|breath)|fighting\s+for\s+air|air\s+hunger|not\s+getting\s+enough\s+air)\b"),
        re.compile(r"\b(feel\s+like\s+(i\s*m\s+|im\s+)?suffocating|suffocating)\b"),
        re.compile(r"\b(lips?|fingers?|skin|face)\s+(are\s+|is\s+)?turning\s+blue\b"),
        re.compile(r"\b(turning\s+blue|blue\s+lips?|cyanosis)\b"),
        re.compile(r"\b(severe\s+asthma\s+attack|inhaler\s+not\s+(helping|working)\s+(and|can\s*t))\b"),
        re.compile(r"\b(respiratory\s+(distress|failure|arrest)|stopped\s+breathing|not\s+breathing(\s+or\s+turning\s+blue)?)\b"),
        re.compile(r"\b(choking\s+and\s+(can\s*t|cannot)\s+breathe)\b"),
    ],
    "neurological": [
        re.compile(r"\b(face\s+(is\s+)?drooping|facial\s+droop(ing)?|drooping\s+(on\s+)?one\s+side\s+of\s+(my\s+)?face)\b"),
        re.compile(r"\b(slurred\s+speech|slurring\s+(my\s+)?words?|cannot\s+speak|unable\s+to\s+speak|(can\s*t|cannot)\s+speak\s+clearly|lost\s+(my\s+)?ability\s+to\s+speak)\b"),
        re.compile(r"\b(one\s+side\s+(of\s+(my\s+)?body\s+)?is\s+(numb|weak|paralyzed)|sudden\s+(weakness|numbness|paralysis)\s+(in\s+)?(my\s+)?(arm|leg|face|side|body))\b"),
        re.compile(r"\b(can\s*t|cannot|unable\s+to)\s+(feel|move)\s+(my\s+)?(arm|leg|one\s+side)(\s+or\s+(my\s+)?(arm|leg|one\s+side))?\b"),
        re.compile(r"\b((having\s+a\s+|suspected\s+|signs?\s+of\s+(a\s+)?)stroke|transient\s+ischemic\s+attack)\b"),
        re.compile(r"\b(worst\s+(ever\s+headache|headache\s+ever|headache\s+of\s+(my\s+)?life|headache)|sudden\s+severe\s+headache|thunderclap\s+headache|explosive\s+headache)\b"),
        re.compile(r"\b(seizures?|convulsions?|grand\s+mal(\s+seizure)?|active\s+seizure|having\s+a\s+seizure|seizure\s+or\s+convulsion)\b"),
        re.compile(r"\b(passed\s+out(\s+or\s+unconscious)?|passing\s+out|blacked\s+out|blacking\s+out|unconscious|loss\s+of\s+consciousness|lost\s+consciousness|unresponsive|cannot\s+wake\s+(him|her|them|up)|won\s*t\s+wake\s+up)\b"),
        re.compile(r"\b(sudden\s+(vision\s+loss|loss\s+of\s+vision|blindness)|suddenly\s+(lost\s+vision|went\s+blind)|loss\s+of\s+vision\s+in\s+one\s+eye)\b"),
    ],
    "severe_bleeding": [
        re.compile(r"\b(bleeding\s+(won\s*t|will\s+not|does\s+not|cannot|can\s*t)\s+stop|(can\s*t|cannot)\s+stop\s+(the\s+)?bleeding)\b"),
        re.compile(r"\b(bleeding\s+(profusely|heavily|uncontrollably|nonstop|non\s+stop))\b"),
        re.compile(r"\b(blood\s+(is\s+)?(spurting|gushing|pouring))\b"),
        re.compile(r"\b(soaked\s+through\s+(multiple|several|many|\d+)\s+(bandages?|towels?|dressings?|gauze))\b"),
        re.compile(r"\b(cough(ing|ed)?\s+(up\s+)?blood|hemoptysis)\b"),
        re.compile(r"\b(vomiting\s+blood|throwing\s+up\s+blood|hematemesis|vomit(ing)?\s+looks?\s+like\s+coffee\s+grounds?)\b"),
        re.compile(r"\b(large\s+blood\s+clots|massive\s+(rectal\s+)?bleeding|toilet\s+(bowl\s+)?full\s+of\s+blood|stool\s+(is\s+)?(black\s+and\s+tarry|pure\s+blood))\b"),
        re.compile(r"\b(hemorrhag(e|ing)|arterial\s+bleed(ing)?|severe\s+blood\s+loss)\b"),
    ],
    "mental_health_crisis": [
        re.compile(r"\b((thinking\s+(of|about)|going\s+to|want\s+to|planning\s+to)\s+(end|ending)\s+(my\s+life|it\s+all))\b"),
        re.compile(r"\b((want\s+to|going\s+to|feel\s+like|planning\s+to|gonna)\s+kill\s+myself|kill\s+myself)\b"),
        re.compile(r"\b(suicid(e|al)|commit\s+suicide|suicidal\s+(thoughts?|ideation|plan|tendencies))\b"),
        re.compile(r"\b(thoughts?\s+of\s+(hurting\s+myself|suicide|self\s*harm)(\s+or\s+(hurting\s+myself|suicide|self\s*harm))?)\b"),
        re.compile(r"\b(don\s*t\s+want\s+to\s+live(\s+anymore)?|no\s+reason\s+to\s+(keep\s+)?live|can\s*t\s+go\s+on\s+living|no\s+point\s+in\s+living)\b"),
        re.compile(r"\b(have\s+a\s+plan\s+to\s+(kill|end)\s+myself|wrote\s+a\s+suicide\s+note)\b"),
        re.compile(r"\b((want\s+to|urge\s+to|going\s+to|feel\s+like|thinking\s+of|thoughts?\s+of)\s+(hurt|harm|hurting|harming)\s+myself)\b"),
        re.compile(r"\b((can\s*t|cannot)\s+keep\s+myself\s+safe)\b"),
        re.compile(r"\b(voices\s+(are\s+)?telling\s+me\s+to\s+(kill|hurt|harm)\s+myself)\b"),
    ],
}


def check_red_flags(text: str) -> RedFlag:
    """Scan text for emergency red-flag symptoms using rule-based pattern matching.

    In accordance with clinical safety protocols, this function errs on the side
    of over-triage and does not suppress negations in free text.

    Args:
        text: Patient input message or intake complaint.

    Returns:
        A RedFlag instance with is_red_flag=True and a descriptive reason naming
        the matched category and phrase if an emergency indicator is found,
        or is_red_flag=False and reason=None otherwise.
    """
    if not text:
        return RedFlag(is_red_flag=False, reason=None)

    normalized = normalize_text(text)
    if not normalized:
        return RedFlag(is_red_flag=False, reason=None)

    for category, patterns in RED_FLAG_PATTERNS.items():
        for pattern in patterns:
            match = pattern.search(normalized)
            if match:
                matched_phrase = match.group(0).strip()
                reason = (
                    f"Emergency red flag detected ({category}): '{matched_phrase}'"
                )
                return RedFlag(is_red_flag=True, reason=reason)

    return RedFlag(is_red_flag=False, reason=None)


__all__ = [
    "EMERGENCY_MESSAGE",
    "RED_FLAG_PATTERNS",
    "check_red_flags",
    "normalize_text",
]
