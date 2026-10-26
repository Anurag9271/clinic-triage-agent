"""Patient contact extraction, validation, grounding, and merging utilities.

Provides deterministic regex extraction for phone numbers and email addresses,
input validation for intake forms, grounding of LLM-extracted contact details
against the raw message text, and prioritized merging across intake channels.
"""

from __future__ import annotations

import re
from typing import Final, Optional

from src.schema import Contact

# Standard email regex pattern matching user@domain.tld
EMAIL_REGEX: Final[re.Pattern[str]] = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

# Phone regex capturing 7 to 15 digits allowing spaces, dashes, brackets, and leading +
PHONE_REGEX: Final[re.Pattern[str]] = re.compile(
    r"(?:(?<=\s)|(?<=^)|(?<=[,;:\"\'\(\[]))"
    r"(\+?(?:\(\d{1,5}\)|\d{1,5})[\d\s\-\(\)\.]{5,25}\d)"
    r"(?=[.,;:!?\s\"\'\)\]]|$)"
)


def extract_contact_regex(message: str) -> Contact:
    """Deterministically extract email and phone from raw message text using regex.

    Names are not extracted by regex and are always returned as None.
    Phone numbers must contain between 7 and 15 digits, allowing spaces, dashes,
    brackets, and an optional leading '+'.

    Args:
        message: Raw message string.

    Returns:
        Contact instance populated with extracted phone and email, or None.
    """
    if not message or not message.strip():
        return Contact()

    extracted_phone: Optional[str] = None
    for match in PHONE_REGEX.finditer(message):
        candidate = match.group(1).strip(".,;:!? \t\n")
        digits = re.sub(r"\D", "", candidate)
        if 7 <= len(digits) <= 15:
            # Must consist only of digits, spaces, -, (, ), ., and optional leading +
            if re.match(r"^\+?[\d\s\-\(\)\.]+$", candidate):
                if candidate.count("(") == candidate.count(")"):
                    extracted_phone = candidate
                    break

    extracted_email: Optional[str] = None
    email_match = EMAIL_REGEX.search(message)
    if email_match:
        extracted_email = email_match.group(0).strip()

    return Contact(
        name=None,
        phone=extracted_phone,
        email=extracted_email,
    )


def validate_contact_input(
    name: Optional[str] = None,
    phone: Optional[str] = None,
    email: Optional[str] = None,
) -> list[str]:
    """Validate intake form contact inputs.

    Enforces standard email format and phone numbers with 7 to 15 digits.
    Blank, empty, or None values are allowed and produce no errors.

    Args:
        name: Optional patient name.
        phone: Optional patient telephone number.
        email: Optional patient email address.

    Returns:
        List of validation error messages. Empty list if all inputs are valid.
    """
    errors: list[str] = []

    if email is not None and email.strip():
        cleaned_email = email.strip()
        if not re.fullmatch(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", cleaned_email):
            errors.append("Invalid email format.")

    if phone is not None and phone.strip():
        digits = re.sub(r"\D", "", phone.strip())
        if len(digits) < 7 or len(digits) > 15:
            errors.append("Phone number must have between 7 and 15 digits.")

    return errors


def ground_llm_contact(llm_contact: Contact, message: str) -> Contact:
    """Ground LLM-extracted contact details against the raw patient message.

    Safety guard against LLM hallucination:
    - Retains phone or email only if its digits or text actually appear in the message.
    - Retains name only if its text appears in the message (case-insensitive).
    - Otherwise, sets the corresponding field to None.

    Args:
        llm_contact: Contact details parsed from LLM JSON output.
        message: Raw message text.

    Returns:
        New Contact instance with hallucinated or ungrounded fields stripped to None.
    """
    if not message:
        return Contact()

    grounded_name: Optional[str] = None
    if llm_contact.name and llm_contact.name.strip():
        cand_name = llm_contact.name.strip()
        if cand_name.lower() in message.lower():
            grounded_name = cand_name

    grounded_phone: Optional[str] = None
    if llm_contact.phone and llm_contact.phone.strip():
        cand_phone = llm_contact.phone.strip()
        phone_digits = re.sub(r"\D", "", cand_phone)
        msg_digits = re.sub(r"\D", "", message)
        if cand_phone in message or (phone_digits and phone_digits in msg_digits):
            grounded_phone = cand_phone

    grounded_email: Optional[str] = None
    if llm_contact.email and llm_contact.email.strip():
        cand_email = llm_contact.email.strip()
        if cand_email.lower() in message.lower():
            grounded_email = cand_email

    return Contact(
        name=grounded_name,
        phone=grounded_phone,
        email=grounded_email,
        preferred_method=getattr(llm_contact, "preferred_method", None),
    )


def merge_contact(
    form: Optional[Contact],
    message: str,
    llm_contact: Contact,
) -> Contact:
    """Merge contact details with strict field-level precedence.

    Precedence order for each field (name, phone, email):
    1. Explicit intake form input (highest priority)
    2. Deterministic regex extraction from message
    3. Grounded LLM extraction (verified against message text)

    Args:
        form: Optional Contact instance from intake form inputs.
        message: Raw message text.
        llm_contact: Raw Contact details extracted by the LLM.

    Returns:
        Consolidated Contact instance conforming to precedence hierarchy.
    """
    regex_contact = extract_contact_regex(message)
    grounded_llm = ground_llm_contact(llm_contact, message)

    def _pick(
        form_val: Optional[str],
        regex_val: Optional[str],
        grounded_val: Optional[str],
    ) -> Optional[str]:
        if form_val and form_val.strip():
            return form_val.strip()
        if regex_val and regex_val.strip():
            return regex_val.strip()
        if grounded_val and grounded_val.strip():
            return grounded_val.strip()
        return None

    merged_name = _pick(
        getattr(form, "name", None),
        regex_contact.name,
        grounded_llm.name,
    )
    merged_phone = _pick(
        getattr(form, "phone", None),
        regex_contact.phone,
        grounded_llm.phone,
    )
    merged_email = _pick(
        getattr(form, "email", None),
        regex_contact.email,
        grounded_llm.email,
    )
    pref_method = getattr(form, "preferred_method", None) if form else None

    return Contact(
        name=merged_name,
        phone=merged_phone,
        email=merged_email,
        preferred_method=pref_method,
    )


def has_contact_method(contact: Contact) -> bool:
    """Return True if the contact contains a valid phone number or email address.

    Args:
        contact: Contact instance to check.

    Returns:
        True if at least one contact method (phone or email) is present, False otherwise.
    """
    has_phone = bool(contact.phone and contact.phone.strip())
    has_email = bool(contact.email and contact.email.strip())
    return has_phone or has_email


__all__ = [
    "EMAIL_REGEX",
    "PHONE_REGEX",
    "extract_contact_regex",
    "ground_llm_contact",
    "has_contact_method",
    "merge_contact",
    "validate_contact_input",
]
