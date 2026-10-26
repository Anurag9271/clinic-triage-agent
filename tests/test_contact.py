"""Unit tests for contact extraction, validation, grounding, and merging logic."""

from __future__ import annotations

import pytest

from src.contact import (
    extract_contact_regex,
    ground_llm_contact,
    has_contact_method,
    merge_contact,
    validate_contact_input,
)
from src.schema import Contact


# ============================================================================
# 1. Regex Extraction Tests
# ============================================================================


def test_extract_contact_regex_phone_formats():
    """Verify regex extraction across diverse valid and invalid phone formats."""
    # 7 digits with hyphen
    c1 = extract_contact_regex("Please call 555-0144 for Arthur.")
    assert c1.phone == "555-0144"
    assert c1.email is None
    assert c1.name is None

    # 10 digits with parentheses
    c2 = extract_contact_regex("My number is (555) 012-3456.")
    assert c2.phone == "(555) 012-3456"

    # International phone format with leading +
    c3 = extract_contact_regex("Reach me at +1 (555) 123-4567 anytime.")
    assert c3.phone == "+1 (555) 123-4567"

    # International UK format
    c4 = extract_contact_regex("Contact: +44 20 7946 0919.")
    assert c4.phone == "+44 20 7946 0919"

    # Plain 7 digits
    c5 = extract_contact_regex("Direct line 1234567.")
    assert c5.phone == "1234567"

    # Maximum 15 digits
    c6 = extract_contact_regex("International: +123456789012345.")
    assert c6.phone == "+123456789012345"

    # Too few digits (< 7) should not extract as phone
    c7 = extract_contact_regex("I have a fever of 101.5 and 2 pills left.")
    assert c7.phone is None

    # 6 digits should not extract
    c8 = extract_contact_regex("Code: 123456.")
    assert c8.phone is None

    # Too many digits (> 15) should not extract
    c9 = extract_contact_regex("Account: 1234567890123456.")
    assert c9.phone is None


def test_extract_contact_regex_email_formats():
    """Verify regex extraction across standard and complex email formats."""
    # Standard email
    c1 = extract_contact_regex("Email me at sarah.j@example.com please.")
    assert c1.email == "sarah.j@example.com"
    assert c1.phone is None

    # Subdomain and plus tag
    c2 = extract_contact_regex("Send records to patient+tag@sub.clinic.co.uk.")
    assert c2.email == "patient+tag@sub.clinic.co.uk"

    # Trailing period and punctuation handled cleanly
    c3 = extract_contact_regex("Write to info@healthcenter.org!")
    assert c3.email == "info@healthcenter.org"

    # Both phone and email extracted together
    c4 = extract_contact_regex("Maria Garcia, 555-0128, maria.g@example.com.")
    assert c4.phone == "555-0128"
    assert c4.email == "maria.g@example.com"
    assert c4.name is None  # Names are never extracted by regex

    # Invalid email formats
    c5 = extract_contact_regex("Contact invalid@ or notanemail.")
    assert c5.email is None


# ============================================================================
# 2. Input Validation Tests
# ============================================================================


def test_validate_contact_input_valid():
    """Valid inputs and blank inputs should produce no validation errors."""
    # Empty / blank values are allowed
    assert validate_contact_input(None, None, None) == []
    assert validate_contact_input("", "", "") == []
    assert validate_contact_input("   ", "   ", "   ") == []

    # Valid name, phone, email
    errors = validate_contact_input(
        name="John Doe",
        phone="(555) 012-3456",
        email="john.doe@example.com",
    )
    assert errors == []


def test_validate_contact_input_errors():
    """Invalid email format and out-of-range phone numbers produce error messages."""
    # Invalid email format
    err_email = validate_contact_input(email="not-an-email")
    assert len(err_email) == 1
    assert "email" in err_email[0].lower()

    # Phone with fewer than 7 digits
    err_phone_short = validate_contact_input(phone="12345")
    assert len(err_phone_short) == 1
    assert "between 7 and 15 digits" in err_phone_short[0]

    # Phone with more than 15 digits
    err_phone_long = validate_contact_input(phone="1234567890123456")
    assert len(err_phone_long) == 1
    assert "between 7 and 15 digits" in err_phone_long[0]

    # Multiple errors returned together
    err_both = validate_contact_input(phone="123", email="invalid@")
    assert len(err_both) == 2


# ============================================================================
# 3. LLM Contact Grounding Tests
# ============================================================================


def test_ground_llm_contact_removes_invented_phone():
    """Grounding strips phone numbers invented by the LLM that do not appear in the message."""
    message = "I need an urgent appointment with Dr. Evans for migraine symptoms."
    hallucinated_contact = Contact(
        name=None,
        phone="555-9999",  # Not in message
        email=None,
    )
    grounded = ground_llm_contact(hallucinated_contact, message)
    assert grounded.phone is None


def test_ground_llm_contact_retains_verified_details():
    """Grounding retains details whose text or digits actually appear in the message."""
    message = "Hello, please call Arthur Davis at 555-0144 or email arthur@example.com."
    llm_contact = Contact(
        name="Arthur Davis",
        phone="555-0144",
        email="arthur@example.com",
    )
    grounded = ground_llm_contact(llm_contact, message)
    assert grounded.name == "Arthur Davis"
    assert grounded.phone == "555-0144"
    assert grounded.email == "arthur@example.com"


def test_ground_llm_contact_removes_invented_name_and_email():
    """Grounding removes ungrounded names and emails while keeping genuine ones."""
    message = "Need a refill for my Lisinopril 20mg. Contact: 555-0144."
    llm_contact = Contact(
        name="Jane Hallucinated",
        phone="555-0144",
        email="fake@clinic.org",
    )
    grounded = ground_llm_contact(llm_contact, message)
    assert grounded.name is None
    assert grounded.phone == "555-0144"
    assert grounded.email is None


def test_ground_llm_contact_case_insensitive_name():
    """Name grounding is case-insensitive."""
    message = "Patient name is ROBERT CHEN, please reschedule my checkup."
    llm_contact = Contact(name="Robert Chen")
    grounded = ground_llm_contact(llm_contact, message)
    assert grounded.name == "Robert Chen"


# ============================================================================
# 4. Merge Precedence Tests
# ============================================================================


def test_merge_contact_precedence_per_field():
    """Precedence hierarchy: form value > regex value > grounded LLM value."""
    message = "Call me at 555-0111 or write to regex@example.com. Thanks, Bob."
    form_contact = Contact(
        name="Alice Form",
        phone="555-0999",
        email=None,
        preferred_method="phone",
    )
    llm_contact = Contact(
        name="Bob",
        phone="555-0111",
        email="regex@example.com",
    )

    merged = merge_contact(
        form=form_contact,
        message=message,
        llm_contact=llm_contact,
    )

    # Name: form overrides grounded LLM ("Alice Form" > "Bob")
    assert merged.name == "Alice Form"
    # Phone: form overrides regex and LLM ("555-0999" > "555-0111")
    assert merged.phone == "555-0999"
    # Email: form is None, so regex takes precedence
    assert merged.email == "regex@example.com"
    # Preferred method preserved from form
    assert merged.preferred_method == "phone"


def test_merge_contact_regex_over_grounded_llm():
    """When form is absent, regex extraction takes precedence over grounded LLM."""
    message = "Contact phone: 555-0122, email: test@example.com."
    llm_contact = Contact(
        name=None,
        phone="555-0122",
        email="test@example.com",
    )

    merged = merge_contact(
        form=None,
        message=message,
        llm_contact=llm_contact,
    )

    assert merged.phone == "555-0122"
    assert merged.email == "test@example.com"


def test_merge_contact_discards_llm_hallucinations_when_form_absent():
    """Invented LLM contact is discarded when message contains no contact."""
    message = "I have a sore throat for three days."
    llm_contact = Contact(
        name="Invented Person",
        phone="555-1234",
        email="invented@example.com",
    )

    merged = merge_contact(
        form=None,
        message=message,
        llm_contact=llm_contact,
    )

    assert merged.name is None
    assert merged.phone is None
    assert merged.email is None


# ============================================================================
# 5. has_contact_method Tests
# ============================================================================


def test_has_contact_method():
    """Return True if phone or email is provided, False otherwise."""
    assert has_contact_method(Contact(phone="555-0144")) is True
    assert has_contact_method(Contact(email="user@example.com")) is True
    assert has_contact_method(Contact(phone="555-0144", email="user@example.com")) is True

    # Name alone does NOT provide a contact method
    assert has_contact_method(Contact(name="Arthur Davis")) is False
    assert has_contact_method(Contact()) is False
    assert has_contact_method(Contact(phone="", email="   ")) is False
