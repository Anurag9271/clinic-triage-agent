# Prompt Log: Clinic Inbox Triage Agent

This file records every significant prompt used to build the project, including failures and follow-up prompts that improved the results.

**How to read this log**

- **Build prompts:** instructions given to the AI coding agent (Antigravity) to generate or fix code.
- **System prompts:** instructions the triage LLM runs on inside the app (v1, v2, v3...), refined using evaluation results.
- Every entry follows the same structure: goal, prompt, result, problem, follow-up, new result, lesson.

**AI tools used:** Antigravity IDE (coding agent), Ollama (local models: Llama 3.2 3B, Gemma 3 4B, Llama 3.1 8B / Qwen2.5 7B), Claude (planning, prompt refinement).

---

## Summary table

| # | Type | Topic | Outcome |
|---|------|-------|---------|
| 01 | Build | Project structure and Pydantic schema | |
| 02 | Build | Synthetic dataset | |
| 03 | System | Triage prompt v1 | |

(Add a row each time you add an entry.)

---

## Entry 01: Project structure and schema

**Type:** Build prompt
**Date:**
**Goal:** Create the folder layout and the Pydantic output schema.

**Prompt:**

```
I'm building a clinic inbox triage agent in Python 3.12 using Streamlit, Pydantic v2, and Ollama (local LLMs). The project folder structure and virtual environment already exist. Do not recreate them.

Implement src/schema.py with Pydantic v2 models:
- Enums: Intent (appointment, refill, test_results, administrative, general_query, other), Urgency (routine, soon, urgent, emergency), RouteTo (on_call_clinician, pharmacist_or_ma, front_desk)
- RedFlag: is_red_flag (bool), reason (str or None)
- Contact: name, phone, email (all optional)
- FollowUpAnswer: question (str), answer (str)
- TriageResult: intent, urgency, route_to, suggested_specialty (optional), red_flag, follow_up (list of FollowUpAnswer), contact, summary (one sentence), draft_reply

Add docstrings and short field descriptions. Do not implement any LLM calls yet. Then write tests/test_schema.py with pytest covering valid input, invalid enum values, and missing required fields.
```

**Output / result:**
**Problem:**
**Follow-up prompt:**
**New result:**
**What I learned:**

---

## Entry 02: Synthetic dataset

**Type:** Build prompt
**Date:**
**Goal:** Generate 30 to 40 labeled synthetic patient messages (intent, urgency, red flag, route).

**Prompt:**

```
Implement src/red_flags.py with a rule-based emergency detector. It must not call any LLM.

Requirements:
1. check_red_flags(text: str) -> RedFlag (from src/schema.py). Normalize text (lowercase, strip punctuation, collapse whitespace) before matching.
2. Define pattern groups as a dict of category -> list of compiled regexes: cardiac, respiratory, neurological, severe_bleeding, anaphylaxis, overdose_poisoning, mental_health_crisis. Cover paraphrases, for example "can't catch my breath", "tightness in my chest", "face is drooping", "thinking of ending my life", not just exact keywords.
3. If any pattern matches, return is_red_flag=True with a reason naming the category and the matched phrase. If nothing matches, return is_red_flag=False.
4. Err on the side of over-triage: do not try to suppress negations in free text.
5. Add a module-level EMERGENCY_MESSAGE constant telling the patient to call emergency services immediately.
6. Write tests/test_red_flags.py with at least 8 messages per category that must trigger, 10 ordinary messages that must not (refills, billing, appointment requests, test results), and paraphrase cases. Use pytest parametrize.
```

**Output / result:**
**Problem:**
**Follow-up prompt:**

```
Testing found a missed case in src/red_flags.py: "my chest feels heavy" is not detected. Fix this by broadening the cardiac patterns to cover chest heaviness and pressure phrasings, for example: heavy chest, chest feels heavy, weight on my chest, pressure in my chest, squeezing or crushing chest, chest feels tight, pain in my left arm or jaw.

Then audit all categories for similar paraphrase gaps and add patterns for: sudden severe or "worst ever" headache, coughing up blood, passed out or unconscious, not breathing or turning blue, seizure or convulsion, can't feel my arm or leg, sudden vision loss, thoughts of hurting myself or suicide.

Add each of these phrasings as must-trigger test cases in tests/test_red_flags.py. Keep the existing must-not-trigger cases passing, and add these as must-not-trigger: "heavy menstrual bleeding question for my next visit" is acceptable to flag, but "need a refill", "reschedule my appointment", and "question about my bill" must not.
```

**New result:**
**What I learned:**

---

## Entry 03: Triage system prompt v1

**Type:** System prompt (triage LLM)
**Date:**
**Goal:** First working classification prompt returning strict JSON.

**Prompt:**

```
Implement src/intake_questions.py for adaptive intake. It must not call any LLM.

1. Define a Pydantic model Question: id (str), group (str), text (str), answer_type ("yes_no" or "text"), red_flag_if (optional str, e.g. "yes"), reason (optional str, shown when it triggers).

2. Define SYMPTOM_QUESTIONS: dict mapping symptom group -> list[Question], with 2 to 3 questions per group for: sore_throat, cough, fever, headache, abdominal_pain, rash, back_pain, dizziness, injury, medication_refill. Include red-flag yes/no questions where clinically sensible, for example:
   - sore_throat: fever? difficulty breathing or swallowing, or drooling (red flag)?
   - cough: shortness of breath (red flag)? coughing up blood (red flag)? how many days?
   - headache: sudden severe "worst headache of my life" (red flag)? stiff neck or confusion (red flag)?
   - abdominal_pain: severe constant pain (red flag)? vomiting blood or black stools (red flag)?
   - back_pain: numbness or weakness in legs, or loss of bladder control (red flag)?
   - rash: swelling of face, lips, or tongue (red flag)? fever?
   - medication_refill: how many days of medication are left? any side effects?

3. Define SYMPTOM_PATTERNS: dict group -> list of compiled regexes that detect each group in a message (include common phrasings, e.g. "sore throat", "throat hurts", "coughing", "temperature", "stomach ache", "tummy pain", "lower back pain", "feel dizzy", "need a refill").

4. get_follow_up_questions(message: str, max_questions: int = 3) -> list[Question]: normalize text, detect groups, collect their questions, put red-flag questions first, remove duplicates, cap at max_questions. Return [] if no group matches.

5. evaluate_answers(answers: list[tuple[Question, str]]) -> RedFlag (from src/schema.py): if any answer matches that question's red_flag_if (treat yes, y, yeah, yep, true, "yes i do" as yes; case-insensitive), return is_red_flag=True with a reason naming the question. Otherwise False. Unclear answers must not trigger.

6. to_follow_up_answers(answers) -> list[FollowUpAnswer] converting to the schema model.

7. Write tests/test_intake_questions.py: each group detected by at least 2 phrasings; cap of 3 enforced; red-flag questions ordered first; no match returns []; yes variants trigger; "no" and unclear answers do not; to_follow_up_answers output is valid.

Keep code typed and documented.
```

**Output / result:** (accuracy, red-flag recall, JSON validity)
**Problem:**
**Follow-up prompt:**
**New result:**
**What I learned:**

---

## Entry template (copy for each new entry)

## Entry XX: Title

**Type:** Build prompt / System prompt
**Date:**
**Goal:**

**Prompt:**

```
(paste exact prompt)
```

**Output / result:**
**Problem:**
**Follow-up prompt:**

```
(paste exact follow-up)
```

**New result:**
**What I learned:**

---

## Prompt version history (system prompts)

| Version | Change | Intent acc. | Urgency acc. | Red-flag recall | JSON valid |
|---------|--------|-------------|--------------|-----------------|------------|
| v1 | Baseline | | | | |
| v2 | | | | | |
| v3 | | | | | |