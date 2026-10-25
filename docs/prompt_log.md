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
I'm building a clinic inbox triage agent in Python using Streamlit, Pydantic, and Ollama (local LLMs). Create the project folder structure and a Pydantic schema in src/schema.py with these fields: intent (appointment, refill, test_results, administrative, general_query, other), urgency (routine, soon, urgent, emergency), route_to, suggested_specialty, red_flag (bool + reason), follow_up, contact, summary, draft_reply. Don't implement the LLM calls yet.
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
(paste here)
```

**Output / result:**
**Problem:**
**Follow-up prompt:**
**New result:**
**What I learned:**

---

## Entry 03: Triage system prompt v1

**Type:** System prompt (triage LLM)
**Date:**
**Goal:** First working classification prompt returning strict JSON.

**Prompt:**

```
(paste here)
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