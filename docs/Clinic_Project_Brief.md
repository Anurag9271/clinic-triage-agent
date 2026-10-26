# Clinic Inbox Triage Agent: Project Brief

AI Growth & Automation Associate assessment (Task 1) | Domain: Healthcare | Area: Automation + tool innovation

## 1. Overview

The Clinic Inbox Triage Agent is an automated pipeline that reads incoming patient messages (from a web form or email), understands what the patient needs, flags urgent or dangerous situations, drafts a reply for staff to approve, and logs everything for review. It runs entirely on open-source models on a local machine, so patient data never leaves the device.

All data in this project is synthetic. The tool is a prototype and not a medical device or medical advice.

## 2. Problem

Clinic front desks spend a large share of their day reading and sorting messages: appointment requests, prescription refills, test result queries, and general questions. Urgent messages can get buried, and replies are slow and repetitive. The goal is to speed up triage while keeping a human in control and never letting a dangerous message depend on model behavior.

## 3. Solution: the workflow

1. **Message in:** a patient fills a Streamlit intake form (simulating a web form or email).
2. **Adaptive follow-up:** on Next, a fixed symptom-to-questions table asks the mandatory questions (for a sore throat: fever? difficulty breathing?) and the LLM adds up to 2 context-specific ones. One round, 3 questions at most.
3. **Red-flag check (rule-based):** keywords in the message or the answers, such as chest pain or difficulty breathing, bypass the AI and trigger an immediate "call emergency services" escalation.
4. **LLM extraction:** an open-source model returns structured JSON: intent, urgency, suggested specialty, and contact details.
5. **Routing:** a deterministic mapping from intent and urgency assigns the queue: on-call clinician, pharmacist or medical assistant, or front desk.
6. **Reply drafting:** the model drafts a polite, suitable reply based on the intent and urgency.
7. **Staff approval (human in the loop):** staff edit, approve, or reject the draft before anything is sent.
8. **Logging:** every message, answer, classification, route, draft, and decision is stored in SQLite (or Google Sheets) and shown in a dashboard with one queue per role.

## 4. Output schema (draft)

- intent: appointment, refill, test\_results, administrative, general\_query, other
- urgency: routine, soon, urgent, emergency
- route\_to: on\_call\_clinician, pharmacist\_or\_ma, front\_desk (set by rules from intent and urgency, not by the LLM)
- suggested\_specialty: for example GP, cardiology, dermatology
- red\_flag: true or false, with the reason
- follow\_up: the questions asked and the patient's answers
- contact: name, phone, email (if present)
- summary: one sentence
- draft\_reply: text for staff review

Validated with Pydantic so malformed outputs are caught and retried.

## 5. Routing by action type

Routing is a deterministic mapping from intent and urgency, not a free-form LLM decision, so it is easy to test and explain. The dashboard shows one queue per role.

| Category | Routes to |
| --- | --- |
| Emergency or urgent clinical | On-call nurse or doctor |
| Prescription refill | Pharmacist or medical assistant |
| Administrative (sick notes, paperwork, billing) | Front desk |
| Appointment, test results, general query | Front desk; nurse if urgency is soon or higher |

## 6. Adaptive intake (two-way triage)

Adaptive intake turns the classifier into an agent: the form asks follow-up questions before submission, so the clinic receives complete clinical information upfront.

- Triggered by a Next button, not every keystroke, because local models take a few seconds per call.
- Safety-critical questions come from a fixed symptom-to-questions table; the LLM adds at most 2 context-specific questions.
- One round, 3 questions at most.
- A red-flag answer, such as yes to difficulty breathing, triggers emergency escalation immediately.
- Fallback: if time runs short, ship the rule-based question table only.

## 7. AI tools and models

| Component | Choice | Why |
| --- | --- | --- |
| Local runtime | Ollama | Easy setup, supports JSON-schema structured output |
| Small LLM | Llama 3.2 3B | Fast baseline |
| Mid LLM | Gemma 3 4B | Fits fully in 6 GB VRAM |
| Large LLM | Llama 3.1 8B or Qwen2.5 7B | Strongest accuracy within hardware limits |
| Classical baseline | TF-IDF + logistic regression (scikit-learn) | Shows when an LLM is worth the cost |
| AI coding assistant | Claude | Scaffolding, debugging, prompt refinement |
| UI | Streamlit | Quick, clean demo |

## 8. Evaluation harness (the differentiator)

- Build 30 to 40 synthetic patient messages with labeled ground truth (intent, urgency, red flag, route).
- Run every prompt version and every model against the set.
- Report: intent accuracy, urgency accuracy, routing accuracy, red-flag recall, JSON validity rate, and latency.
- Compare urgency accuracy with and without the follow-up answers to show the value of adaptive intake.
- Show prompt v1 versus v2 versus v3 improvements, and a final model comparison table including the TF-IDF baseline.

This gives the video a data-backed story about iteration rather than just a demo.

## 9. Safety and ethics

- Synthetic data only, stated clearly in the README.
- Rule-based emergency escalation runs before any LLM call.
- Safety-critical follow-up questions come from a fixed table, not the model.
- Human approval required before any reply is sent.
- Visible disclaimer: not a medical device, not medical advice.


