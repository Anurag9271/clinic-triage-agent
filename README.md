# Clinic Inbox Triage Agent

An AI-assisted triage pipeline for clinic patient messages. It asks adaptive follow-up questions, flags emergencies with rule-based checks, classifies each message with a local open-source LLM, routes it to the right staff queue, drafts a reply for human approval, and logs everything.

> **Disclaimer:** This is a prototype built for an assessment. It is not a medical device and does not provide medical advice. All data in this repository is synthetic.

## Why this project

Clinic front desks spend a large part of the day sorting messages: appointments, refills, test results, paperwork. Urgent messages can get buried. This project speeds up triage while keeping a human in control, and it runs fully on local models so patient data never leaves the machine.

## Workflow

1. **Message in:** patient fills an intake form (Streamlit).
2. **Adaptive follow-up:** a fixed symptom-to-questions table asks the mandatory questions; the LLM adds up to 2 context-specific ones.
3. **Red-flag check (rule-based):** emergency keywords bypass the AI and trigger immediate escalation.
4. **LLM extraction:** structured JSON (intent, urgency, specialty, contact) validated with Pydantic.
5. **Routing:** deterministic mapping from intent and urgency to a staff queue.
6. **Reply drafting:** the model drafts a reply.
7. **Staff approval:** staff edit, approve, or reject.
8. **Logging:** all steps stored in SQLite and shown in a dashboard.

## Routing rules

| Category | Routes to |
|---|---|
| Emergency or urgent clinical | On-call nurse or doctor |
| Prescription refill | Pharmacist or medical assistant |
| Administrative (sick notes, paperwork, billing) | Front desk |
| Appointment, test results, general query | Front desk; nurse if urgency is soon or higher |

## Tech stack

- Python, Pydantic, Streamlit, SQLite
- Ollama with local models: Llama 3.2 3B, Gemma 3 4B, Llama 3.1 8B or Qwen2.5 7B
- scikit-learn (TF-IDF + logistic regression baseline)
- Built with Antigravity IDE; prompts documented in [docs/prompt_log.md](docs/prompt_log.md)

## Setup

```bash
git clone https://github.com/Anurag9271/clinic-triage-agent.git
cd clinic-triage-agent
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

# Install Ollama from https://ollama.com, then pull a model
ollama pull llama3.2:3b

# Run the app
streamlit run app.py
```

## Evaluation

A synthetic set of 30 to 40 labeled messages is used to compare prompt versions and models.

| Model / method | Intent acc. | Urgency acc. | Red-flag recall | Latency (s) |
|---|---|---|---|---|
| TF-IDF + LogReg | | | | |
| Llama 3.2 3B | | | | |
| Gemma 3 4B | | | | |
| Llama 3.1 8B / Qwen2.5 7B | | | | |

Run the evaluation:

```bash
python eval/run_eval.py
```

## Project structure

```
clinic-triage-agent/
├── README.md
├── docs/
│   ├── project_brief.md
│   └── prompt_log.md
├── src/
│   ├── schema.py
│   ├── red_flags.py
│   ├── triage.py
│   ├── routing.py
│   ├── intake_questions.py
│   └── db.py
├── data/synthetic_messages.csv
├── eval/run_eval.py
├── app.py
├── requirements.txt
└── .gitignore
```

## Safety and ethics

- Synthetic data only.
- Rule-based emergency escalation runs before any LLM call.
- Safety-critical follow-up questions come from a fixed table, not the model.
- A human approves every reply before it is sent.

## Demo

Video walkthrough: (add link)

## Author

Anurag