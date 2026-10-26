# Clinic Inbox Triage Agent

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/frontend-Streamlit-red.svg)](https://streamlit.io/)
[![Ollama](https://img.shields.io/badge/runtime-Ollama-black.svg)](https://ollama.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An AI-assisted, privacy-first clinical inbox triage agent for primary care clinics. It provides adaptive symptom screening, deterministic emergency red-flag escalation, local LLM classification and response drafting, contact anti-hallucination grounding, and human-in-the-loop staff review.

> **Disclaimer:** This software is a functional prototype built for demonstration and evaluation. It is **not** a certified medical device and does **not** provide clinical diagnosis or medical advice. All data in this repository is synthetic.

---

## Key Features

1. **Deterministic Safety-First Red-Flag Screening:**
   - Scans incoming patient messages across 7 critical emergency categories (cardiac, respiratory, neurological, severe hemorrhage, anaphylaxis, overdose/poisoning, mental health crisis).
   - Life-threatening symptoms immediately trigger emergency advisory guidance, bypass the LLM entirely, and escalate directly to the On-Call Clinician.
2. **Adaptive Intake (Two-Way Triage):**
   - Automatically detects relevant symptoms (e.g. sore throat, cough, fever) and prompts the patient with structured screening questions before submission.
   - Positive answers to safety-critical follow-ups immediately trigger emergency escalation.
3. **Explicit Contact Capture & Hallucination Grounding:**
   - Captures optional contact details (Name, Phone, Email, Preferred Method) via form inputs with real-time validation.
   - Anti-hallucination engine verifies any LLM-extracted contact details against the raw message text.
   - Field-by-field merge precedence: **Form Input > Regex Extraction > Grounded LLM**. Warns staff if no contact method is available for dispatch.
4. **Structured Classification & Routing:**
   - Extracts intent (`appointment`, `refill`, `test_results`, `administrative`, `general_query`, `other`), urgency (`routine`, `soon`, `urgent`, `emergency`), and suggested specialty.
   - Deterministic rule-based routing to specific clinical queues: **On-Call Clinician**, **Pharmacist / MA**, or **Front Desk**.
5. **Human-in-the-Loop Staff Review Queue:**
   - Clinical staff can filter by role, inspect patient messages, review clinical summaries, edit preliminary draft responses, and approve or reject tickets.
6. **Operational Metrics Dashboard:**
   - Live aggregate volume tiles, urgency distributions, queue breakdowns, and review statuses backed by local SQLite persistence.
7. **Fast Demo Mode (Zero-GPU / Instant Testing):**
   - Toggle fast demo mode in the sidebar to test the entire clinical workflow using a deterministic keyword-based mock classifier with near-zero latency, without requiring Ollama or a GPU.

---

## Routing Architecture

| Category / Urgency | Target Queue | Responsible Staff |
|---|---|---|
| **Emergency or Urgent Clinical** | `on_call_clinician` | On-Call Physician or Nurse Practitioner |
| **Prescription Refills** | `pharmacist_or_ma` | Clinical Pharmacist or Medical Assistant |
| **Administrative** (sick notes, records, billing) | `front_desk` | Reception & Practice Administration |
| **Appointments & Test Results (Routine)** | `front_desk` | Front Desk Staff |
| **Appointments & Test Results (Soon / Urgent)** | `on_call_clinician` | Triage Nurse |

---

## Prerequisites

Before running the application, make sure you have:
- **Python 3.10+** (tested on Python 3.11, 3.12, and 3.14).
- **Git** installed on your system.
- *(Optional for Full LLM Mode)* **[Ollama](https://ollama.com)** installed and running. If you do not have Ollama, you can run the app in **Fast Demo Mode** right out of the box!

---

## Step-by-Step Installation & Setup

### 1. Clone the Repository

Open your terminal or command prompt and clone the repository:

```bash
git clone https://github.com/Anurag9271/clinic-triage-agent.git
cd clinic-triage-agent
```

### 2. Create and Activate a Virtual Environment

- **On Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```
  *(If execution policy blocks scripts, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

- **On Windows (Command Prompt `cmd`):**
  ```cmd
  python -m venv .venv
  .venv\Scripts\activate.bat
  ```

- **On macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 3. Install Python Dependencies

Upgrade `pip` and install the required packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## How to Run the Application

You can run the triage portal in two different modes:

### Option A: Local LLM Mode (Recommended for AI testing)

1. **Start the Ollama server** in a terminal window (if not already running as a background service):
   ```bash
   ollama serve
   ```

2. **Pull the default local model** (Llama 3.2 3B):
   ```bash
   ollama pull llama3.2:3b
   ```
   *(Optional supported models: `llama3.2:1b`, `llama3.1:8b`, `gemma2:2b`, `mistral:7b`)*.

3. **Launch the Streamlit web portal:**
   ```bash
   streamlit run app.py
   ```
   The portal will open automatically in your browser at `http://localhost:8501`.

---

### Option B: Fast Demo Mode (Zero-GPU, No Ollama Required)

If you do not have Ollama installed or are on a battery-powered / CPU-only machine:

1. **Launch Streamlit directly:**
   ```bash
   streamlit run app.py
   ```
2. In the left sidebar under **⚙️ Triage Settings**, check:
   `☑ Fast demo mode (mock LLM, no model call)`
3. A yellow `⚡ MOCK MODE` badge will appear. All rule-based red-flag checks, adaptive screening questions, contact extraction, database logging, and staff queues run with sub-millisecond execution!

---

## Using the Portal

The application features three integrated clinical workflow tabs:

1. **📝 Patient Intake:**
   - **Step 1:** Enter a patient message or pick one of 6 representative clinical scenarios from the sidebar (e.g. cardiac emergency, refill request, multi-symptom fever). Optionally enter patient name, phone, email, and preferred contact channel.
   - **Step 2:** If acute symptoms are detected (e.g. sore throat, cough), complete the adaptive follow-up screening questions.
   - **Step 3:** View the consolidated triage decision: urgency badge, queue assignment, clinical summary, verified contact card, and draft reply. Red flags immediately show emergency escalation guidance.
2. **📥 Staff Queue:**
   - Filter inbox items by discipline (`On-Call Clinician`, `Pharmacist / MA`, `Front Desk`).
   - Emergency tickets are pinned with bold `🚨 [RED FLAG]` headers.
   - Review patient context, edit preliminary replies directly in the text area, and click **Approve & Send** or **Reject**.
   - If the patient provided no phone or email, an alert informs staff that no contact method is available for outbound dispatch.
3. **📊 Dashboard:**
   - Real-time clinical operations metrics: total message volume, pending queue counts, red-flag emergency frequency, and median latency.
   - Live distribution charts tracking triage categories, urgency tiers, and review statuses stored in local SQLite.

---

## Running the Automated Test Suite

The project includes an end-to-end test suite containing **402 automated tests** covering models, schema validators, red-flag patterns, adaptive questions, contact regex/grounding, database operations, and headless Streamlit UI interactions (`streamlit.testing.v1.AppTest`).

Run the test suite:

```bash
pytest
```

To run with verbose output or on a specific test module:

```bash
# Run contact extraction & validation tests
pytest tests/test_contact.py -v

# Run headless Streamlit app UI tests
pytest tests/test_app.py -v

# Run rule-based emergency safety tests
pytest tests/test_red_flags.py -v
```

---

## Running the Evaluation Benchmark

To benchmark inference accuracy, red-flag recall, and latency across local models and prompt versions against the labeled synthetic dataset (`data/synthetic_messages.csv`):

```bash
# Run LLM evaluation harness
python eval/run_eval.py

# Run classical baseline (TF-IDF + Logistic Regression)
python eval/baseline.py
```

Evaluation metrics are saved directly to `results/summary.csv` and detailed prediction CSVs.

---

## Performance & Latency Tips

- **GPU Sleep State Latency:** On laptops with switchable graphics (NVIDIA Optimus / AMD), waking the GPU from sleep can add 15–30 seconds on the very first inference call.
- **Persistent Keep-Alive:** The Ollama client in [`src/triage.py`](src/triage.py) uses `keep_alive="30m"` (`DEFAULT_KEEP_ALIVE`), keeping model weights resident in GPU VRAM across consecutive messages.
- **Token Budgeting:** Inference calls specify `num_predict=300` (`DEFAULT_NUM_PREDICT`) to bound generation latency and prevent runaway responses.
- **Verify Active Model:** Run `ollama ps` in your terminal to verify that the model is loaded in VRAM (`100% GPU`).

---

## Project Structure

```
clinic-triage-agent/
├── README.md                          # Project documentation and setup guide
├── requirements.txt                   # Pinned Python package dependencies
├── app.py                             # Streamlit interactive triage application
├── docs/
│   └── Clinic_Project_Brief.md        # Clinical specification and project brief
├── src/
│   ├── __init__.py                    # Package exports
│   ├── schema.py                      # Pydantic v2 domain schemas (TriageResult, Contact, etc.)
│   ├── red_flags.py                   # Rule-based emergency red-flag pattern matcher
│   ├── contact.py                     # Regex extraction, grounding, and contact merge logic
│   ├── triage.py                      # Core triage orchestration & Ollama interface
│   ├── intake_questions.py            # Adaptive screening question bank and answer evaluation
│   ├── routing.py                     # Deterministic queue assignment rules
│   ├── prompts.py                     # Versioned clinical system prompts (v1, etc.)
│   ├── mock_llm.py                    # Deterministic keyword mock LLM for fast demo mode
│   └── db.py                          # SQLite persistence layer and query functions
├── tests/
│   ├── test_app.py                    # Headless Streamlit AppTest integration tests
│   ├── test_contact.py                # Regex extraction, validation, and grounding unit tests
│   ├── test_red_flags.py              # Positive & negative emergency safety trigger tests
│   ├── test_triage.py                 # End-to-end triage pipeline and fallback tests
│   ├── test_schema.py                 # Pydantic model validation tests
│   ├── test_routing.py                # Queue routing rule tests
│   ├── test_intake_questions.py       # Adaptive intake questions & answer evaluation tests
│   ├── test_db.py                     # SQLite CRUD, stats, and concurrency tests
│   ├── test_mock_llm.py               # Mock LLM classifier tests
│   ├── test_dataset.py                # Synthetic message dataset integrity tests
│   └── test_eval_metrics.py           # Evaluation calculation tests
├── data/
│   └── synthetic_messages.csv         # 40 labeled synthetic clinical benchmark messages
└── eval/
    ├── run_eval.py                    # Multi-model evaluation benchmark harness
    ├── baseline.py                    # TF-IDF + Logistic Regression classical baseline
    └── metrics.py                     # Evaluation scoring & metrics calculation
```

---

## Safety and Ethics

- **Synthetic Data Only:** All patient messages, contact details, and clinical data used in this project are entirely synthetic.
- **Deterministic Escalation:** Safety-critical red flags run prior to any LLM inference, ensuring emergencies never depend on model availability or behavior.
- **Grounding Safety:** Contact details come from validated form inputs first; any LLM-extracted contact data is verified against the raw message text to eliminate hallucinated phone numbers or names.
- **Human in the Loop:** No drafted reply is ever dispatched without explicit review and approval by authorized clinical or administrative staff.
- **Local & Private:** Zero external API calls. Patient health inquiries remain strictly on the local device.

---

## Author

**Anurag**
- GitHub: [@Anurag9271](https://github.com/Anurag9271)
- Repository: [clinic-triage-agent](https://github.com/Anurag9271/clinic-triage-agent)