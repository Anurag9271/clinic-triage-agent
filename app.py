"""Clinic Inbox Triage Agent - Streamlit Application.

Interactive clinical portal featuring:
1. Patient Intake: Adaptive multi-step follow-up and instant red-flag detection.
2. Staff Queue: Clinical review, draft reply editing, approval, and rejection.
3. Dashboard: Real-time operational metrics and triage distribution charts.
"""

from __future__ import annotations

import logging
from typing import Final, Optional

import pandas as pd
import streamlit as st

from src.db import (
    get_stats,
    init_db,
    list_records,
    save_record,
    update_status,
)
from src.intake_questions import (
    Question,
    evaluate_answers,
    get_follow_up_questions,
    to_follow_up_answers,
)
from src.prompts import PROMPTS
from src.red_flags import EMERGENCY_MESSAGE
from src.schema import Intent, RecordStatus, RouteTo, TriageRecord, Urgency
from src.triage import triage_message

logger = logging.getLogger(__name__)

# ============================================================================
# Constants & Sample Data
# ============================================================================

MODELS: Final[list[str]] = [
    "llama3.2:3b",
    "llama3.2:1b",
    "llama3.1:8b",
    "gemma2:2b",
    "mistral:7b",
]

SAMPLE_MESSAGES: Final[dict[str, str]] = {
    "🚨 Emergency: Chest pain & arm radiation": (
        "I have a heavy weight on my chest and severe pain shooting down my left arm. "
        "I can barely catch my breath. Need help now!"
    ),
    "💊 Prescription Refill: Low supply": (
        "I take metformin 500mg twice daily and only have 2 pills left before I run out "
        "completely tomorrow night. Please renew as soon as possible. Arthur Davis, 555-0144."
    ),
    "🤒 Two Symptoms: Sore throat & worsening cough": (
        "I have had a severe sore throat and a worsening cough with a fever of 101.5 for the past "
        "three days. What should I do? Contact: Maria Garcia, 555-0128, maria.g@example.com."
    ),
    "📅 Routine Appointment: Annual physical": (
        "Hello, I would like to schedule my annual preventive physical examination with Dr. Evans "
        "sometime next month. Any weekday morning works. Contact: Sarah Jenkins, 555-0112, sarah.j@example.com."
    ),
    "🔬 Lab Results: Lipid panel review": (
        "Could Dr. Patel please review my routine lipid panel results from last Friday? "
        "I see the numbers in the portal but would like an explanation when available. Lisa Ray, 555-0155."
    ),
    "📋 Administrative: Work sick note": (
        "I missed work yesterday and today due to severe illness. My employer requires a doctor's "
        "sick note by 5 PM tomorrow or I face disciplinary action. Elena Rostova, 555-0182, elena.r@example.com."
    ),
}

ROLE_LABELS: Final[dict[str, str]] = {
    "all": "All Staff Roles",
    "on_call_clinician": "On-Call Clinician (Doctor / Nurse)",
    "pharmacist_or_ma": "Pharmacist / Medical Assistant",
    "front_desk": "Front Desk / Reception",
}


# ============================================================================
# App Setup & Custom Styling
# ============================================================================

st.set_page_config(
    page_title="Clinic Inbox Triage Agent",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for clinical design aesthetics, badges, and card containers
st.markdown(
    """
    <style>
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 14px 18px;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetricLabel"] {
        font-weight: 600;
        font-size: 0.9rem;
        color: #475569;
    }
    div[data-testid="stMetricValue"] {
        font-weight: 700;
        color: #0f172a;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-right: 6px;
    }
    .badge-emergency {
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #fca5a5;
    }
    .badge-urgent {
        background-color: #ffedd5;
        color: #9a3412;
        border: 1px solid #fdba74;
    }
    .badge-soon {
        background-color: #fef3c7;
        color: #92400e;
        border: 1px solid #fcd34d;
    }
    .badge-routine {
        background-color: #dcfce7;
        color: #166534;
        border: 1px solid #86efac;
    }
    .badge-route {
        background-color: #e0f2fe;
        color: #0369a1;
        border: 1px solid #7dd3fc;
    }
    .badge-intent {
        background-color: #f1f5f9;
        color: #334155;
        border: 1px solid #cbd5e1;
    }

    /* Card Panels */
    .clinical-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .clinical-card-header {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 10px;
        border-bottom: 1px solid #f1f5f9;
        padding-bottom: 6px;
    }
    .sidebar-disclaimer {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 10px 12px;
        border-radius: 4px;
        font-size: 0.82rem;
        color: #92400e;
        line-height: 1.4;
        margin-bottom: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _init_app_session_state() -> None:
    """Initialize persistent Streamlit session state keys."""
    if "step" not in st.session_state:
        st.session_state.step = 1
    if "message_input" not in st.session_state:
        st.session_state.message_input = ""
    if "follow_up_qs" not in st.session_state:
        st.session_state.follow_up_qs = []
    if "triage_record" not in st.session_state:
        st.session_state.triage_record = None
    if "triage_latency" not in st.session_state:
        st.session_state.triage_latency = 0.0
    if "triage_error" not in st.session_state:
        st.session_state.triage_error = None


# Ensure database and table structures are initialized on startup
init_db()
_init_app_session_state()


# ============================================================================
# Sidebar Configuration
# ============================================================================

st.sidebar.markdown(
    """
    <div class="sidebar-disclaimer">
        <strong>⚠️ Prototype Notice:</strong><br>
        Prototype. Synthetic data only. Not a medical device and not medical advice.
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("⚙️ Triage Settings")

# Model selection
model_options = MODELS
selected_model = st.sidebar.selectbox(
    "Inference Model",
    options=model_options,
    index=0,
    help="Ollama LLM used for structured intent extraction and response drafting.",
)

# Prompt version selection
prompt_versions = list(PROMPTS.keys())
selected_prompt_version = st.sidebar.selectbox(
    "Prompt Version",
    options=prompt_versions,
    index=0,
    help="Version of system instruction prompt used by the triage agent.",
)

st.sidebar.divider()
st.sidebar.subheader("📋 Sample Messages")


def _on_sample_selection_change() -> None:
    """Callback when a sample message is selected from the sidebar."""
    selected_sample = st.session_state.sample_selector
    if selected_sample and selected_sample != "-- Custom / Blank --":
        st.session_state.message_input = SAMPLE_MESSAGES[selected_sample]
        st.session_state.step = 1
        st.session_state.follow_up_qs = []
        st.session_state.triage_record = None
        st.session_state.triage_error = None


st.sidebar.selectbox(
    "Load Sample Patient Message",
    options=["-- Custom / Blank --"] + list(SAMPLE_MESSAGES.keys()),
    key="sample_selector",
    on_change=_on_sample_selection_change,
    help="Select a representative synthetic message to fill the patient message box.",
)

st.sidebar.markdown(
    """
    <div style="font-size: 0.78rem; color: #64748b; margin-top: 24px;">
        Local Ollama Engine | Zero external API calls<br>
        Deterministic safety escalation active.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# Main Navigation Tabs
# ============================================================================

tab_intake, tab_queue, tab_dashboard = st.tabs(
    ["📝 Patient Intake", "📥 Staff Queue", "📊 Dashboard"]
)


# ============================================================================
# Helper Functions
# ============================================================================


def _render_urgency_badge(urgency: Urgency | str) -> str:
    """Generate HTML badge for urgency level."""
    urg_val = urgency.value if hasattr(urgency, "value") else str(urgency).lower()
    badge_class = f"badge-{urg_val}" if urg_val in ("emergency", "urgent", "soon", "routine") else "badge-routine"
    return f'<span class="badge {badge_class}">{urg_val.upper()}</span>'


def _render_route_badge(route_to: RouteTo | str) -> str:
    """Generate HTML badge for routing destination."""
    route_val = route_to.value if hasattr(route_to, "value") else str(route_to).lower()
    label = ROLE_LABELS.get(route_val, route_val.replace("_", " ").title())
    return f'<span class="badge badge-route">{label}</span>'


def _execute_triage(
    message: str,
    answers: Optional[list[tuple[Question, str]]] = None,
) -> None:
    """Run triage pipeline, save record, and store result in session state."""
    with st.spinner("Classifying message with clinical triage pipeline..."):
        try:
            record, latency_s = triage_message(
                message=message,
                answers=answers,
                model=selected_model,
                prompt_version=selected_prompt_version,
            )
            save_record(
                record=record,
                model=selected_model,
                prompt_version=selected_prompt_version,
                latency_s=latency_s,
            )
            st.session_state.triage_record = record
            st.session_state.triage_latency = latency_s
            st.session_state.triage_error = None
            st.session_state.step = 3
            st.rerun()
        except Exception as exc:
            st.session_state.triage_error = str(exc)
            st.error(
                f"**Triage Failed:** {exc}\n\n"
                f"Please verify that Ollama is running and `{selected_model}` is pulled locally."
            )


# ============================================================================
# Tab 1: Patient Intake
# ============================================================================

with tab_intake:
    st.header("Patient Inbox Message Intake")
    st.caption("AI-assisted multi-step clinical intake with adaptive symptom screening.")

    # Step progress indicator
    step = st.session_state.step
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.markdown(
            f"**Step 1: Patient Message** {'✅' if step > 1 else '👉'}"
        )
    with col_s2:
        st.markdown(
            f"**Step 2: Adaptive Questions** {'✅' if step > 2 else ('👉' if step == 2 else '⏳')}"
        )
    with col_s3:
        st.markdown(
            f"**Step 3: Triage Result** {'👉' if step == 3 else '⏳'}"
        )
    st.divider()

    # ------------------------------------------------------------------------
    # Step 1: Input Patient Message
    # ------------------------------------------------------------------------
    if step == 1:
        st.subheader("Step 1: Enter Patient Message")
        message_input = st.text_area(
            "Incoming Patient Message",
            value=st.session_state.message_input,
            key="msg_text_area",
            height=140,
            placeholder="Type patient inquiry or select a sample message from the sidebar...",
            help="Raw communication received from the patient portal or messaging channel.",
        )

        col_next, _ = st.columns([1, 4])
        with col_next:
            if st.button("Next ➔", type="primary", use_container_width=True):
                cleaned_msg = message_input.strip()
                if not cleaned_msg:
                    st.warning("Please enter a patient message to proceed.")
                else:
                    st.session_state.message_input = cleaned_msg
                    # Check for adaptive follow-up questions
                    questions = get_follow_up_questions(cleaned_msg)
                    if questions:
                        st.session_state.follow_up_qs = questions
                        st.session_state.step = 2
                        st.rerun()
                    else:
                        # No follow-up questions required; proceed directly to triage submission
                        _execute_triage(cleaned_msg, answers=None)

    # ------------------------------------------------------------------------
    # Step 2: Adaptive Intake Follow-up Questions
    # ------------------------------------------------------------------------
    elif step == 2:
        st.subheader("Step 2: Adaptive Clinical Screening Questions")
        st.info(
            "Relevant symptoms were detected in the patient message. "
            "Please answer the following adaptive screening questions:"
        )

        # Render each question
        answers: list[tuple[Question, str]] = []
        for i, q in enumerate(st.session_state.follow_up_qs):
            st.markdown(f"**Question {i+1}:** {q.text}")
            if q.answer_type == "yes_no":
                user_val = st.radio(
                    f"Answer for {q.id}",
                    options=["No", "Yes"],
                    index=0,
                    horizontal=True,
                    key=f"ans_widget_{q.id}",
                    label_visibility="collapsed",
                )
            else:
                user_val = st.text_input(
                    f"Answer for {q.id}",
                    placeholder="Type details or duration...",
                    key=f"ans_widget_{q.id}",
                    label_visibility="collapsed",
                )
            answers.append((q, str(user_val)))
            st.write("")

        col_back, col_submit = st.columns([1, 2])
        with col_back:
            if st.button("⬅ Back to Message", use_container_width=True):
                st.session_state.step = 1
                st.rerun()
        with col_submit:
            if st.button("Submit for Triage ➔", type="primary", use_container_width=True):
                _execute_triage(st.session_state.message_input, answers=answers)

    # ------------------------------------------------------------------------
    # Step 3: Triage Outcome & Review
    # ------------------------------------------------------------------------
    elif step == 3:
        record: Optional[TriageRecord] = st.session_state.triage_record
        if record is None:
            st.error("No triage result found in session. Please start a new message.")
            if st.button("Start New Message"):
                st.session_state.step = 1
                st.rerun()
        else:
            result = record.result

            # Red Flag Alert Banner (Emergency Message at top if red flag triggers)
            if result.red_flag.is_red_flag or result.urgency == Urgency.EMERGENCY:
                st.error(
                    f"### 🚨 EMERGENCY ADVISORY\n\n"
                    f"**{EMERGENCY_MESSAGE}**\n\n"
                    f"*Clinical Red Flag Trigger:* {result.red_flag.reason or 'Critical acute symptom threshold reached.'}"
                )

            # Metadata bar (Urgency badge, Route badge, Intent, Specialty)
            st.markdown(
                f"""
                <div class="clinical-card">
                    <div style="display: flex; align-items: center; flex-wrap: wrap; gap: 8px; margin-bottom: 8px;">
                        {_render_urgency_badge(result.urgency)}
                        {_render_route_badge(result.route_to)}
                        <span class="badge badge-intent">INTENT: {result.intent.value.upper()}</span>
                        {f'<span class="badge badge-intent">SPECIALTY: {result.suggested_specialty}</span>' if result.suggested_specialty else ''}
                    </div>
                    <div style="font-size: 0.85rem; color: #64748b;">
                        Record ID: <code>{record.id}</code> &nbsp;|&nbsp; 
                        Received: {record.received_at.strftime('%Y-%m-%d %H:%M:%S UTC')} &nbsp;|&nbsp; 
                        Latency: {st.session_state.triage_latency:.2f}s
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Summary and Extracted Contact Cards
            col_summ, col_contact = st.columns([3, 2])
            with col_summ:
                st.markdown(
                    f"""
                    <div class="clinical-card">
                        <div class="clinical-card-header">📋 Clinical Summary</div>
                        <p style="margin: 0; font-size: 0.95rem; color: #1e293b;">{result.summary}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col_contact:
                contact = result.contact
                name_str = contact.name or "Not provided"
                phone_str = contact.phone or "Not provided"
                email_str = contact.email or "Not provided"
                st.markdown(
                    f"""
                    <div class="clinical-card">
                        <div class="clinical-card-header">👤 Extracted Patient Contact</div>
                        <div style="font-size: 0.9rem; line-height: 1.5; color: #334155;">
                            <strong>Name:</strong> {name_str}<br>
                            <strong>Phone:</strong> {phone_str}<br>
                            <strong>Email:</strong> {email_str}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Follow-up Answers (if any)
            if result.follow_up:
                with st.expander("🔍 View Intake Follow-Up Q&A Details", expanded=False):
                    for item in result.follow_up:
                        st.markdown(f"- **Q:** {item.question}\n  **A:** `{item.answer}`")

            # Draft Reply Box
            st.markdown(
                """
                <div class="clinical-card">
                    <div class="clinical-card-header">💬 Preliminary Draft Reply (For Staff Review)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.info(result.draft_reply)

            # Reset Button
            st.divider()
            if st.button("Start new message", type="primary"):
                st.session_state.step = 1
                st.session_state.message_input = ""
                st.session_state.follow_up_qs = []
                st.session_state.triage_record = None
                st.session_state.triage_latency = 0.0
                st.session_state.triage_error = None
                st.rerun()


# ============================================================================
# Tab 2: Staff Queue
# ============================================================================

with tab_queue:
    st.header("Staff Clinical Review Queue")
    st.caption("Review, edit, and approve pending triaged patient messages.")

    # Filter by role selectbox
    queue_col, _ = st.columns([2, 3])
    with queue_col:
        role_filter_choice = st.selectbox(
            "Filter by Queue Role",
            options=["all", "on_call_clinician", "pharmacist_or_ma", "front_desk"],
            format_func=lambda x: ROLE_LABELS.get(x, x),
            help="Filter pending messages by assigned staff discipline.",
        )

    selected_route = None if role_filter_choice == "all" else role_filter_choice
    pending_records = list_records(status="pending", route_to=selected_route)

    if not pending_records:
        st.info("🎉 No pending records in this queue.")
    else:
        st.write(f"Showing **{len(pending_records)}** pending message(s):")

        for r in pending_records:
            is_red_flag = r.result.red_flag.is_red_flag or r.result.urgency == Urgency.EMERGENCY
            marker = "🚨 [RED FLAG] " if is_red_flag else ""
            urgency_str = r.result.urgency.value.upper()
            intent_str = r.result.intent.value.replace("_", " ").title()
            timestamp_str = r.received_at.strftime("%Y-%m-%d %H:%M:%S UTC")

            expander_title = f"{marker}[{urgency_str}] {intent_str} — {timestamp_str}"

            with st.expander(expander_title, expanded=is_red_flag):
                if is_red_flag:
                    st.error(
                        f"🚨 **Emergency Red Flag:** {r.result.red_flag.reason or EMERGENCY_MESSAGE}"
                    )

                # Message details
                st.markdown("**Original Patient Message:**")
                st.info(r.original_message)

                col_det1, col_det2 = st.columns(2)
                with col_det1:
                    st.markdown(f"**Clinical Summary:** {r.result.summary}")
                    st.markdown(f"**Assigned Queue:** `{r.result.route_to.value}`")
                    if r.result.suggested_specialty:
                        st.markdown(f"**Suggested Specialty:** `{r.result.suggested_specialty}`")
                with col_det2:
                    c = r.result.contact
                    st.markdown(
                        f"**Patient Contact:** {c.name or 'N/A'} &nbsp;|&nbsp; "
                        f"**Phone:** {c.phone or 'N/A'} &nbsp;|&nbsp; "
                        f"**Email:** {c.email or 'N/A'}"
                    )

                # Intake Q&A if present
                if r.result.follow_up:
                    st.markdown("**Intake Follow-Up Q&A:**")
                    for qa in r.result.follow_up:
                        st.markdown(f"- **Q:** {qa.question}\n  **A:** *{qa.answer}*")

                st.markdown("**Staff Reply (editable):**")
                edited_reply = st.text_area(
                    "Edit draft response before patient dispatch",
                    value=r.final_reply or r.result.draft_reply,
                    key=f"reply_edit_{r.id}",
                    height=100,
                    label_visibility="collapsed",
                )

                col_approve, col_reject, _ = st.columns([1, 1, 3])
                with col_approve:
                    if st.button("Approve & Send", key=f"btn_app_{r.id}", type="primary", use_container_width=True):
                        update_status(
                            record_id=r.id,
                            status="approved",
                            final_reply=edited_reply.strip(),
                        )
                        st.success("Record approved successfully!")
                        st.rerun()

                with col_reject:
                    if st.button("Reject", key=f"btn_rej_{r.id}", use_container_width=True):
                        update_status(
                            record_id=r.id,
                            status="rejected",
                        )
                        st.warning("Record rejected.")
                        st.rerun()


# ============================================================================
# Tab 3: Operational Dashboard
# ============================================================================

with tab_dashboard:
    st.header("Triage Operations & Analytics Dashboard")
    st.caption("Live aggregate metrics and distribution charts from clinic database.")

    stats = get_stats()

    # Four Metric Tiles
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            label="Total Messages",
            value=stats.get("total_records", 0),
        )
    with m2:
        pending_count = stats.get("by_status", {}).get("pending", 0)
        st.metric(
            label="Pending Review",
            value=pending_count,
        )
    with m3:
        st.metric(
            label="Emergency Red Flags",
            value=stats.get("red_flag_count", 0),
        )
    with m4:
        avg_lat = stats.get("average_latency", 0.0)
        st.metric(
            label="Avg Latency",
            value=f"{avg_lat:.2f} s",
        )

    st.divider()

    # Distribution Bar Charts
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Counts by Staff Route")
        route_data = {
            "Front Desk": stats.get("by_route_to", {}).get("front_desk", 0),
            "On-Call Clinician": stats.get("by_route_to", {}).get("on_call_clinician", 0),
            "Pharmacist / MA": stats.get("by_route_to", {}).get("pharmacist_or_ma", 0),
        }
        df_route = pd.DataFrame(
            list(route_data.items()), columns=["Queue Role", "Count"]
        ).set_index("Queue Role")
        st.bar_chart(df_route, color="#0284c7")

    with chart_col2:
        st.subheader("Counts by Urgency")
        urgency_data = {
            "Routine": stats.get("by_urgency", {}).get("routine", 0),
            "Soon": stats.get("by_urgency", {}).get("soon", 0),
            "Urgent": stats.get("by_urgency", {}).get("urgent", 0),
            "Emergency": stats.get("by_urgency", {}).get("emergency", 0),
        }
        df_urgency = pd.DataFrame(
            list(urgency_data.items()), columns=["Urgency Level", "Count"]
        ).set_index("Urgency Level")
        st.bar_chart(df_urgency, color="#ea580c")

    st.subheader("Counts by Review Status")
    status_data = {
        "Pending": stats.get("by_status", {}).get("pending", 0),
        "Approved": stats.get("by_status", {}).get("approved", 0),
        "Rejected": stats.get("by_status", {}).get("rejected", 0),
    }
    df_status = pd.DataFrame(
        list(status_data.items()), columns=["Status", "Count"]
    ).set_index("Status")
    st.bar_chart(df_status, color="#16a34a")
