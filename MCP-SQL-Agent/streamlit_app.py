"""Streamlit UI for the MCP SQL Agent."""

import asyncio
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from database.setup_db import setup_database
from agent.agent import MCPAgent

# --- Page config ---
st.set_page_config(
    page_title="MCP SQL Agent",
    page_icon="🗄️",
    layout="wide",
)

# --- Ensure database exists ---
DB_PATH = os.path.join(os.path.dirname(__file__), "database", "company.db")
if not os.path.exists(DB_PATH):
    setup_database()

# --- Custom CSS ---
st.markdown("""
<style>
    .step-plan { border-left: 3px solid #6366f1; padding-left: 12px; margin: 8px 0; }
    .step-action { border-left: 3px solid #f59e0b; padding-left: 12px; margin: 8px 0; }
    .step-observation { border-left: 3px solid #10b981; padding-left: 12px; margin: 8px 0; }
    .step-answer { border-left: 3px solid #3b82f6; padding-left: 12px; margin: 8px 0; }
    .main-answer { padding: 16px; border-radius: 8px; background: #f0f9ff; border: 1px solid #bae6fd; }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("🗄️ MCP SQL Agent")
st.caption("Ask natural language questions about the company database. "
           "The agent discovers schema, generates SQL, and answers — all through MCP tools.")

# --- Sidebar: example questions ---
with st.sidebar:
    st.header("Example Questions")
    examples = [
        "Fetch employee details where department = AI",
        "Which AI-team members have open issues on Project X?",
        "How many open issues does each department have?",
        "Show all projects and their departments",
        "Who has the most open issues?",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex}", use_container_width=True):
            st.session_state["input_question"] = ex

    st.divider()
    st.header("About")
    st.markdown("""
    **How it works:**
    1. Agent calls `list_tables()` via MCP
    2. Agent calls `describe_schema()` for each table
    3. LLM generates SQL from schema + question
    4. Agent executes via `run_query()` through MCP
    5. LLM summarizes results in natural language

    The LLM **never** sees database credentials.
    """)


def _label_color(label: str) -> str:
    """Map step label to a CSS class."""
    return {
        "PLAN": "step-plan",
        "ACTION": "step-action",
        "OBSERVATION": "step-observation",
        "ANSWER": "step-answer",
    }.get(label, "")


def _label_icon(label: str) -> str:
    """Map step label to an emoji."""
    return {
        "PLAN": "📋",
        "ACTION": "⚡",
        "OBSERVATION": "👁️",
        "ANSWER": "✅",
    }.get(label, "📌")


async def run_agent(question: str, followup: str = None) -> tuple[str, list]:
    """Run the agent and return (answer, steps)."""
    agent = MCPAgent()
    await agent.connect()
    try:
        answer = await agent.ask(question, followup=followup)
        return answer, agent.steps
    finally:
        await agent.close()


# --- Main input ---
default_q = st.session_state.get("input_question", "")
question = st.text_input(
    "Ask a question about the company database:",
    value=default_q,
    placeholder="e.g. Which AI employees have open issues?",
    key="question_input",
)

# Clear the sidebar-injected value after using it
if "input_question" in st.session_state:
    del st.session_state["input_question"]

if st.button("Ask Agent", type="primary", disabled=not question):
    with st.spinner("Agent is thinking..."):
        try:
            answer, steps = asyncio.run(run_agent(question))
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    # --- Display answer ---
    if answer.startswith("CLARIFICATION:"):
        clarification_msg = answer.replace("CLARIFICATION:", "").strip()
        st.warning(f"🤔 **The agent needs clarification:** {clarification_msg}")

        followup = st.text_input("Your response:", key="followup_input")
        if st.button("Send follow-up", key="followup_btn"):
            with st.spinner("Agent is processing follow-up..."):
                answer, steps = asyncio.run(run_agent(question, followup=followup))
            st.markdown(f'<div class="main-answer"><strong>Answer:</strong> {answer}</div>',
                        unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="main-answer"><strong>Answer:</strong> {answer}</div>',
                    unsafe_allow_html=True)

    # --- Display steps ---
    st.divider()
    with st.expander("🔍 View agent reasoning steps", expanded=False):
        for step in steps:
            label = step["label"]
            content = step["content"]
            icon = _label_icon(label)
            css_class = _label_color(label)

            st.markdown(f'<div class="{css_class}"><strong>{icon} [{label}]</strong></div>',
                        unsafe_allow_html=True)

            if label == "OBSERVATION" and len(content) > 200:
                # Collapse long observations (schema dumps, query results)
                with st.expander(f"View full {label.lower()}", expanded=False):
                    st.code(content, language="json")
            else:
                st.code(content)
