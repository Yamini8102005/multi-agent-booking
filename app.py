"""Production-ready Streamlit UI for the multi-agent appointment booking system.

The application renders a ChatGPT-style interface, initializes the LangGraph
workflow, persists conversation state per thread, and displays the full chat history
with a sidebar for thread management and configuration.
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from graph import graph

load_dotenv()

st.set_page_config(page_title="Multi-Agent Booking", page_icon="📅", layout="wide")


def _initialize_session_state() -> None:
    """Initialize persistent Streamlit session state for the chat UI."""
    if "is_new_conversation" not in st.session_state:
        st.session_state.is_new_conversation = False

    if "thread_id" in st.query_params:
        st.session_state.thread_id = st.query_params["thread_id"]
    elif "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
        st.query_params["thread_id"] = st.session_state.thread_id

    if "messages" not in st.session_state or st.session_state.is_new_conversation:
        st.session_state.is_new_conversation = False
        # Sync from LangGraph SqliteSaver checkpointer
        try:
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            graph_state = graph.get_state(config)
            graph_messages = graph_state.values.get("messages", [])
            st_messages = []
            for msg in graph_messages:
                content = ""
                msg_type = ""
                
                # Extract content
                if hasattr(msg, "content"):
                    content = msg.content
                elif isinstance(msg, dict):
                    content = msg.get("content", "")
                elif isinstance(msg, tuple) and len(msg) == 2:
                    content = msg[1]

                # Extract msg_type
                if hasattr(msg, "type"):
                    msg_type = msg.type
                elif isinstance(msg, dict):
                    msg_type = msg.get("type") or msg.get("role")
                elif isinstance(msg, tuple) and len(msg) == 2:
                    msg_type = msg[0]

                msg_type = str(msg_type).lower() if msg_type else ""

                if msg_type in ("human", "user"):
                    st_messages.append({"role": "user", "content": content})
                elif msg_type in ("ai", "assistant", "model", "system"):
                    st_messages.append({"role": "assistant", "content": content})
            st.session_state.messages = st_messages
        except Exception:
            st.session_state.messages = []


def _render_sidebar() -> None:
    """Render the application sidebar with configuration and chat controls."""
    with st.sidebar:
        st.title("Booking Assistant")
        st.caption("Multi-agent appointment booking experience")

        # Dynamic Google API Key input
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
        if "google_api_key" not in st.session_state:
            st.session_state.google_api_key = api_key
            
        user_key = st.text_input("Google API Key", value=st.session_state.google_api_key, type="password")
        if user_key != st.session_state.google_api_key:
            st.session_state.google_api_key = user_key
            os.environ["GOOGLE_API_KEY"] = user_key
            os.environ["GEMINI_API_KEY"] = user_key

        if st.button("New Conversation", use_container_width=True):
            new_id = str(uuid.uuid4())
            st.session_state.thread_id = new_id
            st.query_params["thread_id"] = new_id
            st.session_state.messages = []
            st.session_state.is_new_conversation = True
            st.rerun()

        if st.button("Clear Chat", use_container_width=True):
            new_id = str(uuid.uuid4())
            st.session_state.thread_id = new_id
            st.query_params["thread_id"] = new_id
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")
        st.caption(f"Thread ID: {st.session_state.thread_id}")


def _render_chat_history(messages: List[Dict[str, str]]) -> None:
    """Render the user and assistant messages in the Streamlit interface."""
    for message in messages:
        role = message.get("role", "assistant")
        content = message.get("content", "")
        with st.chat_message(role):
            st.markdown(content)


def _run_graph(message: str, thread_id: str) -> str:
    """Invoke the LangGraph workflow for a user message and return the assistant response."""
    state = {
        "messages": [
            {"type": "human", "content": message},
        ],
        "thread_id": thread_id,
    }
    try:
        result = graph.invoke(
            state,
            config={"configurable": {"thread_id": thread_id}},
        )
        return str(result.get("last_response", "I’m sorry, I could not process that request."))
    except Exception as exc:  # pragma: no cover - runtime dependent
        return f"I’m sorry, something went wrong: {exc}"


def main() -> None:
    """Render the main chat interface and process user input."""
    _initialize_session_state()
    _render_sidebar()

    st.title("📅 Appointment Booking Assistant")
    st.caption("Ask for an appointment and I’ll coordinate the booking flow for you.")

    _render_chat_history(st.session_state.messages)

    if prompt := st.chat_input("Ask about appointments or booking details"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Thinking..."):
            assistant_response = _run_graph(prompt, st.session_state.thread_id)

        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        with st.chat_message("assistant"):
            st.markdown(assistant_response)


if __name__ == "__main__":
    main()
