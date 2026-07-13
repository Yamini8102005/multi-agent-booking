"""Production-ready LangGraph workflow for the multi-agent booking assistant.

This module compiles the triage and booking specialist nodes into a StateGraph,
adds conditional routing, and uses SQLite-backed checkpointing for persistent
thread-scoped conversation memory.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Literal, TypedDict, Annotated

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agents import AgentState, booking_specialist, tool_execution, triage_agent, merge_booking_context

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
MEMORY_DB_PATH = DATA_DIR / "memory.db"


class GraphState(TypedDict, total=False):
    """Graph state alias used by the workflow runtime."""

    messages: Annotated[list[Any], add_messages]
    intent: str
    route_to: str
    booking_context: Annotated[dict[str, Any], merge_booking_context]
    last_response: str
    thread_id: str


def route_after_triage(state: AgentState) -> Literal["booking_specialist", "__end__"]:
    """Route to the booking specialist only when triage selected a booking request."""
    if state.get("route_to") == "booking_specialist":
        return "booking_specialist"
    return "__end__"


def route_after_specialist(state: AgentState) -> Literal["tool_execution", "__end__"]:
    """Route to tool execution when the specialist prepared a pending action."""
    context = state.get("booking_context") or {}
    if context.get("pending_tool"):
        return "tool_execution"
    return "__end__"


def _create_checkpointer() -> SqliteSaver:
    """Create a SQLite-backed checkpointer compatible with the installed LangGraph release."""
    db_path = str(MEMORY_DB_PATH)
    try:
        # Avoid WAL mode to prevent shared memory/mmap limitations on Render's container overlayfs.
        # Add timeout to handle concurrent accesses gracefully.
        connection = sqlite3.connect(db_path, check_same_thread=False, timeout=30.0)
        return SqliteSaver(connection)
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.exception("Failed to initialize SQLite checkpoint saver")
        raise RuntimeError("Unable to initialize conversation memory store") from exc


def build_graph() -> Any:
    """Build and compile the workflow graph with persistent SQLite checkpoints."""
    saver = _create_checkpointer()
    workflow = StateGraph(AgentState)
    workflow.add_node("triage_agent", triage_agent)
    workflow.add_node("booking_specialist", booking_specialist)
    workflow.add_node("tool_execution", tool_execution)

    workflow.add_edge(START, "triage_agent")
    workflow.add_conditional_edges(
        "triage_agent",
        route_after_triage,
        {
            "booking_specialist": "booking_specialist",
            "__end__": END,
        },
    )
    workflow.add_conditional_edges(
        "booking_specialist",
        route_after_specialist,
        {
            "tool_execution": "tool_execution",
            "__end__": END,
        },
    )
    workflow.add_edge("tool_execution", END)

    return workflow.compile(checkpointer=saver)


graph = build_graph()
