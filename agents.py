"""Production-ready agent implementations for the booking orchestration workflow.

This module defines the state structure used by the LangGraph workflow, the Triage
Agent node, the Booking Specialist node, and the tool execution node that interacts
with the existing database and notification helpers.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from dateutil import parser as dateutil_parser
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

try:  # pragma: no cover - fallback to local prompts when prompts.py is not populated
    from prompts import BOOKING_PROMPT, TRIAGE_PROMPT
except ImportError:  # pragma: no cover - prompts.py exists but may be empty
    TRIAGE_PROMPT = (
        "You are a Triage Agent for a booking assistant. Classify the user's intent as "
        "either 'booking' or 'general'. Return a compact JSON object with keys 'intent', "
        "'route_to', and 'response'."
    )
    BOOKING_PROMPT = (
        "You are a Booking Specialist for an appointment system. Collect missing details "
        "for date, time, and email. Return a compact JSON object with keys 'status', "
        "'message', 'date', 'time', 'email', and 'needs_more_info'."
    )

from tools import check_availability as tool_check_availability
from tools import reserve_slot as tool_reserve_slot
from tools import send_booking_notification as tool_send_booking_notification

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def merge_booking_context(left: Optional[Dict[str, Any]], right: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge two booking contexts. If right is None or contains a clear signal, reset."""
    if left is None:
        left = {}
    if right is None:
        return left
    if right.get("_clear"):
        new_dict = {k: v for k, v in right.items() if k != "_clear"}
        return new_dict
    merged = dict(left)
    for k, v in right.items():
        if v is not None and str(v).strip() != "":
            merged[k] = v
    return merged


class AgentState(TypedDict, total=False):
    """State carried through the LangGraph workflow."""

    messages: Annotated[List[BaseMessage], add_messages]
    intent: str
    route_to: str
    booking_context: Annotated[Dict[str, Any], merge_booking_context]
    last_response: str
    thread_id: str


class TriageDecision(BaseModel):
    """Structured output for the triage agent."""

    intent: Literal["general", "booking"]
    route_to: Literal["end", "booking_specialist"]
    response: str = Field(default="How can I help you today?")


class BookingDecision(BaseModel):
    """Structured output for the booking specialist."""

    status: Literal["needs_info", "ready_to_book", "negotiation_required", "completed", "failed"]
    message: str
    date: Optional[str] = None
    time: Optional[str] = None
    email: Optional[str] = None
    needs_more_info: bool = False
    available_slots: List[str] = Field(default_factory=list)


def _create_llm() -> Optional[Any]:
    """Create a Gemini-backed chat model when the API key is configured."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.warning("GOOGLE_API_KEY is not configured; routing will fall back to deterministic heuristics")
        return None

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_key,
            temperature=0.2,
            max_retries=1,
        )
    except Exception as exc:  # pragma: no cover - import/env dependent
        logger.exception("Failed to initialize Gemini model")
        return None


def _extract_latest_user_message(messages: Optional[List[BaseMessage]]) -> str:
    """Return the most recent human message from the workflow state."""
    if not messages:
        return ""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
        if isinstance(message, dict):
            if message.get("type") == "human" or message.get("role") == "user":
                return str(message.get("content", ""))
        elif hasattr(message, "type") and getattr(message, "type") == "human":
            return str(getattr(message, "content", ""))
    return ""


def _extract_booking_fields(text: str) -> Dict[str, str]:
    """Extract booking-related fields from user text using regex heuristics."""
    extracted: Dict[str, str] = {}
    lower_text = text.lower()

    # 1. Date extraction
    match = re.search(r"\b\d{4}[-/]\d{2}[-/]\d{2}\b", text)
    if match:
        extracted["date"] = match.group(0)
    else:
        found_keyword = False
        for keyword in (
            "today",
            "tomorrow",
            "next monday",
            "next tuesday",
            "next wednesday",
            "next thursday",
            "next friday",
            "next saturday",
            "next sunday",
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        ):
            if re.search(rf"\b{keyword}\b", lower_text):
                extracted["date"] = keyword
                found_keyword = True
                break
        if not found_keyword:
            on_match = re.search(r"\b(?:on|for|date)\s+([A-Za-z0-9_\-/]+)\b", text, re.IGNORECASE)
            if on_match:
                candidate = on_match.group(1).strip()
                if candidate.lower() not in ("the", "a", "an", "at", "my"):
                    extracted["date"] = candidate

    # 2. Email extraction
    email_match = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text)
    if email_match:
        extracted["email"] = email_match.group(0)

    # 3. Time extraction (preprocess text to avoid conflicts with dates and emails)
    temp_text = text
    if match:
        temp_text = temp_text.replace(match.group(0), "")
    for keyword in (
        "today", "tomorrow", "next monday", "next tuesday", "next wednesday",
        "next thursday", "next friday", "next saturday", "next sunday",
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"
    ):
        temp_text = re.sub(rf"\b{keyword}\b", "", temp_text, flags=re.IGNORECASE)
    if email_match:
        temp_text = temp_text.replace(email_match.group(0), "")

    # Match time with spaces e.g. "5 PM" or "10:30 am"
    time_match = re.search(r"\b(\d{1,2}(?::\d{2})?\s*(?:am|pm))\b", temp_text, re.IGNORECASE)
    if not time_match:
        # Fallback to hh:mm
        time_match = re.search(r"\b(\d{1,2}:\d{2})\b", temp_text)
    if not time_match:
        # Fallback to hh prefixed by at
        time_match = re.search(r"\b(?:at|around|slot)\s+(\d{1,2})\b", temp_text, re.IGNORECASE)

    if time_match:
        extracted["time"] = time_match.group(1)

    return extracted


def _normalize_relative_date(value: str, reference_date: Optional[datetime] = None) -> str:
    """Normalize common relative date phrases such as today, tomorrow, weekdays, and 'next <weekday>'."""
    if not value:
        raise ValueError("Date is required")

    text = str(value).strip().lower()
    base_date = (reference_date or datetime.now()).date()
    normalized_str = None
    
    # 1. Handle today
    if text == "today":
        normalized_str = base_date.strftime("%Y-%m-%d")
        
    # 2. Handle tomorrow
    elif text == "tomorrow":
        normalized_str = (base_date + timedelta(days=1)).strftime("%Y-%m-%d")

    else:
        weekday_map = {
            "monday": 0,
            "tuesday": 1,
            "wednesday": 2,
            "thursday": 3,
            "friday": 4,
            "saturday": 5,
            "sunday": 6,
        }

        # 3. Handle next <weekday>
        if text.startswith("next "):
            weekday_name = text[5:].strip()
            if weekday_name in weekday_map:
                current_weekday = base_date.weekday()
                target_weekday = weekday_map[weekday_name]
                delta = (target_weekday - current_weekday) % 7
                # "next <weekday>" implies next week's weekday
                normalized_str = (base_date + timedelta(days=delta + 7)).strftime("%Y-%m-%d")

        # 4. Handle plain weekday name (e.g. "Friday")
        elif text in weekday_map:
            current_weekday = base_date.weekday()
            target_weekday = weekday_map[text]
            delta = (target_weekday - current_weekday) % 7
            normalized_str = (base_date + timedelta(days=delta)).strftime("%Y-%m-%d")

    # 5. Fallback to parsing absolute dates
    if not normalized_str:
        try:
            parsed = dateutil_parser.parse(text, fuzzy=True)
            normalized_str = parsed.date().strftime("%Y-%m-%d")
        except (TypeError, ValueError) as exc:
            raise ValueError("I could not understand the requested date") from exc

    # Validate that date is not in the past
    parsed_date = datetime.strptime(normalized_str, "%Y-%m-%d").date()
    if parsed_date < base_date:
        raise ValueError("Date cannot be in the past")
        
    return normalized_str


def _normalize_time(value: str) -> str:
    """Normalize common time values such as 9am, 9:00, 9 AM, or raw hour numbers to HH:MM."""
    if not value:
        raise ValueError("Time is required")

    raw_value = str(value).strip().lower().replace(" ", "")
    
    if re.fullmatch(r"\d{1,2}:\d{2}", raw_value):
        hour_str, minute_str = raw_value.split(":", 1)
        hour = int(hour_str)
        minute = int(minute_str)
        if hour > 23 or minute > 59:
            raise ValueError("Time must be between 00:00 and 23:59")
        return f"{hour:02d}:{minute:02d}"

    if re.fullmatch(r"\d{1,2}:\d{2}(am|pm)", raw_value):
        hour_str, minute_str = raw_value[:-2].split(":", 1)
        hour = int(hour_str)
        minute = int(minute_str)
        suffix = raw_value[-2:]
        if suffix == "pm" and hour < 12:
            hour += 12
        elif suffix == "am" and hour == 12:
            hour = 0
        if hour > 23 or minute > 59:
            raise ValueError("Time must be between 00:00 and 23:59")
        return f"{hour:02d}:{minute:02d}"

    if re.fullmatch(r"\d{1,2}(am|pm)", raw_value):
        hour = int(raw_value[:-2])
        suffix = raw_value[-2:]
        if suffix == "pm" and hour < 12:
            hour += 12
        elif suffix == "am" and hour == 12:
            hour = 0
        if hour > 23:
            raise ValueError("Time must be between 00:00 and 23:59")
        return f"{hour:02d}:00"

    if re.fullmatch(r"\d{1,4}", raw_value):
        if len(raw_value) in (1, 2):
            hour = int(raw_value)
            if hour > 23:
                raise ValueError("Time must be between 00:00 and 23:59")
            return f"{hour:02d}:00"
        if len(raw_value) in (3, 4):
            hour = int(raw_value[:-2])
            minute = int(raw_value[-2:])
            if hour > 23 or minute > 59:
                raise ValueError("Time must be between 00:00 and 23:59")
            return f"{hour:02d}:{minute:02d}"

    raise ValueError("Time must be in HH:MM format")


def _normalize_email(value: str) -> str:
    """Validate and normalize the email address."""
    if not value:
        raise ValueError("Email is required")
    cleaned = str(value).strip().lower()
    if not re.fullmatch(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$", cleaned):
        raise ValueError("Email must be a valid email address")
    return cleaned


def clean_json_text(text: str) -> str:
    """Clean markdown code blocks and sanitize text for JSON parsing."""
    text = text.strip()
    if "```" in text:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if match:
            text = match.group(1).strip()
        else:
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        text = json_match.group(0)
    return text.strip()


def triage_agent(state: AgentState) -> AgentState:
    """Classify the user's intent and decide whether to hand off to the booking specialist."""
    messages = state.get("messages") or []
    latest_message = _extract_latest_user_message(messages)
    llm = _create_llm()

    context: Dict[str, Any] = state.get("booking_context") or {}
    is_active_booking = bool(context) and not context.get("completed")

    decision = TriageDecision(intent="general", route_to="end", response="How can I help you today?")
    
    normalized = latest_message.lower()
    
    booking_keywords = {
        "book",
        "booking",
        "appointment",
        "schedule",
        "slot",
        "slots",
        "reserve",
        "reservation",
        "meeting",
        "availability",
        "available",
    }
    has_booking_word = any(re.search(rf"\b{kw}\b", normalized) for kw in booking_keywords)

    temporal_keywords = {
        "tomorrow",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    }
    has_temporal = any(re.search(rf"\b{kw}\b", normalized) for kw in temporal_keywords)
    
    has_today = re.search(r"\btoday\b", normalized)
    is_greeting = re.search(r"\b(hello|hi|hey|how\s+are\s+you|how\s+is\s+your\s+day|good\s+morning|good\s+afternoon|good\s+evening)\b", normalized)
    
    if has_today and is_greeting:
        has_today = False
        
    is_booking_heuristic = is_active_booking or has_booking_word or (has_temporal and not is_greeting) or (has_today and not is_greeting)

    if is_booking_heuristic:
        decision = TriageDecision(
            intent="booking",
            route_to="booking_specialist",
            response="I can help you book an appointment. I’ll collect the required details and check availability." if not is_active_booking else "Let's continue your booking.",
        )

    if llm is not None:
        try:
            response = llm.invoke(
                [
                    SystemMessage(content=TRIAGE_PROMPT),
                    HumanMessage(content=latest_message),
                ]
            )
            payload = clean_json_text(str(response.content))
            try:
                parsed = json.loads(payload)
                decision = TriageDecision(**parsed)
            except Exception as exc:
                logger.warning("Unable to parse triage JSON: %s. Payload was: %s", exc, payload)
        except Exception as exc:
            logger.warning("Falling back to heuristic triage: %s", exc)

    if is_active_booking:
        decision.intent = "booking"
        decision.route_to = "booking_specialist"

    return {
        "intent": decision.intent,
        "route_to": decision.route_to,
        "last_response": decision.response,
        "messages": [AIMessage(content=decision.response)] if not is_active_booking else [],
    }


def booking_specialist(state: AgentState) -> AgentState:
    """Collect booking details, normalize dates, and prepare tool execution."""
    messages = state.get("messages") or []
    latest_message = _extract_latest_user_message(messages)
    context: Dict[str, Any] = dict(state.get("booking_context") or {})

    if context.get("completed"):
        context = {"_clear": True, "is_active_booking": True}
    else:
        # Ensure booking is tracked as active
        context["is_active_booking"] = True

    # 1. Extraction phase
    extracted: Dict[str, str] = {}
    llm = _create_llm()
    if llm is not None:
        collected_info = []
        if context.get("date"):
            collected_info.append(f"Date: {context['date']}")
        if context.get("time"):
            collected_info.append(f"Time: {context['time']}")
        if context.get("email"):
            collected_info.append(f"Email: {context['email']}")
        
        info_str = "\n".join(collected_info) if collected_info else "None"
        
        extraction_prompt = (
            "You are an information extraction assistant for an appointment booking system.\n"
            f"Currently collected details:\n{info_str}\n\n"
            "Analyze the conversation history and the latest user message. Extract any newly provided "
            "or updated values for 'date', 'time', and 'email'.\n"
            "Return a JSON object with the keys 'date', 'time', 'email'. Use null for fields "
            "that are not mentioned or cannot be extracted. Do not overwrite already collected fields "
            "with null unless the user is explicitly correcting or changing them.\n"
            "For relative dates like 'tomorrow', 'next Monday', or weekdays like 'Friday', extract the "
            "literal phrase (e.g. 'tomorrow', 'Friday') as-is. Do not try to convert them into YYYY-MM-DD "
            "format yourself.\n"
            "Respond ONLY with the JSON object."
        )
        try:
            llm_messages = [SystemMessage(content=extraction_prompt)]
            # Add up to 3 context messages
            for msg in messages[-3:]:
                llm_messages.append(msg)
            # Ensure latest message is present
            if not any(msg.content == latest_message for msg in messages[-3:]):
                llm_messages.append(HumanMessage(content=latest_message))
                
            response = llm.invoke(llm_messages)
            payload = clean_json_text(str(response.content))
            try:
                extracted = json.loads(payload)
            except Exception as exc:
                logger.warning("Failed to parse specialist extraction JSON: %s. Payload was: %s", exc, payload)
                extracted = {}
        except Exception as exc:
            logger.warning("LLM extraction failed, falling back to heuristics: %s", exc)
            extracted = {}

    # Heuristic fallback/merge
    heuristic_extracted = _extract_booking_fields(latest_message)
    for field in ("date", "time", "email"):
        val = extracted.get(field) or heuristic_extracted.get(field)
        if val:
            extracted[field] = val

    # 2. Update context & validation phase
    if extracted.get("date"):
        try:
            normalized_date = _normalize_relative_date(extracted["date"])
            context["date"] = normalized_date
            context["date_source"] = extracted["date"]
            context.pop("date_error", None)
        except ValueError as exc:
            context["date_error"] = str(exc)
            context.pop("date", None)
            context.pop("date_source", None)

    if extracted.get("time"):
        try:
            context["time"] = _normalize_time(extracted["time"])
            context.pop("time_error", None)
            if context.get("pending_tool") == "negotiate":
                context.pop("pending_tool", None)
                context.pop("tool_args", None)
        except ValueError as exc:
            context["time_error"] = str(exc)
            context.pop("time", None)

    if extracted.get("email"):
        try:
            context["email"] = _normalize_email(extracted["email"])
            context.pop("email_error", None)
        except ValueError as exc:
            context["email_error"] = str(exc)
            context.pop("email", None)

    # Check for missing/invalid details
    missing_fields: List[str] = []
    if not context.get("date"):
        missing_fields.append("date")
    if not context.get("time"):
        missing_fields.append("time")
    if not context.get("email"):
        missing_fields.append("email")

    error_messages: List[str] = []
    if context.get("date_error"):
        error_messages.append(f"Date error: {context['date_error']}.")
    if context.get("time_error"):
        error_messages.append(f"Time error: {context['time_error']}.")
    if context.get("email_error"):
        error_messages.append(f"Email error: {context['email_error']}.")

    # If there are parsing/validation errors, report them immediately
    if error_messages:
        err_prompt = " ".join(error_messages)
        if missing_fields:
            prompt = f"{err_prompt} Please also provide the missing fields: {', '.join(missing_fields)}."
        else:
            prompt = f"{err_prompt} Please correct these details."
        decision = BookingDecision(status="needs_info", message=prompt, needs_more_info=True)
        return {
            "booking_context": context,
            "last_response": decision.message,
            "messages": [AIMessage(content=decision.message)],
        }

    # If date and time are present and valid, check if the slot is available
    if context.get("date") and context.get("time"):
        try:
            availability = tool_check_availability(context["date"])
            requested_slot = context["time"]
            availability_status = availability.get(requested_slot, "booked")
            
            if availability_status != "available":
                available_slots = [slot for slot, status in availability.items() if status == "available"]
                context.pop("pending_tool", None)
                context.pop("tool_args", None)
                
                if available_slots:
                    msg = (
                        f"The requested slot is unavailable. Please choose one of the available alternatives: "
                        f"{', '.join(available_slots)}"
                    )
                else:
                    msg = "No appointment slots are available on that date."
                
                decision = BookingDecision(
                    status="negotiation_required",
                    message=msg,
                    date=context["date"],
                    time=requested_slot,
                    email=context.get("email"),
                    available_slots=available_slots,
                )
                return {
                    "booking_context": context,
                    "last_response": decision.message,
                    "messages": [AIMessage(content=decision.message)],
                }
        except Exception as exc:
            logger.exception("Availability lookup failed")
            decision = BookingDecision(status="failed", message=f"I could not check availability right now: {exc}")
            return {
                "booking_context": context,
                "last_response": decision.message,
                "messages": [AIMessage(content=decision.message)],
            }

    # If the slot is available but we are missing some other information (e.g. email), ask for it specifically
    if missing_fields:
        if len(missing_fields) == 1:
            field = missing_fields[0]
            if field == "email":
                prompt = "Please provide your email address."
            elif field == "time":
                prompt = "Please provide the preferred booking time."
            elif field == "date":
                prompt = "Please provide the preferred booking date."
            else:
                prompt = f"Please provide your {field}."
        else:
            parts = []
            if "date" in missing_fields:
                parts.append("preferred booking date")
            if "time" in missing_fields:
                parts.append("preferred booking time")
            if "email" in missing_fields:
                parts.append("email address")
            
            if len(parts) == 2:
                prompt = f"Please provide your {parts[0]} and {parts[1]}."
            else:
                prompt = f"Please provide your {', '.join(parts[:-1])}, and {parts[-1]}."

        decision = BookingDecision(status="needs_info", message=prompt, needs_more_info=True)
        return {
            "booking_context": context,
            "last_response": decision.message,
            "messages": [AIMessage(content=decision.message)],
        }

    # All fields are valid and slot is available
    requested_slot = context["time"]
    context["pending_tool"] = "reserve_slot"
    context["tool_args"] = {
        "date": context["date"],
        "time": requested_slot,
        "email": context["email"],
    }
    decision = BookingDecision(
        status="ready_to_book",
        message="I’m ready to reserve your requested slot.",
        date=context["date"],
        time=requested_slot,
        email=context["email"],
    )
    return {
        "booking_context": context,
        "last_response": decision.message,
        "messages": [AIMessage(content=decision.message)],
    }


def tool_execution(state: AgentState) -> AgentState:
    """Execute the relevant tool call for the booking workflow with error boundaries."""
    context: Dict[str, Any] = dict(state.get("booking_context") or {})
    action = context.get("pending_tool")
    response_text = state.get("last_response") or "I’m processing the booking request."

    if action == "reserve_slot":
        tool_args = context.get("tool_args") or {}
        
        try:
            reservation_result = tool_reserve_slot(
                date=tool_args.get("date"),
                time=tool_args.get("time"),
                email=tool_args.get("email"),
            )
        except Exception as exc:
            logger.exception("Database reservation failed during tool execution")
            reservation_result = {"success": False, "message": f"Reservation failed: {exc}"}

        if reservation_result.get("success"):
            response_text = reservation_result.get("message", "Booking completed.")
            
            try:
                notification_result = tool_send_booking_notification(
                    email=tool_args.get("email"),
                    details={
                        "date": tool_args.get("date"),
                        "time": tool_args.get("time"),
                        "status": "confirmed",
                    },
                )
                if notification_result.get("status") == "warning":
                    response_text = f"{response_text} Warning: {notification_result.get('message')}"
            except Exception as exc:
                logger.exception("Notification webhook failed during tool execution")
                response_text = f"{response_text} Warning: Notification could not be delivered: {exc}"
            
            # Clear context but set completed flag
            context = {"completed": True, "_clear": True}
        else:
            # Reservation failed (e.g. slot already taken)
            error_message = reservation_result.get("message", "Booking could not be completed.")
            context.pop("pending_tool", None)
            context.pop("tool_args", None)
            
            # Let's check available alternatives
            date_val = tool_args.get("date")
            alternatives = []
            if date_val:
                try:
                    availability = tool_check_availability(date_val)
                    alternatives = [slot for slot, status in availability.items() if status == "available"]
                except Exception as exc:
                    logger.warning("Failed to check alternatives: %s", exc)
            
            if alternatives:
                response_text = (
                    f"Reservation failed: {error_message} "
                    f"Please choose one of the available alternatives: {', '.join(alternatives)}"
                )
            else:
                response_text = f"Reservation failed: {error_message} No other slots are available on this date."
    else:
        response_text = state.get("last_response") or "No action was needed."

    updates = {
        "booking_context": context,
        "last_response": response_text,
    }
    
    if action == "reserve_slot":
        updates["messages"] = [AIMessage(content=response_text)]
        
    return updates
