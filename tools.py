"""Production-ready tool functions for appointment availability, reservation, and notifications.

These helpers wrap the persistence layer and outbound webhook integration used by
agents in the booking workflow.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv

from database import check_availability as db_check_availability
from database import get_available_slots as db_get_available_slots
from database import reserve_slot as db_reserve_slot

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

load_dotenv()


def check_availability(date: str) -> Dict[str, str]:
    """Return slot availability status for the provided date."""
    try:
        return db_check_availability(date)
    except ValueError as exc:
        logger.warning("Invalid date supplied to availability tool: %s", exc)
        return {"error": str(exc)}
    except RuntimeError as exc:
        logger.exception("Availability lookup failed")
        return {"error": str(exc)}


def reserve_slot(date: str, time: str, email: str) -> Dict[str, Any]:
    """Reserve a booking slot and return a structured success or failure message."""
    try:
        booking = db_reserve_slot(date, time, email)
        return {
            "success": True,
            "message": (
                f"Booking reserved successfully for {booking.date} at {booking.time} "
                f"for {booking.email}."
            ),
            "date": booking.date,
            "time": booking.time,
            "email": booking.email,
        }
    except ValueError as exc:
        logger.warning("Reservation rejected: %s", exc)
        msg = str(exc)
        if "already booked" in msg.lower():
            msg = "Requested slot is already booked."
        return {"success": False, "message": msg}
    except RuntimeError as exc:
        logger.exception("Reservation failed")
        return {"success": False, "message": str(exc)}


def send_booking_notification(email: str, details: Dict[str, Any]) -> Dict[str, Any]:
    """Send a booking confirmation notification to the configured webhook."""
    webhook_url = os.getenv("WEBHOOK_URL")
    if not webhook_url:
        logger.warning("WEBHOOK_URL is not configured; skipping notification")
        return {
            "success": False,
            "message": "Webhook URL is not configured; notification was skipped.",
            "status": "warning",
        }

    payload = {
        "email": email,
        "date": details.get("date"),
        "time": details.get("time"),
        "status": details.get("status", "confirmed"),
    }

    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
        logger.info("Booking notification sent successfully to %s", webhook_url)
        return {
            "success": True,
            "message": "Notification delivered successfully.",
            "status": "sent",
        }
    except requests.Timeout:
        logger.exception("Webhook request timed out")
        return {
            "success": False,
            "message": "Notification could not be delivered because the request timed out.",
            "status": "warning",
        }
    except requests.RequestException as exc:
        logger.exception("Webhook request failed")
        return {
            "success": False,
            "message": f"Notification could not be delivered: {exc}",
            "status": "warning",
        }
