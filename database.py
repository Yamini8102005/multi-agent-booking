"""Production-ready SQLite persistence layer for appointment bookings.

This module provides a lightweight SQLAlchemy-based data access layer for storing
bookings in a local SQLite database. It creates the bookings table automatically,
prevents duplicate bookings for the same date and time, and exposes helper
functions for checking availability and reserving slots.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from sqlalchemy import Integer, String, UniqueConstraint, create_engine
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "bookings.db"
DB_URL = f"sqlite:///{DB_PATH}"

DEFAULT_SLOTS = ("09:00", "10:00", "11:00", "13:00", "14:00", "15:00", "16:00")
EMAIL_REGEX = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")


class Base(DeclarativeBase):
    """Base class for ORM models."""


class Booking(Base):
    """Represents a single booking record."""

    __tablename__ = "bookings"
    __table_args__ = (UniqueConstraint("date", "time", name="uq_booking_date_time"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    time: Mapped[str] = mapped_column(String(5), nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(255), nullable=False, index=True)

    def __repr__(self) -> str:
        return (
            f"Booking(id={self.id!r}, date={self.date!r}, time={self.time!r}, "
            f"email={self.email!r})"
        )


engine = create_engine(DB_URL, future=True, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)


def create_database() -> None:
    """Create the SQLite schema if it does not exist."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database schema created successfully at %s", DB_PATH)
    except SQLAlchemyError as exc:
        logger.exception("Failed to initialize database schema")
        raise RuntimeError("Unable to initialize database schema") from exc


def _normalize_date(value: str) -> str:
    """Normalize the incoming date string to YYYY-MM-DD if possible."""
    if not value or not str(value).strip():
        raise ValueError("Date is required")

    raw_value = str(value).strip()
    for date_format in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            parsed = datetime.strptime(raw_value, date_format)
            return parsed.strftime("%Y-%m-%d")
        except ValueError:
            continue

    raise ValueError("Date must be in YYYY-MM-DD format")


def _normalize_time(value: str) -> str:
    """Normalize the incoming time string to HH:MM format."""
    if not value or not str(value).strip():
        raise ValueError("Time is required")

    raw_value = str(value).strip().lower()
    raw_value = re.sub(r"\s+", "", raw_value)

    if re.fullmatch(r"\d{1,2}:\d{2}", raw_value):
        hour_str, minute_str = raw_value.split(":", 1)
        hour = int(hour_str)
        minute = int(minute_str)
        if hour > 23 or minute > 59:
            raise ValueError("Time must be in HH:MM format")
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
            raise ValueError("Time must be in HH:MM format")
        return f"{hour:02d}:{minute:02d}"

    if re.fullmatch(r"\d{1,2}(am|pm)", raw_value):
        hour = int(raw_value[:-2])
        suffix = raw_value[-2:]
        if suffix == "pm" and hour < 12:
            hour += 12
        elif suffix == "am" and hour == 12:
            hour = 0
        if hour > 23:
            raise ValueError("Time must be in HH:MM format")
        return f"{hour:02d}:00"

    if re.fullmatch(r"\d{1,4}", raw_value):
        if len(raw_value) == 1 or len(raw_value) == 2:
            hour = int(raw_value)
            if hour > 23:
                raise ValueError("Time must be in HH:MM format")
            return f"{hour:02d}:00"
        if len(raw_value) == 3:
            hour = int(raw_value[:-2])
            minute = int(raw_value[-2:])
            if hour > 23 or minute > 59:
                raise ValueError("Time must be in HH:MM format")
            return f"{hour:02d}:{minute:02d}"
        if len(raw_value) == 4:
            hour = int(raw_value[:-2])
            minute = int(raw_value[-2:])
            if hour > 23 or minute > 59:
                raise ValueError("Time must be in HH:MM format")
            return f"{hour:02d}:{minute:02d}"

    try:
        parsed = datetime.strptime(raw_value, "%H:%M")
        return parsed.strftime("%H:%M")
    except ValueError as exc:
        raise ValueError("Time must be in HH:MM format") from exc


def _normalize_email(value: str) -> str:
    """Normalize the email address to a lowercase trimmed string."""
    cleaned = str(value).strip().lower()
    if not cleaned:
        raise ValueError("Email is required")
    if not EMAIL_REGEX.fullmatch(cleaned):
        raise ValueError("Email must be a valid email address")
    return cleaned


def check_availability(date: str) -> Dict[str, str]:
    """Return a slot-status mapping with human-readable availability values."""
    try:
        normalized_date = _normalize_date(date)
    except ValueError as exc:
        logger.warning("Invalid date provided for availability check: %s", exc)
        raise

    try:
        available_slots = set(get_available_slots(normalized_date))
        return {slot: "available" if slot in available_slots else "booked" for slot in DEFAULT_SLOTS}
    except SQLAlchemyError as exc:
        logger.exception("Availability check failed for date %s", normalized_date)
        raise RuntimeError("Unable to check availability") from exc


def get_available_slots(date: str) -> List[str]:
    """Return a list of available appointment slots for the provided date."""
    try:
        normalized_date = _normalize_date(date)
    except ValueError as exc:
        logger.warning("Invalid date provided for slot lookup: %s", exc)
        raise

    try:
        with SessionLocal() as session:
            booked_times = {
                booking.time
                for booking in session.query(Booking).filter(Booking.date == normalized_date).all()
            }
    except SQLAlchemyError as exc:
        logger.exception("Failed to read bookings for date %s", normalized_date)
        raise RuntimeError("Unable to fetch available slots") from exc

    return [slot for slot in DEFAULT_SLOTS if slot not in booked_times]


def reserve_slot(date: str, time: str, email: str) -> Booking:
    """Reserve a slot if it is not already taken."""
    try:
        normalized_date = _normalize_date(date)
        normalized_time = _normalize_time(time)
        normalized_email = _normalize_email(email)
    except ValueError as exc:
        logger.warning("Invalid booking data: %s", exc)
        raise

    try:
        with SessionLocal() as session:
            existing_booking = (
                session.query(Booking)
                .filter(Booking.date == normalized_date, Booking.time == normalized_time)
                .first()
            )
            if existing_booking is not None:
                raise ValueError(f"Slot {normalized_date} at {normalized_time} is already booked")

            booking = Booking(date=normalized_date, time=normalized_time, email=normalized_email)
            session.add(booking)
            session.commit()
            session.refresh(booking)
            logger.info("Reserved slot %s %s for %s", normalized_date, normalized_time, normalized_email)
            return booking
    except IntegrityError as exc:
        logger.warning("Duplicate reservation attempted for %s at %s", normalized_date, normalized_time)
        raise ValueError(f"Slot {normalized_date} at {normalized_time} is already booked") from exc
    except SQLAlchemyError as exc:
        logger.exception("Failed to reserve slot for %s at %s", normalized_date, normalized_time)
        raise RuntimeError("Unable to reserve slot") from exc


create_database()
