import os
import shutil
import tempfile
import unittest
from datetime import datetime

# Set up temp data directory for test database isolation BEFORE importing graph/database
TEST_DIR = tempfile.mkdtemp()
os.environ["DATA_DIR"] = TEST_DIR

from database import SessionLocal, Booking, Base, engine
from graph import graph
from agents import _normalize_relative_date, _normalize_time


class TestMultiAgentBooking(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Create schema
        Base.metadata.create_all(bind=engine)

    @classmethod
    def tearDownClass(cls):
        # Release database engine connection pool to prevent Windows file lock issues
        engine.dispose()
        # Release memory.db connections in graph checkpointer
        if hasattr(graph, "checkpointer") and graph.checkpointer:
            if hasattr(graph.checkpointer, "conn") and graph.checkpointer.conn:
                try:
                    graph.checkpointer.conn.close()
                except Exception:
                    pass
        # Cleanup directory
        shutil.rmtree(TEST_DIR)

    def setUp(self):
        # Clear database bookings table before each test
        with SessionLocal() as session:
            session.query(Booking).delete()
            session.commit()

    def test_general_conversation(self):
        """1. General conversation: Message should be handled by Triage Agent directly."""
        thread_id = "test-thread-general"
        state = {
            "messages": [
                {"type": "human", "content": "Hello, how are you today?"}
            ],
            "thread_id": thread_id
        }
        result = graph.invoke(state, config={"configurable": {"thread_id": thread_id}})
        
        self.assertEqual(result.get("intent"), "general")
        self.assertEqual(result.get("route_to"), "end")
        self.assertIn("how can i help you", result.get("last_response", "").lower())

    def test_relative_booking_tomorrow_5pm(self):
        """2. Relative booking: 'Book tomorrow at 5 PM' should normalize date/time and suggest alternatives."""
        thread_id = "test-thread-5pm"
        state = {
            "messages": [
                {"type": "human", "content": "Book tomorrow at 5 PM"}
            ],
            "thread_id": thread_id
        }
        result = graph.invoke(state, config={"configurable": {"thread_id": thread_id}})
        
        tomorrow_date = _normalize_relative_date("tomorrow")
        context = result.get("booking_context") or {}
        
        self.assertEqual(result.get("intent"), "booking")
        self.assertEqual(context.get("date"), tomorrow_date)
        self.assertEqual(context.get("time"), "17:00")
        
        # 5 PM (17:00) is outside DEFAULT_SLOTS. Assistant should suggest alternative slots.
        self.assertIn("requested slot is unavailable", result.get("last_response", "").lower())
        self.assertIn("choose one of the available alternatives", result.get("last_response", "").lower())

    def test_missing_information(self):
        """3. Missing information: Missing email/time/date should request required fields."""
        # Scenario A: Missing email
        thread_id_a = "test-thread-missing-email"
        state_a = {
            "messages": [
                {"type": "human", "content": "I want to book tomorrow at 10 AM"}
            ],
            "thread_id": thread_id_a
        }
        result_a = graph.invoke(state_a, config={"configurable": {"thread_id": thread_id_a}})
        self.assertIn("email", result_a.get("last_response", "").lower())
        
        context_a = result_a.get("booking_context") or {}
        self.assertEqual(context_a.get("time"), "10:00")
        self.assertIsNone(context_a.get("email"))

        # Scenario B: Missing date
        thread_id_b = "test-thread-missing-date"
        state_b = {
            "messages": [
                {"type": "human", "content": "Book an appointment at 10:00 for test@example.com"}
            ],
            "thread_id": thread_id_b
        }
        result_b = graph.invoke(state_b, config={"configurable": {"thread_id": thread_id_b}})
        self.assertIn("date", result_b.get("last_response", "").lower())

    def test_duplicate_slot(self):
        """4. Duplicate slot: Already booked slot should return alternative suggestions."""
        tomorrow_date = _normalize_relative_date("tomorrow")
        
        # Manually reserve a slot in the database
        with SessionLocal() as session:
            booking = Booking(date=tomorrow_date, time="10:00", email="existing@example.com")
            session.add(booking)
            session.commit()

        thread_id = "test-thread-duplicate"
        state = {
            "messages": [
                {"type": "human", "content": f"Book on {tomorrow_date} at 10 AM, my email is test@example.com"}
            ],
            "thread_id": thread_id
        }
        result = graph.invoke(state, config={"configurable": {"thread_id": thread_id}})
        
        # 10:00 is unavailable, should suggest alternatives and not contain 10:00 as alternative
        self.assertIn("requested slot is unavailable", result.get("last_response", "").lower())
        self.assertNotIn("10:00", result.get("last_response", "").replace("10:00 am", ""))

    def test_invalid_dates(self):
        """5. Invalid dates: Past dates and malformed dates should be rejected."""
        # Past date validation
        thread_id_past = "test-thread-past"
        state_past = {
            "messages": [
                {"type": "human", "content": "Book on 2020-01-01 at 10 AM, my email is test@example.com"}
            ],
            "thread_id": thread_id_past
        }
        result_past = graph.invoke(state_past, config={"configurable": {"thread_id": thread_id_past}})
        self.assertIn("date error", result_past.get("last_response", "").lower())
        self.assertIn("cannot be in the past", result_past.get("last_response", "").lower())

        # Malformed date validation
        thread_id_malformed = "test-thread-malformed"
        state_malformed = {
            "messages": [
                {"type": "human", "content": "Book on invalid-date-string at 10 AM, my email is test@example.com"}
            ],
            "thread_id": thread_id_malformed
        }
        result_malformed = graph.invoke(state_malformed, config={"configurable": {"thread_id": thread_id_malformed}})
        self.assertIn("date error", result_malformed.get("last_response", "").lower())

    def test_successful_booking(self):
        """6. Successful booking: Complete flow from request to database save and notification."""
        tomorrow_date = _normalize_relative_date("tomorrow")
        thread_id = "test-thread-success"
        
        # Step 1: Initial request (missing email)
        state_1 = {
            "messages": [
                {"type": "human", "content": f"I want to book an appointment on {tomorrow_date} at 11 AM"}
            ],
            "thread_id": thread_id
        }
        result_1 = graph.invoke(state_1, config={"configurable": {"thread_id": thread_id}})
        self.assertIn("email", result_1.get("last_response", "").lower())

        # Step 2: Provide missing email (completing requirements)
        state_2 = {
            "messages": [
                {"type": "human", "content": "My email is success@example.com"}
            ],
            "thread_id": thread_id
        }
        result_2 = graph.invoke(state_2, config={"configurable": {"thread_id": thread_id}})
        
        # Check confirmation response
        self.assertIn("successfully", result_2.get("last_response", "").lower())
        
        # Verify entry in local SQLite database
        with SessionLocal() as session:
            booking = session.query(Booking).filter(
                Booking.date == tomorrow_date,
                Booking.time == "11:00",
                Booking.email == "success@example.com"
            ).first()
            self.assertIsNotNone(booking)

        # Verify context completed flag
        context = result_2.get("booking_context") or {}
        self.assertTrue(context.get("completed"))


if __name__ == "__main__":
    unittest.main()
