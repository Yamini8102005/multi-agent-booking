# Multi-Agent Appointment Booking System

A production-ready multi-agent appointment booking system built with Python, Streamlit, LangGraph, LangChain, Google Gemini, SQLite, SQLAlchemy, and Requests. The application routes user requests through a Triage Agent and a Booking Specialist, persists conversation state across refreshes, and stores bookings locally in SQLite.

## Overview

This project demonstrates a practical multi-agent architecture for appointment reservations. The assistant can:

- understand whether a user wants general help or an appointment booking;
- collect appointment details such as date, time, and email;
- normalize relative dates such as tomorrow and next Monday;
- check slot availability;
- negotiate alternatives when the requested time is unavailable;
- reserve the appointment and issue a webhook notification.

## Features

- ChatGPT-style Streamlit interface
- LangGraph-based orchestration with conditional routing
- Gemini-powered triage and specialist reasoning
- SQLite-backed booking persistence with SQLAlchemy
- Persistent conversation memory using LangGraph SqliteSaver
- Structured tool calls for availability checks, reservations, and webhook notifications
- Robust validation, logging, and exception handling

## Multi-Agent Architecture

- Triage Agent: classifies incoming user intent as general or booking-related
- Booking Specialist: collects required booking details, validates them, and triggers the booking tools
- Tool layer: handles slot availability checks, reservation logic, and outbound notifications

## LangGraph Workflow Diagram

```text
START
  ↓
Triage Agent
  ↓
Conditional Routing
  ├─ general → END
  └─ booking → Booking Specialist
                  ↓
            Tool Execution
                  ↓
                  END
```

## Folder Structure

```text
multi-agent-booking/
├── app.py
├── graph.py
├── agents.py
├── tools.py
├── database.py
├── prompts.py
├── requirements.txt
├── README.md
├── .env.example
└── data/
    ├── bookings.db
    └── memory.db
```

## Installation

1. Clone the repository.
2. Create and activate a Python 3.12+ virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a local environment file with the following values:

```env
GOOGLE_API_KEY=your_google_api_key_here
WEBHOOK_URL=https://example.com/webhook
```

## How to Run Locally

Run the Streamlit app:

```bash
streamlit run app.py
```

The application will open in your browser and allow you to chat with the booking assistant.

## Render Deployment Steps

1. Create a new Render Web Service.
2. Connect the GitHub repository.
3. Set the runtime to Python 3.12.
4. Use the following build command:

```bash
pip install -r requirements.txt
```

5. Use the following start command:

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

6. Add the required environment variables in the Render dashboard:
   - GOOGLE_API_KEY
   - WEBHOOK_URL

## How Persistent Memory Works

Conversation memory is backed by LangGraph SqliteSaver and stored in the SQLite file under the data directory. Each thread is identified by a thread_id so that the assistant can preserve previous dialogue context even after a Streamlit page refresh.

## Tool Descriptions

- check_availability(date): returns slot status information for the requested date.
- reserve_slot(date, time, email): validates and reserves an appointment slot.
- send_booking_notification(email, details): sends a webhook notification with booking details.

## Example Conversations

```text
User: Book me an appointment tomorrow at 10:00 for jane@example.com
Assistant: I’ll check the availability and reserve the slot if available.
```

```text
User: I need a meeting next Monday at 3pm for alex@example.com
Assistant: I’ll normalize the date and check the requested slot before reserving it.
```

## Technologies Used

- Python 3.12+
- Streamlit
- LangGraph
- LangChain
- Google Gemini
- SQLite
- SQLAlchemy
- Requests
- python-dotenv
- python-dateutil
- pydantic
- aiosqlite

## Future Improvements

- Add authentication and role-based access controls
- Integrate with external calendar providers such as Google Calendar or Microsoft Outlook
- Add richer conversation state with multi-turn confirmation flows
- Improve notification support with email and SMS backends
- Support multi-location and multi-provider scheduling
