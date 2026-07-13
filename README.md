# Multi-Agent Appointment Booking Assistant

A production-ready multi-agent appointment booking assistant built using LangGraph, LangChain, Streamlit, and Google Gemini.

## Features

- Multi-agent workflow
- Intelligent booking routing
- Persistent conversation memory
- SQLite checkpointing
- Appointment availability checking
- Webhook notifications
- Streamlit chat interface

## Tech Stack

- Python
- Streamlit
- LangGraph
- LangChain
- Google Gemini
- SQLite
- SQLAlchemy

## Live Demo

https://multi-agent-booking.onrender.com

## Installation

```bash
git clone https://github.com/Yamini8102005/multi-agent-booking.git
cd multi-agent-booking

python -m venv .venv

source .venv/bin/activate
# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

Create a `.env`

```
GOOGLE_API_KEY=your_api_key
WEBHOOK_URL=your_webhook_url
```

Run

```bash
streamlit run app.py
```

## Project Structure

```
app.py
graph.py
agents.py
database.py
tools.py
prompts.py
data/
requirements.txt
README.md
```

## Example Flow

User:
Hi, I want to book an appointment.

↓

Assistant:
Please provide date, time and email.

↓

User:
Date: 2026-07-15
Time: 10:00 AM
Email: user@example.com

↓

Assistant:
Booking confirmed.