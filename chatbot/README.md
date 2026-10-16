# LangGraph Chatbot

A small chat application with a Streamlit interface and a LangGraph backend. It sends user messages to the Groq-hosted `openai/gpt-oss-20b` model and uses an in-memory checkpointer to retain graph conversation state while the backend process is running.

## Requirements

- Python 3.11 or newer
- A Groq API key

## Setup

From the repository root, activate the project's virtual environment (or create one) and install the chatbot dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install langgraph langchain-core langchain-groq python-dotenv streamlit
```

Add your Groq API key to the `.env` file in the repository root:

```dotenv
GROQ_API_KEY=your_groq_api_key
```

Keep the `.env` file private; do not commit API keys.

## Run

From the repository root, with the virtual environment active, run the in-memory version:

```powershell
streamlit run chatbot/frontend.py
```

Streamlit will print a local URL to open in your browser.

To run the database-backed version, which displays saved conversations in the sidebar, use:

```powershell
streamlit run chatbot/frontend_database.py
```

Run this command from the repository root. The database-backed backend opens `chatbot.db` using a path relative to the current working directory, so launching it from a different directory may create or use a different database file.

## How it works

- `backend.py` defines the LangGraph state, Groq chat model, chatbot node, and in-memory checkpointer.
- `frontend.py` renders the chat interface and sends each submitted message to the graph.
- The in-memory frontend keeps displayed messages in the Streamlit session and graph state in memory; this data is not persisted after the backend process exits.
- `backend_database.py` defines the SQLite-backed graph checkpointer and stores conversation titles in the `chat_titles` table.
- `frontend_database.py` provides the sidebar for browsing saved conversations. Its conversation list is loaded from the SQLite checkpoints when the Streamlit session starts.
