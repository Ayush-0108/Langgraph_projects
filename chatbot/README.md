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

From the repository root, with the virtual environment active:

```powershell
streamlit run chatbot/frontend.py
```

Streamlit will print a local URL to open in your browser.

## How it works

- `backend.py` defines the LangGraph state, Groq chat model, chatbot node, and in-memory checkpointer.
- `frontend.py` renders the chat interface and sends each submitted message to the graph.
- The UI keeps the displayed message history in the Streamlit session. The graph checkpointer stores conversation state in memory and does not persist it after the backend process exits.
