# OpenAI Chatbot

An AI-powered CLI chatbot built with Python and the `openai` library. It maintains conversation context within a session and persists user information across sessions via a JSON profile.

## Prerequisites

- Python 3.8 or higher
- `pip`
- An [OpenAI API key](https://platform.openai.com/api-keys)

## Installation

```bash
pip install openai
```

Or install from the requirements file:

```bash
pip install -r requirements.txt
```

## Setting the API key

The chatbot reads your OpenAI API key from the `OPENAI_API_KEY` environment variable. Never hardcode credentials in source files.

**macOS / Linux:**

```bash
export OPENAI_API_KEY="sk-..."
```

**Windows (Command Prompt):**

```cmd
set OPENAI_API_KEY=sk-...
```

**Windows (PowerShell):**

```powershell
$env:OPENAI_API_KEY = "sk-..."
```

The chatbot will exit with a descriptive error if the variable is not set.

## Running the chatbot

From inside the `chatbot2-openai/` directory:

```bash
python bot.py
```

Type your message and press Enter. Type `quit` or `exit` to end the session.

## How conversation history works

Every message exchanged during a session is kept in a `conversation_history` list in memory. Each entry is a dict with a `role` (`system`, `user`, or `assistant`) and `content` string.

When calling the OpenAI API, the full history is sent with every request:

```python
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=conversation_history  # includes system prompt + all prior turns
)
```

This gives the model complete context of the conversation, enabling coherent multi-turn dialogue. The history grows by two entries per turn (one user message, one assistant response) and is held in memory for the duration of the session only.

## How `user_profile.json` persists information across sessions

When the chatbot starts, it loads `user_profile.json` from the same directory. If the file does not exist (or is malformed), it starts with an empty profile:

```json
{
  "name": null,
  "preferences": {}
}
```

The profile is injected into the system prompt at the start of the conversation, so the model is immediately aware of previously stored information. When the session ends (via `quit`, `exit`, or Ctrl+C), the updated profile is saved back to `user_profile.json`.

This means details like the user's name or preferences are remembered the next time the chatbot is launched.

## Project structure

```
chatbot2-openai/
├── bot.py               # Entry point: manages history, calls OpenAI API
├── user_profile.json    # Persisted user info (created at runtime)
├── requirements.txt
└── README.md
```

## Python version

Tested with **Python 3.8+**. The `openai` library requires Python 3.7.1 or higher; Python 3.8+ is recommended.
