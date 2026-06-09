# -Terminal-Chatbot
Ollama Terminal Chatbot
# Ollama Terminal Chatbot
<img width="1842" height="721" alt="image" src="https://github.com/user-attachments/assets/bc8ca8ee-2f37-4a77-b4c2-f81b562955ae" />

A simple terminal-based chatbot powered by Ollama and the OpenAI Python SDK. The application runs locally, maintains conversation history during the session, and supports basic user profile personalization through a JSON file.

## Features
!!! YOU HAVE TO DOWNLOAD OLLAMA
* Runs locally using Ollama
* Uses OpenAI-compatible API endpoints
* Maintains conversation context during a chat session
* Loads and saves user profile information
* Customizable system prompt based on user preferences
* Simple terminal interface
* Supports any Ollama model installed on your machine

## Requirements

* Python 3.10+
* Ollama installed and running
* OpenAI Python SDK

Install dependencies:

```bash
pip install openai
```

## Setup

1. Install Ollama from https://ollama.com
2. Start the Ollama server:

```bash
ollama serve
```

3. Pull a model (example: Llama 3):

```bash
ollama pull llama3
```

4. Update the `MODEL` variable in `bot.py` if needed.

## Usage

Run the chatbot:

```bash
python bot.py
```

Example:

```text
Ollama Chatbot (llama3)

You: Hello
Bot: Hi! How can I help you today?

You: Tell me about Python
Bot: Python is a popular programming language...
```

Type `quit` or `exit` to end the session.

## User Profile

The chatbot stores user information in `user_profile.json`.

Example:

```json
{
  "name": "John",
  "preferences": {
    "language": "English",
    "theme": "dark"
  }
}
```

Profile data is automatically loaded when the application starts and saved when it exits.

## Project Structure

```text
project/
├── bot.py
├── user_profile.json
└── README.md
```

## Configuration

Change the model by editing:

```python
MODEL = "llama3"
```

Examples:

```python
MODEL = "mistral"
MODEL = "phi3"
MODEL = "gemma"
```

## Error Handling

The application handles:

* Missing profile files
* Invalid JSON profile data
* Ollama API errors
* Keyboard interrupts (Ctrl+C)
* End-of-file interrupts (Ctrl+D)

## License

This project is provided for educational and personal use. Feel free to modify and extend it according to your needs.
