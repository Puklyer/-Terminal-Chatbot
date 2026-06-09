# chatbot2-openai/bot.py
# LLM chatbot powered by Ollama (local, free)
# Requires: pip install openai
# Requires: Ollama running locally -> https://ollama.com

import json
import os
import sys
from typing import Any

from openai import OpenAI, APIError

# Ollama runs a local server that is OpenAI API-compatible
OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL = "llama3"  # change to any model you have pulled, e.g. "mistral", "phi3"
PROFILE_PATH = os.path.join(os.path.dirname(__file__), "user_profile.json")


def load_profile(path: str) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"name": None, "preferences": {}}
    except json.JSONDecodeError:
        print("Warning: user_profile.json is malformed. Starting with an empty profile.")
        return {"name": None, "preferences": {}}


def save_profile(path: str, profile: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)


def build_system_message(profile: dict[str, Any]) -> str:
    parts = ["You are a helpful, friendly assistant."]
    if profile.get("name"):
        parts.append(f"The user's name is {profile['name']}.")
    prefs = profile.get("preferences", {})
    if prefs:
        pref_str = ", ".join(f"{k}: {v}" for k, v in prefs.items())
        parts.append(f"Known preferences: {pref_str}.")
    return " ".join(parts)


def main() -> None:
    # Ollama does not need a real API key, but the openai client requires a non-empty value
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

    profile = load_profile(PROFILE_PATH)
    conversation_history: list[dict[str, str]] = [
        {"role": "system", "content": build_system_message(profile)}
    ]

    print(f"Ollama Chatbot ({MODEL}) — type 'quit' or 'exit' to end the session.")
    print("Make sure Ollama is running: ollama serve")
    print("-" * 50)

    while True:
        try:
            user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            save_profile(PROFILE_PATH, profile)
            break

        if not user_input.strip():
            continue

        if user_input.strip().lower() in ("quit", "exit"):
            print("Goodbye!")
            save_profile(PROFILE_PATH, profile)
            break

        conversation_history.append({"role": "user", "content": user_input})

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=conversation_history,  # type: ignore[arg-type]
            )
            assistant_message = response.choices[0].message.content or ""
            conversation_history.append({"role": "assistant", "content": assistant_message})
            print(f"Bot: {assistant_message}")
        except APIError as e:
            print(f"Ollama error: {e}")
            print("Is Ollama running? Try: ollama serve")


if __name__ == "__main__":
    main()
