"""
chatbot1-aiml/bot.py
Entry point for the AIML rule-based chatbot.
"""

import os
import sys
import glob
import aiml


def load_kernel(aiml_dir: str) -> aiml.Kernel:
    """Initialize the AIML kernel and load all .aiml files from aiml_dir."""
    kernel = aiml.Kernel()
    # Suppress the default verbose output from python-aiml
    kernel.verbose(False)

    aiml_files = glob.glob(os.path.join(aiml_dir, "*.aiml"))
    if not aiml_files:
        print(f"Error: no .aiml files found in '{aiml_dir}'")
        sys.exit(1)

    for path in sorted(aiml_files):
        kernel.learn(path)

    return kernel


def run(kernel: aiml.Kernel) -> None:
    """Run the interactive CLI loop."""
    print("AIML Chatbot ready. Type 'quit' or 'exit' to stop.")
    print("Tip: try saying 'My name is Alice' or 'I like reading'.\n")

    while True:
        try:
            user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            sys.exit(0)

        # Skip whitespace-only input
        if not user_input.strip():
            continue

        # Exit on quit/exit (case-insensitive)
        if user_input.strip().lower() in ("quit", "exit"):
            print("Goodbye!")
            sys.exit(0)

        response = kernel.respond(user_input)
        print(f"Bot: {response}")


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    aiml_dir = os.path.join(script_dir, "aiml")
    kernel = load_kernel(aiml_dir)
    run(kernel)


if __name__ == "__main__":
    main()
