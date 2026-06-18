import json
import logging
from pathlib import Path
from typing import Dict, List

MEMORY_FILE = "memory.json"
MAX_HISTORY = 20

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_memory() -> List[Dict[str, str]]:
    """
    Load conversation history from memory.json.

    Returns:
        List of conversations exchanges or empty list if file doesn't exist or is invalid.
    """
    try:
        memory_path = Path(MEMORY_FILE)
        if not memory_path.exists():
            save_memory([])  # Initialize empty memory file
            return []

        with memory_path.open(mode="r", encoding="utf-8") as f:
            loaded = json.load(f)
            if not isinstance(loaded, list):
                logger.error(f"{MEMORY_FILE} has invalid format")
                return []
            return loaded
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse {MEMORY_FILE}: {e}")
    except Exception as e:
        logger.error(f"Error loading memory: {e}")
    return []


def save_memory(history: List[Dict[str, str]]) -> None:
    """
    Save conversation history to memory.json with atomic writes.

    Args:
        history: list of conversation exchanges to save.
    """
    try:
        memory_path = Path(MEMORY_FILE)
        temp_path = memory_path.with_suffix(".tmp")

        # Write to temp file first
        with temp_path.open(mode="w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        # Atomic replace
        temp_path.replace(memory_path)
    except Exception as e:
        logger.error(f"Error saving memory: {e}")
        # Clean up temp file if it exists
        temp_path.unlink(missing_ok=True)


def add_exchange(
    history: List[Dict[str, str]],
    user_text: str,
    jinora_text: str,
) -> List[Dict[str, str]]:
    """
    Add a new exchange to the conversation history and trim to MAX_HISTORY.

    Args:
        history: Current conversation history.
        user_text: User's input text.
        jinora_text: JIN's reply text.

    Returns:
        Updated history list.
    """
    if not isinstance(user_text, str) or not isinstance(jinora_text, str):
        logger.error("user_text and jinora_text must be strings.")
        return history

    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": jinora_text})

    # Trim history to MAX_HISTORY exchanges (user + assistant per exchange)
    if len(history) > MAX_HISTORY * 2:
        history = history[-MAX_HISTORY * 2 :]

    save_memory(history)
    return history
