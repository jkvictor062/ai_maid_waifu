import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from tkinter.constants import CHAR
from typing import Any, Dict, List

MEMORY_DIR = Path("memory")

SHORT_TERM_FILE = MEMORY_DIR / "short_term.json"
SUMMARIES_FILE = MEMORY_DIR / "summaries.json"
CHARACTER_NOTES_FILE = MEMORY_DIR / "character_notes.json"

MAX_HISTORY = 20

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _ensure_memory_files() -> None:
    MEMORY_DIR.mkdir(exist_ok=True)

    defaults = {
        SHORT_TERM_FILE: [],
        SUMMARIES_FILE: [],
        CHARACTER_NOTES_FILE: {},
    }

    for file_path, default in defaults.items():
        if not file_path.exists():
            _save_json(file_path, default)


def _load_json(file_path: Path, default: Any) -> Any:
    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            # Check if the loaded data is of the expected type
            if isinstance(default, list) and not isinstance(data, list):
                return default
            if isinstance(default, dict) and not isinstance(data, dict):
                return default
            return data
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Failed to load {file_path}, using defaults. Error: {e}")
        return default


def _save_json(file_path: Path, data: Any) -> None:
    temp_path = file_path.with_suffix(".tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        # Atomic replace
        temp_path.replace(file_path)
    except Exception as e:
        logger.error(f"Failed saving {file_path}: {e}")
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


# Memory modules


def load_memory() -> List[Dict]:
    _ensure_memory_files()
    return _load_json(SHORT_TERM_FILE, [])


def save_memory(history: List[Dict]) -> None:
    _ensure_memory_files()  # Internal helper or just call _save_json
    _save_json(SHORT_TERM_FILE, history)


def load_summaries() -> List[Dict]:
    _ensure_memory_files()
    return _load_json(SUMMARIES_FILE, [])


def save_summary(summary_text: str) -> None:
    _ensure_memory_files()
    summaries = load_summaries()
    summaries.append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": summary_text,
        }
    )
    _save_json(SUMMARIES_FILE, summaries)


def load_character_notes() -> Dict:
    _ensure_memory_files()
    return _load_json(CHARACTER_NOTES_FILE, {})


def save_character_note(key: str, value: Any) -> None:
    _ensure_memory_files()
    notes = load_character_notes()
    notes[key] = value
    _save_json(CHARACTER_NOTES_FILE, notes)


# Conversation management


def add_exchange(
    history: List[Dict],
    user_text: str,
    assistant_text: str,
) -> List[Dict]:
    if not isinstance(user_text, str) or not isinstance(assistant_text, str):
        return history

    timestamp = datetime.now(timezone.utc).isoformat()

    history.append({"role": "user", "content": user_text, "timestamp": timestamp})
    history.append(
        {"role": "assistant", "content": assistant_text, "timestamp": timestamp}
    )

    if len(history) > MAX_HISTORY * 2:
        # Calculate how many items are over the limit
        overflow_count = len(history) - (MAX_HISTORY * 2)
        overflow = history[:-overflow_count]

        if overflow:
            conversation_text = "\n".join(
                f"{m['role']}: {m['content']}" for m in overflow
            )
            save_summary(conversation_text)

        # Keep only the most recent items
        history = history[-(MAX_HISTORY * 2) :]

    save_memory(history)
    return history
