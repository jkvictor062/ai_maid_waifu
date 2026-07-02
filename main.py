import json
import logging
import os
import random

from faster_whisper import WhisperModel
from openai import (
    APIConnectionError,
    APIError,
    NotFoundError,
    OpenAI,  # pyright: ignore[reportMissingImports]
)

from audio.record import VoiceRecorder
from memory.memory import (
    add_exchange,
    load_character_notes,
    load_memory,
    load_summaries,
    save_memory,
)
from prompts.jin import CHARACTER
from prompts.runtime import RUNTIME_RULES
from tts.tts import speak

conversation_summaries = load_summaries()
character_notes = load_character_notes()

MAX_HISTORY = 20
WHISPER_MODEL = "small"  # tiny, base, small, medium, large-v3

SYSTEM_PROMPT = f"""
    {CHARACTER}

    {RUNTIME_RULES}
    """
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
)
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_loaded_model():
    try:
        models = client.models.list()

        if models.data:
            return models.data[0].id

    except Exception as e:
        logger.error(f"Failed to get model list: {e}")

    return "local-model"


MODEL_NAME = get_loaded_model()


# Whisper
logger.info("Loading Whisper model...")
whisper_model = WhisperModel(WHISPER_MODEL, device="auto", compute_type="int8")
logger.info("Loading recorder...")
recorder = VoiceRecorder()
logger.info("Jin is ready.")

conversation_history = load_memory()


def speech_to_text():
    try:
        audio = recorder.record()

        if audio is None:
            return {
                "ok": False,
                "error": "timeout",
            }

        segments, info = whisper_model.transcribe(
            audio,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=600,
            ),
        )

        text = " ".join(segment.text.strip() for segment in segments)

        if not text:
            return {
                "ok": False,
                "error": "understood_none",
            }

        return {
            "ok": True,
            "text": text,
        }

    except Exception:
        logger.exception("Transcription failed")

        return {
            "ok": False,
            "error": "recognition_failed",
        }


def handle_speech_error(error_type):
    error_responses = {
        "understood_none": [
            "How unfortunate… I did not hear that clearly, Master.",
            "My hearing seems to be failing you at the worst possible moment.",
            "I require a repeat, Master. Preferably in a more intelligible form.",
        ],
        "service_down": [
            "The communication line to the speech service seems unstable, Master."
        ],
        "timeout": ["You were silent for too long, Master."],
        "recognition_failed": [
            "My auditory systems are experiencing difficulties, Master."
        ],
    }

    responses = error_responses.get(
        error_type, ["An unexpected issue occurred, Master."]
    )

    return random.choice(responses)


def ai_chat(user_text: str) -> str:
    global conversation_history

    try:
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
        ]
        if conversation_summaries:
            summary_text = "\n\n".join(
                summary["summary"] for summary in conversation_summaries[-5:]
            )

            messages.append(
                {
                    "role": "system",
                    "content": (f"Previous conversation summaries:\n\n{summary_text}"),
                }
            )
        if character_notes:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Known persistent information:\n\n"
                        f"{json.dumps(character_notes, indent=2)}"
                    ),
                }
            )

        # Convert memory records into OpenAI format
        for item in conversation_history[-MAX_HISTORY:]:
            messages.append(
                {
                    "role": item["role"],
                    "content": item["content"],
                }
            )

        messages.append(
            {
                "role": "user",
                "content": user_text,
            }
        )

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,  # pyright ignore
            temperature=0.7,
            max_tokens=500,
        )

        reply = response.choices[0].message.content.strip()

        conversation_history = add_exchange(
            conversation_history,
            user_text,
            reply,
        )

        return reply

    except NotFoundError:
        return "I cannot locate the requested model, Master."

    except APIConnectionError:
        return "I cannot connect to LM Studio. Is the local server running, Master?"

    except APIError as e:
        logger.error(f"LM Studio API error: {e}")
        return "The model encountered an internal error, Master."

    except Exception as e:
        logger.exception("LLM error")
        return f"Unexpected error: {e}"


def main():
    logger.info("Jin is ready. Say exit, quit, stop, or goodbye.")

    exit_words = {
        "exit",
        "quit",
        "stop",
        "goodbye",
        "bye",
        "shutdown",
    }

    while True:
        result = speech_to_text()

        if not result["ok"]:
            response = handle_speech_error(result["error"])

            print(f"Jin: {response}")
            speak(response)
            continue

        user_text = result["text"].strip()

        print(f"\nYou: {user_text}")

        lower = user_text.lower()
        if any(word in lower for word in exit_words):
            farewell = "Very well, Master. I will stand down."

            print(f"Jin: {farewell}")
            speak(farewell)
            break

        response = ai_chat(user_text)

        print(f"\nJin: {response}")

        speak(response)


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("\nShutting down...")

    finally:
        save_memory(conversation_history)
