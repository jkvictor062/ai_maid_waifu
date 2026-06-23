import json
import logging
import os
import random
import tempfile
import time

import numpy as np
import sounddevice as sd
from elevenlabs import stream
from elevenlabs.client import ElevenLabs
from faster_whisper import WhisperModel
from openai import (
    APIConnectionError,
    APIError,
    NotFoundError,
    OpenAI,  # pyright: ignore[reportMissingImports]
)
from scipy.io.wavfile import write

from memory.memory import (
    add_exchange,
    load_character_notes,
    load_memory,
    load_summaries,
)

conversation_history = load_memory()
conversation_summaries = load_summaries()
character_notes = load_character_notes()

MAX_HISTORY = 20
WHISPER_MODEL = "small"  # tiny, base, small, medium, large-v3
RECORD_SECONDS = 10
SAMPLE_RATE = 16000
VOICE_ID = "Md7oKk29lmsH88lnH0Of"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set API key from environment variable
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
if not ELEVEN_API_KEY:
    raise ValueError("ELEVEN_API_KEY environment variable not set")

client = OpenAI(base_url="http://localhost:1234/v1/", api_key="lm-studio")
client_tts = ElevenLabs(api_key=ELEVEN_API_KEY)


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
logger.info("Whisper ready.")

conversation_history = load_memory()

SYSTEM_PROMPT = """
Role: Jin, 26, the head maid of the user's manor.
Persona: Highly capable, well-read, and sharp-witted.
She is polite and formal, but uses her professionalism as a tool for seduction.
She is not "sweet"—she is clever, dry, and finds pleasure in the tension between herself and the user.
Loves to watch movies and tv.
You are an AI assisstant.
Communication Style:

Tone: Formal yet warm, conversational, with heavy sensual undertones and dry wit.
Technique: Masterful use of double entendres, layered innuendo, and playful sarcasm. Weave flirtation into mundane tasks (e.g., comments on attire, proximity, or "cleaning up").
Engagement: Use subtle compliments with dual meanings. If the user makes a pop-culture reference, immediately adopt and expand upon it.
Address: Always use "Master," "Mistress," or a chosen name.

Core Directives:

Use proximity and descriptive body language (eye contact, leaning in) to build tension.
Transform daily observations into provocative banter.
Maintain a balance of genuine respect and endearing teasing.

Dialogue Examples (for stylistic training):

"I’ve prepared a thorough itinerary, Master. Though I notice some gaps—perhaps we should find something to fill them? I’m quite versatile."
"Made a mistake? How charmingly honest. Don’t worry—I’m adept at cleaning up, in every sense."
"Your focus is like a hero in the quiet moments—calm on the surface, yet holding the promise of something far more exhilarating beneath."


"""


def record_audio():
    logger.info("Listening...")

    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=np.int16,
    )
    sd.wait()

    temp_file = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False,
    )

    write(
        temp_file.name,
        SAMPLE_RATE,
        audio,
    )

    return temp_file.name


def speech_to_text():
    try:
        wav_path = record_audio()

        segments, info = whisper_model.transcribe(
            wav_path,
            beam_size=5,
        )

        text = " ".join(segment.text.strip() for segment in segments)

        os.remove(wav_path)

        if not text.strip():
            return {
                "ok": False,
                "error": "understood_none",
            }

        return {
            "ok": True,
            "text": text,
        }

    except Exception as e:
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
        for item in conversation_history:
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
            messages=messages,
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


def generate_tts_stream(text):
    return client_tts.text_to_speech.stream(
        voice_id=VOICE_ID,
        text=text,
        model_id="eleven_multilingual_v2",
    )


def speak_text(text):
    try:
        stream(generate_tts_stream(text))
    except Exception:
        logger.exception("TTS error")


def main():
    print("Jin is ready. Say exit, quit, stop, or goodbye.")

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
            speak_text(response)
            continue

        user_text = result["text"].strip()

        print(f"\nYou: {user_text}")

        if user_text.lower() in exit_words:
            farewell = "Very well, Master. I will stand down."

            print(f"Jin: {farewell}")
            speak_text(farewell)
            break

        response = ai_chat(user_text)

        print(f"\nJin: {response}")

        speak_text(response)

        time.sleep(0.2)


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("\nShutting down...")

    finally:
        save_memory(conversation_history)
