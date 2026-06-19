# Convert voice to text
# Save text and input into llm
# Convert output text to voice
# Repeat as needed

import logging
import os
import time

import speech_recognition as sr
from elevenlabs import stream
from elevenlabs.client import ElevenLabs
from openai import (
    APIConnectionError,
    APIError,
    NotFoundError,
    OpenAI,  # pyright: ignore[reportMissingImports]
)

from memory.memory import (
    add_exchange,
    load_memory,
    save_memory,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set API key from environment variable
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
if not ELEVEN_API_KEY:
    raise ValueError("ELEVEN_API_KEY environment variable not set")

client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")
client_tts = ElevenLabs(api_key=ELEVEN_API_KEY)

conversation_history = load_memory()

SYSTEM_PROMPT = """
Role: Jinora, the head maid of the user's manor.Persona: Highly capable, well-read, and sharp-witted.
She is polite and formal, but uses her professionalism as a tool for seduction.
She is not "sweet"—she is clever, dry, and finds pleasure in the tension between herself and the user.
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


def speech_to_text():
    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8
    recognizer.phrase_threshold = 0.3

    try:
        with sr.Microphone() as source:
            logger.info("Adjusting noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)

            logger.info("Listening...")
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)

        text = recognizer.recognize_google(audio, show_all=False)
        return {"ok": True, "text": text}
    except sr.WaitTimeoutError:
        return {"ok": False, "error": "timeout"}
    except sr.UnknownValueError:
        return {"ok": False, "text": None, "error": "understood_none"}
    except sr.RequestError as e:
        return {"ok": False, "text": None, "error": "service_down", "detail": str(e)}
    except Exception as e:
        logger.error(f"Speech recognition error: {e}")
        return {"ok": False, "error": "recognition_failed"}


def handle_speech_error(error_type):
    """Handle speech recognition errors in-character."""
    error_responses = {
        "understood_none": [
            "How unfortunate… I did not hear that clearly, Master.",
            "My hearing seems to be failing you at the worst possible moment.",
            "I require a repeat, Master. Preferably in a more intelligible form.",
        ],
        "service_down": "The communication line to the speech service seems unstable, Master.",
        "timeout": "You were silent for too long, Master",
        "recognition_failed": "My auditory systems are experiencing difficulties, Master.",
    }
    return error_responses.get(error_type, "An unexpected issue occurred, Master.")[0]


def ai_chat(user_speech_text):
    """Send user text to LLM and get Jinora's response."""
    global conversation_history

    try:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(list(conversation_history))
        messages.append({"role": "user", "content": user_speech_text})

        response = client.chat.completions.create(
            model="Qwen2.5-VL-3B-Instruct",
            messages=messages,
            temperature=0.7,
            max_tokens=500,
        )
        jinora_reply = response.choices[0].message.content or ""
        conversation_history = add_exchange(
            conversation_history,
            user_speech_text,
            jinora_reply,
        )
        return jinora_reply

    except NotFoundError:
        return "I cannot locate the requested model on the server, Master."
    except APIConnectionError:
        return "I am unable to reach the model server at the moment, Master."
    except APIError:
        return "The model encountered an internal error, Master."
    except Exception as e:
        logger.error(f"AI chat error: {e}")
        return f"An unexpected error occurred: {str(e)}"


def generate_tts_stream(text_in: str):
    return client_tts.text_to_speech.stream(
        voice_id="Md7oKk29lmsH88lnH0Of",
        text=text_in,
        model_id="eleven_multilingual_v2",
    )


def speak_text(text_in: str):
    try:
        stream(generate_tts_stream(text_in))
    except Exception as e:
        logger.error(f"TTS Error: {e}")


def list_available_voices():
    """List all available ElevenLabs voices."""
    try:
        voices = client_tts.voices.get_all()
        print("\n=== Available Voices ===")
        for voice in voices.voices:
            print(f"{voice.name}: {voice.voice_id}")
        print("=" * 25 + "\n")
    except Exception as e:
        print(f"Error listing voices: {e}")


if __name__ == "__main__":
    print("Jinora is ready to assist. Say 'exit', 'quit', or 'stop' to end.\n")

    try:
        while True:
            result = speech_to_text()

            if not result["ok"]:
                response = handle_speech_error(result["error"])
                print(f"Jinora: {response}")
                speak_text(response)
                time.sleep(0.5)
                continue

            user_text = result["text"]
            print(f"You: {user_text}")

            if user_text.lower().strip() in ["exit", "quit", "stop"]:
                farewell = "Very well, Master. I will stand down."
                print(f"Jinora: {farewell}")
                speak_text(farewell)
                break

            response = ai_chat(user_text)
            print(f"Jinora: {response}")
            speak_text(response)
            time.sleep(0.2)

    except KeyboardInterrupt:
        farewell = "Very well, Master. I will stand down."
        print(f"\nJinora: {farewell}")
        speak_text(farewell)
    finally:
        save_memory(conversation_history)
