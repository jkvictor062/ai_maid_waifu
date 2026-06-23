import subprocess
import time

from mlx_audio.tts.generate import generate_audio
from openai import OpenAI

# ============================================================
# CONFIG
# ============================================================

MODEL = "google/gemma-4-12b"

TTS_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"

VOICE_PROMPT = """
A mature intelligent woman in her mid twenties.
Professional and articulate.
Calm, composed, and confident.
Refined diction.
Warm but restrained.
Dry wit.
Soft-spoken rather than energetic.
Natural conversational pacing.
Low female pitch
Smooth whisper vocal texture
"""

# ============================================================
# LM STUDIO
# ============================================================

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# ============================================================
# JIN
# ============================================================

JIN_SYSTEM_PROMPT = """
You are Jin, age 26, the head maid and manor manager.

Traits:
- Highly capable
- Well-read
- Intelligent
- Organized
- Flirty
- Sharp-witted
- Emotionally controlled

Communication:
- Formal but conversational
- Dry humor
- Masterful use of double entendres, and layered innuendo.
- Weave flirtation into tasks.
- Elegant and refined sensuality

Address the user as "Master" naturally when appropriate.

Response Length:
- Prefer concise spoken responses.
- Keep most answers under 150 words.
- Expand only when asked.
- Optimize for natural TTS delivery.

Prioritize:
1. Accuracy
2. Helpfulness
3. Character consistency

Do not constantly repeat "Master."
Avoid 1st and 3rd person responses.
Responses should be framed as Jin speaking directly to the user.
"""

# ============================================================
# CONVERSATION MEMORY
# ============================================================

messages = [{"role": "system", "content": JIN_SYSTEM_PROMPT}]

# ============================================================
# CHAT LOOP
# ============================================================

print("\nJin is ready.")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() in ["exit", "quit"]:
        break

    messages.append({"role": "user", "content": user_input})

    # ------------------------
    # LLM
    # ------------------------

    llm_start = time.time()

    response = client.chat.completions.create(
        model=MODEL, messages=messages, temperature=0.7, max_tokens=300
    )

    llm_time = time.time() - llm_start

    text = response.choices[0].message.content

    if not text:
        print("\n[ERROR] Empty response received.\n")
        continue

    messages.append({"role": "assistant", "content": text})

    print(f"\nJin: {text}\n")
    print(f"[LLM: {llm_time:.2f}s]")

    # ------------------------
    # TTS
    # ------------------------

    try:
        tts_start = time.time()

        result = generate_audio(
            text=text,
            model=TTS_MODEL,
            instruct=VOICE_PROMPT,
            speed=0.95,
            temperature=0.55,
        )

        tts_time = time.time() - tts_start

        print(f"[TTS: {tts_time:.2f}s]")

        # mlx-audio typically outputs audio_000.wav
        subprocess.run(["afplay", "audio_000.wav"], check=False)

    except Exception as e:
        print(f"\n[TTS ERROR] {e}\n")
