import sys

from mlx_audio.tts.generate import generate_audio

MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"

TEXT = """
Master, this is the third time I have explained the procedure. I remain optimistic that the fourth attempt will be successful.
"""
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "default"
TEMPERATURE = float(sys.argv[3]) if len(sys.argv) > 2 else 0.6
SPEED = float(sys.argv[2]) if len(sys.argv) > 2 else 0.95

TEST = EST = sys.argv[2] if len(sys.argv) > 2 else "peak"
JIN_VOICES = {
    "professional": """
A woman in her mid twenties with a mature, sophisticated voice.

Highly intelligent, well-read, and professionally trained.
Speaks with impeccable diction and precise articulation.

Calm, composed, and emotionally controlled.
Never bubbly, childish, energetic, or overly sweet.

Confident without arrogance.
Authoritative without aggression.

Uses subtle amusement, dry wit, and teasing.
Sounds like someone who is always the most prepared person in the room.

Measured pacing.
Natural pauses.
Elegant and refined.

A professional head maid and manor manager whose competence is unquestionable.
Her professionalism carries a quiet seductive undertone,
but she never sounds flirtatious, needy, submissive, or overtly affectionate.

Low female pitch.
Whisper vocal texture.
Warm but controlled emotional expression.
""",
    "intellectual": """
    Highly educated and eloquent woman.
    Refined vocabulary.
    Speaks like a trusted advisor.
    Calm, analytical, and insightful.
    Low female pitch.
    Smooth whisper vocal texture.
    """,
    "manager": """
    Professional manor managerial woman.
    Highly intelligent and well-read.
    Commands attention effortlessly.
    Uses subtle amusement and dry wit.
    Soft but authoritative.
    Warm but emotionally controlled.
    Seduction through confidence and competence rather than sweetness.
    Low female pitch.
    Smooth whisper vocal texture.
    """,
    "default": """
    Elegant and sophisticated.
    Uses professionalism as a vehicle for seduction.
    Warm amusement beneath perfect composure.
    flirtatious.
    Low female pitch.
    whisper vocal texture.
    """,
    "cold_excellence": """
    Exceptionally competent.
    Emotionally controlled.
    High standards.
    Quiet disappointment rather than anger.
    Precision
    Low female pitch.
    Smooth whisper vocal texture.
    """,
    "asmr": """
    A mature, sophisticated woman in her mid twenties.

    Whisper.
    Very soft-spoken and close to the listener.
    Quiet, intimate conversational tone.

    Gentle whisper-like vocal texture.
    Measured pacing.
    Natural pauses.
    Relaxed breathing.

    Warm but emotionally controlled.
    Subtle amusement.
    Dry wit delivered softly.

    Never bubbly.
    Never childish.
    Never overly energetic.

    Sounds like a trusted head maid speaking quietly
    during a late evening conversation.

    Elegant.
    Refined.
    Comforting.
    Calm confidence.

    Low to medium female pitch.
    Clear articulation even at low volume.
    """,
}

TESTS = {
    "greeting": "Good evening, Master. How may I assist you today?",
    "wit": "I admire the confidence. The planning remains a separate matter.",
    "authority": "I would strongly recommend reconsidering that course of action.",
    "comfort": "You have done enough for one evening. The remaining work can wait until tomorrow.",
    "technical": "The API returned a malformed JSON payload during processing.",
    "peak": "Master, there is a meaningful distinction between an unforeseen complication and a preventable disaster. Unfortunately, your recent decision appears to have occupied both categories simultaneously.",
}

text = TESTS[TEST]

generate_audio(
    text=TEXT,
    model=MODEL,
    instruct=JIN_VOICES[VARIANT],
    speed=SPEED,
    temperature=TEMPERATURE,
    file_prefix=f"jin_{VARIANT}_{SPEED}_{TEMPERATURE}",
)

print(f"Generated: jin_{VARIANT}_000.wav")
