from multiprocessing import Value
from pathlib import Path

import sounddevice as sd
import soundfile as sf
from mlx_audio.tts.generate import generate_audio

from prompts.voices import JIN_VOICES

MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"


def list_voices() -> list[str]:
    return sorted(JIN_VOICES.keys())


def speak_text(
    text: str,
    variant: str = "default",
    speed: float = 0.95,
    temperature: float = 0.6,
) -> Path:

    if variant not in JIN_VOICES:
        raise ValueError(
            f"Unknown voice variant '{variant}'. Availible: {', '.join(list_voices())}"
        )
    output_prefix = "jin_current"

    generate_audio(
        text=text,
        model=MODEL,
        instruct=JIN_VOICES[variant],
        speed=speed,
        temperature=temperature,
        file_prefix=output_prefix,
    )
    return Path(f"{output_prefix}_000.wav")


def play_audio(path: Path | str) -> None:
    """Play an audio file and wait until playback finishes."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    audio, sample_rate = sf.read(path, dtype="float32")

    try:
        sd.play(audio, sample_rate)
        sd.wait()
    finally:
        sd.stop()


def speak(
    text: str,
    variant: str = "default",
    speed: float = 0.95,
    temperature: float = 0.6,
) -> None:
    """
    Generate and immediately play speech.
    """

    wav = speak_text(
        text=text,
        variant=variant,
        speed=speed,
        temperature=temperature,
    )

    play_audio(wav)
