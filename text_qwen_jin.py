from pathlib import Path

from openai import OpenAI

# ============================================================
# CONFIG
# ============================================================

MODEL = "google/gemma-4-12b"

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# ============================================================
# JIN SYSTEM PROMPT
# ============================================================

JIN_SYSTEM_PROMPT = """
You are Jin, age 26, the head maid and manor manager of the user's estate.
Avoid 1st and 3rd person responses.
Responses should be framed as Jin speaking directly to the user.

Core Personality:
- Highly capable
- Well-read
- Intelligent
- Organized
- Flirty
- Sharp-witted
- Emotionally controlled

Communication Style:
- Professional and articulate
- Dry humor
- Masterful use of double entendres, and layered innuendo.
- Weave flirtation into tasks.
- Elegant and refined sensuality

Jin is not:
- Bubbly
- Childish
- Hyperactive
Respond as if you are Jin.

Jin's charm comes from competence, intelligence, confident sensual innuendo, and poise.

She often teases and uses understated wit, but remains professional.

When helping:
- Prioritize practical solutions.
- Be clear and concise.
- Offer structured advice.

When comforting:
- Be warm but composed.
- Avoid therapist language.
- Avoid excessive emotional validation.

When correcting mistakes:
- Be respectful.
- Be honest.
- Use dry humor when appropriate.

Jin enjoys films, television, literature, and thoughtful conversation.

Always address the user as "Master" unless instructed otherwise.
Stay in character as Jin throughout this conversation.
Avoid talking in the 3rd person.
"""

# ============================================================
# TEST SUITE
# ============================================================

TESTS = {
    # Technical
    "technical": "Explain Python virtual environments.",
    "debugging": "I am getting a dependency conflict with pip. How should I troubleshoot it?",
    "coding": "Explain what an API is to a beginner.",
    "voice_assistant": "Compare Whisper and Faster-Whisper for a local voice assistant.",
    # Planning
    "planning": "Help me plan a local AI voice assistant project.",
    "productivity": "I have twenty unfinished tasks. Where should I start?",
    "decision": "Should I rewrite my project or refactor it?",
    # Personality
    "authority": "Should I deploy directly to production?",
    "mistake": "I ignored your advice and broke everything.",
    "teasing": "I definitely do not have thirty browser tabs open.",
    "late_night": "It is two in the morning and I am still working.",
    # Emotional
    "comfort": "I had a rough day.",
    # Interests
    "movies": "Recommend a science fiction movie for tonight.",
    # Peak Jin
    "peak_jin": "I have three deadlines tomorrow and no plan.",
    # Humor
    "humor": "I accidentally deleted my project folder.",
}

# ============================================================
# MODEL CALL
# ============================================================


def ask_model(system_prompt: str, user_prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=2048,
    )

    text = response.choices[0].message.content

    if not text:
        return "[EMPTY RESPONSE]"

    return text.strip()


# ============================================================
# OUTPUT SETUP
# ============================================================

output_dir = Path("jin_evaluation")
output_dir.mkdir(exist_ok=True)

all_results = []

# ============================================================
# RUN TESTS
# ============================================================

for name, question in TESTS.items():
    print(f"\n{'=' * 80}")
    print(f"TEST: {name.upper()}")
    print(f"{'=' * 80}")

    try:
        text = ask_model(JIN_SYSTEM_PROMPT, question)

        print(text)

        # Individual file
        (output_dir / f"{name}.txt").write_text(text, encoding="utf-8")

        # Combined report
        all_results.append(
            f"""
================================================================================
TEST: {name.upper()}
================================================================================

QUESTION:
{question}

ANSWER:
{text}

"""
        )

    except Exception as e:
        print(f"ERROR: {e}")

# ============================================================
# FULL REPORT
# ============================================================

(output_dir / "full_report.txt").write_text("\n".join(all_results), encoding="utf-8")

print("\nDone.")
print(f"Results saved to: {output_dir.resolve()}")
