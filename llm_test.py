from pathlib import Path

from openai import OpenAI

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

response = client.chat.completions.create(
    model="google/gemma-4-12b",
    messages=[{"role": "user", "content": "Explain Python virtual environments."}],
)

text = response.choices[0].message.content

output_dir = Path("jin_evaluation")
output_dir.mkdir(exist_ok=True)

file_path = output_dir / "technical.txt"

print("Writing:", file_path.resolve())
print("Text:", repr(text))

file_path.write_text(text, encoding="utf-8")

print("Verification:")
print(file_path.read_text(encoding="utf-8"))
