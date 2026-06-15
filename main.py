# Convert voice to text
# Save text and input into llm
# Covert return text to voice
# Repeat as needed

import os

import speech_recognition as sr
from openai import OpenAI  # pyright: ignore[reportMissingImports]


def speech_to_text():

    recognizer = sr.Recognizer()

    """ recording the sound """

    with sr.Microphone() as source:
        print("Adjusting noise ")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Recording for 4 seconds")
        recorded_audio = recognizer.listen(source, timeout=4)
        print("Done recording")

    """ Recorgnizing the Audio """
    try:
        print("Recognizing the text")
        text = recognizer.recognize_google(recorded_audio, language="en-US")  # pyright: ignore[reportAttributeAccessIssue]
        print("Decoded Text : {}".format(text))
        return text

    except Exception as ex:
        print(ex)


# accepting LLM inputs


client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")


def ask_llm(prompt):
    response = client.chat.completions.create(
        model="Qwen2.5-VL-3B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=500,
    )

    return response.choices[0].message.content


output_of_speech = speech_to_text()
print("This is what was said: ", output_of_speech)
