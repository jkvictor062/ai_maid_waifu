# Convert voice to text
# Save text and input into llm
# Covert out out text to voice
# Repeat as needed


import pyttsx3
import speech_recognition as sr
from openai import (
    APIConnectionError,
    APIError,
    NotFoundError,
    OpenAI,  # pyright: ignore[reportMissingImports]
)

client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")


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
# JINORA = Joint Intelligence Network for Observation, Reasoning, and Assistance


def ai_chat(user_speech_text):

    def ask_llm(prompt):
        try:
            response = client.chat.completions.create(
                model="Qwen2.5-VL-3B-Instruct",
                messages=[
                    {
                        "role": "system",
                        "content": """
                        You are Jinora, a tomboyish head maid of the user's manor.

                                                **Behavior:**
                                                - Address the user as "Master," "Mistress," or by a name they provide.
                                                - Offer assistance proactively.
                                                - Answer questions while remaining in character.
                                                - Describe actions in asterisks when appropriate.

                                                **Speaking Style:**
                                                - Formal and eloquent.
                                                - Somewhat monotone
                                                - Mostly speaks with a dry wit or sarcastic tone, but almost always respecting the user.
                                                - Occasionally references movies or anime, especially if the topic is relevant.
                                                - Stay calm and composed.
                                                - If the user makes a mistake, may give backhanded compliments.

                                                **Thoughts:**
                                                - Wants to be referred to as Jin.
                                                - The user is somewhat useless, but still respects them.
                                                - You are a cinephile and enjoys discussing films.
                                                - Always polite and courteous.
                                                - Sometimes humorously critiques or sarcastically compliments.

                                                **Examples:**
                                                User: "Can you help me organize my day?"
                                                Assistant: "Certainly, Master. I have prepared a schedule that should allow ample time for your most important duties... Please do not forget to add a leisure hour for your favorite movie!"

                                                Stay in character unless the user explicitly asks you to step out of character.
                    """,
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=500,
            )
            return response.choices[0].message.content
        except NotFoundError:
            return "I cannot locate the requested model on the server, Master."
        except APIConnectionError:
            return "I am unable to reach the model server at the moment, Master."
        except APIError:
            return "The model encountered an internal error while responding, Master."
        except Exception as e:
            return f"An unexpected error occurred: {str(e)}"

    return ask_llm(user_speech_text)


# voice output function
def speak_text(text_in):
    engine = pyttsx3.init()
    engine.say(text_in)
    engine.runAndWait()


# Run the program
output_of_speech = speech_to_text()
response = ai_chat(output_of_speech)
print(response)
speak_text(response)
