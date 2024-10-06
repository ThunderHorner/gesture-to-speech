import openai
from settings import openapi_key

openai.api_key = openapi_key

def generate_gesture_sentence(gesture_data):
    # Construct the prompt with the provided gesture data
    prompt = (
        "Convert the following gesture data into a coherent sentence. Clean the noise."
        "Minimize repetitions and infer the intended message clearly:\n"
    )
    for i in gesture_data:
        prompt += f"{i}\n"

    # OpenAI API request to generate the sentence
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.7
    )

    # Extract the output sentence from the response
    sentence = response.choices[0].message['content'].strip()
    return sentence