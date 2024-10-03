import os
import base64
from dotenv import load_dotenv
from anthropic import Anthropic

def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as image_file:
        binary_data = image_file.read()
        base_64_encoded_data = base64.b64encode(binary_data)
        base64_string = base_64_encoded_data.decode('utf-8')
        return base64_string


def get_client():
    load_dotenv()
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return client

def ask_question(client, image_path, question, init_prompt, temperature, deployment_name):
    message_list = [
        {
            "role": 'user',
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": get_base64_encoded_image(image_path)}},
                {"type": "text", "text": question}
            ]
        }
    ]
    response = client.messages.create(
        model=deployment_name,
        max_tokens=2048,
        messages=message_list,
        temperature=temperature,
        system=init_prompt,
    )
    return response.content[0].text