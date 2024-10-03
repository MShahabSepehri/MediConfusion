import os
from dotenv import load_dotenv
from IPython.display import Image
import google.generativeai as genai

def configure_client():
    # Load environment variables from .env file
    load_dotenv()

    # Access the API key
    genai.configure(api_key=os.getenv('GEMINI_API_KEY'))


def load_model(init_prompt, temperature, deployment_name):
    configure_client() 
    model = genai.GenerativeModel(deployment_name, 
                                  generation_config=genai.GenerationConfig(temperature=temperature),
                                  system_instruction=init_prompt,
                                  )
    return model

# Based on https://github.com/google-gemini/cookbook/blob/main/quickstarts/Prompting.ipynb
def ask_question(model, image_path, question):
    img = Image(image_path)
    response = model.generate_content([question, img])
    return response.text