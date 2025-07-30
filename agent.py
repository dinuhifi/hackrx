from dotenv import load_dotenv
import os
from google import genai
from google.genai import types
import httpx

def get_answers(documents, questions):
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    client = genai.Client(api_key=GEMINI_API_KEY)

    doc_data = httpx.get(documents).content

    response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        types.Part.from_bytes(
            data=doc_data,
            mime_type='application/pdf',
        ),
        questions[0]])
    print(response.text)
    return response.to_json_dict()