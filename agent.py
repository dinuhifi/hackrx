from dotenv import load_dotenv
import os
from google import genai
from google.genai import types
import httpx
from models import AnswerResponse

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
        questions, "Answer the questions based on the provided policy document. Do not include any additional information or context in your answers. Provide concise and direct answers.",],
    config={
        'response_mime_type': 'application/json',
        'response_schema': AnswerResponse,
    })
    print(response.text)
    return response.to_json_dict()