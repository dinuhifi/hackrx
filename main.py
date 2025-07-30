from fastapi import FastAPI, Depends, HTTPException, Request, status
from dotenv import load_dotenv
from models import APIRequest
import os
from agent import get_answers

load_dotenv()
TEAM_API_KEY = os.getenv("TEAM_API_KEY")

async def verify_api_key(request: Request):

    auth_header = request.headers.get("Authorization")
    if not auth_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header is missing",
        )
        
    parts = auth_header.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format. Must be 'Bearer <token>'",
        )

    token = parts[1]
    if token != TEAM_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key",
        )
    return True

app = FastAPI(
    title="Bajaj Hackathon API",
    description="An API to answer questions based on a policy document.",
    version="1.0.0",
)

@app.post("/hackrx/run", dependencies=[Depends(verify_api_key)])
async def process_policy_questions(payload: APIRequest):

    documents = payload.documents
    questions = payload.questions
    return get_answers(documents, questions)
