from fastapi import FastAPI, Depends, HTTPException, Request, status
from agent import create_vector_store_from_url_pinecone, create_rag_query_engine, cleanup_pinecone_indexes
from models import APIRequest, AnswerResponse
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()
TEAM_API_KEY = os.getenv("TEAM_API_KEY")

# Removed vector_store_cache - now using Pinecone cloud storage

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

    doc_url = payload.documents
        
    print(f"Creating/accessing Pinecone vector store for {doc_url}")
    # Use Pinecone cloud storage instead of RAM cache
    vector_store_index = create_vector_store_from_url_pinecone(doc_url)
    if not vector_store_index:
        raise HTTPException(
            status_code=500, 
            detail="Failed to create vector store from document URL."
        )

    try:
        # Create the RAG query engine for answering questions
        rag_query_engine = create_rag_query_engine(vector_store_index)

        # Create a list of tasks for parallel processing
        tasks = [rag_query_engine.aquery(q) for q in payload.questions]
        
        print(f"Processing {len(tasks)} questions in parallel...")
        
        # Await all the tasks to complete
        results = await asyncio.gather(*tasks)
        print("All questions processed.")

        # Extract the answers from the LlamaIndex response objects
        answers = [str(res) for res in results]
        
        print(f"Generated answers: {answers}")
        
        cleanup_pinecone_indexes()  # Clean up old indexes
        
        return AnswerResponse(answers=answers)
    except Exception as e:
        print(f"An error occurred during LlamaIndex query invocation: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal Server Error: {str(e)}"
        )