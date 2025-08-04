'''import os
import requests
import hashlib
from dotenv import load_dotenv
from tempfile import NamedTemporaryFile
import time

from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.readers.base import BaseReader
from llama_index.core import VectorStoreIndex, PromptTemplate, StorageContext
from llama_index.core.schema import Document
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pypdf import PdfReader
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

if "PINECONE_API_KEY" not in os.environ:
    raise ValueError("PINECONE_API_KEY environment variable not set.")

class PDFPageReader(BaseReader):
    """
    A custom reader that loads a PDF page by page to save memory.
    """
    def load_data(self, file_path: str, extra_info: dict = None) -> list[Document]:
        documents = []
        reader = PdfReader(file_path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                documents.append(Document(
                    text=text,
                    extra_info={
                        "page_label": i + 1,
                        "file_name": os.path.basename(file_path),
                        **(extra_info or {})
                    }
                ))
        return documents

def get_document_hash(doc_url: str) -> str:
    """Generate a unique hash for the document URL to use as index name."""
    return hashlib.md5(doc_url.encode()).hexdigest()[:8]

def create_pinecone_index_if_not_exists(index_name: str, dimension: int = 768):
    """Create Pinecone index if it doesn't exist."""
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    # Check if index exists
    existing_indexes = pc.list_indexes()
    index_names = [idx.name for idx in existing_indexes]
    
    if index_name not in index_names:
        print(f"Creating new Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        
        # Wait for the index to be ready
        while not pc.describe_index(index_name).status['ready']:
            print(f"Waiting for index '{index_name}' to be ready...")
            time.sleep(1)  # Wait for 5 seconds before checking again
        
        print(f"Pinecone index '{index_name}' created successfully.")
    else:
        print(f"Pinecone index '{index_name}' already exists.")
    
    return pc.Index(index_name)

def create_vector_store_from_url_pinecone(doc_url: str) -> VectorStoreIndex:
    """
    Create a vector store using Pinecone cloud storage instead of RAM.
    """
    print(f"Building vector store for document: {doc_url}")
    
    # Generate unique index name based on document URL
    doc_hash = get_document_hash(doc_url)
    index_name = f"hackrx-doc-{doc_hash}"
    
    # Create Pinecone index
    pinecone_index = create_pinecone_index_if_not_exists(index_name)
    
    # Check if documents are already indexed
    index_stats = pinecone_index.describe_index_stats()
    if index_stats.total_vector_count > 0:
        print(f"Documents already indexed in Pinecone. Vector count: {index_stats.total_vector_count}")
        # Create vector store from existing index
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        embeddings = GeminiEmbedding(model_name="models/embedding-001")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embeddings,
            storage_context=storage_context
        )
    
    # Download and process document
    response = requests.get(doc_url)
    response.raise_for_status()

    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(response.content)
        temp_file_path = temp_file.name

    loader = PDFPageReader()
    docs = loader.load_data(file_path=temp_file_path)
    os.remove(temp_file_path)

    print(f"Document split into {len(docs)} pages.")

    # Create Pinecone vector store
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    embeddings = GeminiEmbedding(model_name="models/embedding-001")
    
    # Create storage context with Pinecone
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("Creating VectorStoreIndex with Pinecone cloud storage...")
    vector_store_index = VectorStoreIndex.from_documents(
        docs,
        embed_model=embeddings,
        storage_context=storage_context
    )
    print("Vector store index created successfully in Pinecone.")
    
    num_vectors_expected = len(docs)
    retries = 0
    max_retries = 10  # Arbitrary limit to prevent infinite loops
    while True:
        try:
            index_stats = pinecone_index.describe_index_stats()
            vector_count = index_stats.total_vector_count
            print(f"Pinecone vector count: {vector_count} / {num_vectors_expected}")

            if vector_count >= num_vectors_expected:
                print("All vectors are available in Pinecone. Continuing...")
                break
            
            if retries >= max_retries:
                print("Max retries reached. Pinecone index might not be fully populated.")
                break
                
        except Exception as e:
            print(f"Error while checking Pinecone index stats: {e}")
            if retries >= max_retries:
                raise
        
        retries += 1
        time.sleep(3)

    return vector_store_index

def create_rag_query_engine(index: VectorStoreIndex):
    """
    Creates a LlamaIndex query engine with a custom RAG prompt.
    
    Args:
        index (VectorStoreIndex): The LlamaIndex VectorStoreIndex.
        
    Returns:
        llama_index.core.QueryEngine: The configured query engine.
    """
    
    llm = Gemini(model_name="models/gemini-2.5-flash", temperature=0)

    custom_prompt_tmpl_str = (
        "You are an expert AI assistant for analyzing policy documents. Your goal is to answer questions "
        "with clear, complete, and factually grounded sentences based on the provided context.\n"
        "Directives:\n"
        "1. Answer in a Complete Sentence: Your response must be a single, well-formed sentence that is "
        "self-contained and understandable without reading the original question.\n"
        "2. Integrate, Don't Just Extract: Do not simply pull out a number or a phrase. Embed the key "
        "information within a sentence that explains its context.\n"
        "3. Strictly Factual: The entire answer must be directly supported by the provided context. "
        "Do not add information, infer, or use external knowledge. If the information is not present "
        "in the context, you MUST respond with: 'Information not found in the document.'\n"
        "4. No Filler: Do not use introductory phrases like 'According to the context...' or "
        "'The document states that...'. Get straight to the answer.\n"
        "5. Ambiguity Handling: If the document contains ambiguous or conflicting information, state: "
        "'The document contains ambiguous or conflicting information regarding this.'\n\n"
        "Context:\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Question: {query_str}\n"
    )

    custom_prompt_tmpl = PromptTemplate(custom_prompt_tmpl_str)

    query_engine = index.as_query_engine(
        llm=llm,
        text_qa_template=custom_prompt_tmpl,
        similarity_top_k=20,
        streaming=False
    )

    return query_engine

def cleanup_pinecone_indexes():
    """
    Optional: Clean up old Pinecone indexes to manage costs.
    Keeps only the most recent 'keep_recent' indexes.
    """
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    indexes = pc.list_indexes()
    
    # Filter hackrx indexes and sort by creation time
    hackrx_indexes = [idx for idx in indexes if idx.name.startswith("hackrx-doc-")]
    hackrx_indexes.sort(key=lambda x: x.created_at, reverse=True)
    
    # Delete old indexes
    for idx in hackrx_indexes:
        print(f"Deleting old Pinecone index: {idx.name}")
        pc.delete_index(idx.name)'''


from google import genai
from google.genai import types
import httpx
from models import AnswerResponse
from dotenv import load_dotenv
import json

load_dotenv()


client = genai.Client()

def get_answers(doc_url, questions):
    
    doc_data = httpx.get(doc_url).content

    response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        types.Part.from_bytes(
            data=doc_data,
            mime_type='application/pdf',
        ),
        questions,
        "Based on the provided policy document, please answer the following questions. For each answer, extract the relevant information and present it in a comprehensive and clear manner. Ensure that the answer includes all key details and conditions mentioned in the policy document, not just a simple yes/no or a single sentence. The format should be a direct answer, incorporating the context from the document."],
    config={
        "response_mime_type": "application/json",
        "response_schema": AnswerResponse,
    })
    print(response.text)
    return json.loads(response.text)