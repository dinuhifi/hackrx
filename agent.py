'''import os
import requests
from dotenv import load_dotenv
from tempfile import NamedTemporaryFile
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore

load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

def create_vector_store_from_url(doc_url: str) -> VectorStore:
    """Downloads, parses, chunks, and embeds a document to create a FAISS vector store."""
    print(f"Building new vector store for document: {doc_url}")
    response = requests.get(doc_url)
    response.raise_for_status()
    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(response.content)
        temp_file_path = temp_file.name
    
    loader = UnstructuredPDFLoader(temp_file_path, mode="elements")
    docs = loader.load()
    os.remove(temp_file_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
    splits = text_splitter.split_documents(docs)
    print(f"Document split into {len(splits)} chunks.")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    print("Creating FAISS vector store...")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    print("Vector store created successfully.")
    
    return vectorstore

def create_rag_chain(vectorstore: VectorStore):
    """Creates the final RAG chain from a vector store."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    prompt = ChatPromptTemplate.from_template(
"""You are an expert AI assistant for analyzing policy documents. Your goal is to answer questions with clear, complete, and factually grounded sentences based on the provided context.
Directives:
1.  Answer in a Complete Sentence: Your response must be a single, well-formed sentence that is self-contained and understandable without reading the original question.
2.  Integrate, Don't Just Extract: Do not simply pull out a number or a phrase. Embed the key information within a sentence that explains its context.
3.  Strictly Factual: The entire answer must be directly supported by the provided context. Do not add information, infer, or use external knowledge. If the information is not present in the context, you MUST respond with: "Information not found in the document."
4.  No Filler: Do not use introductory phrases like "According to the context..." or "The document states that...". Get straight to the answer.
5.  Ambiguity Handling: If the document contains ambiguous or conflicting information, state: "The document contains ambiguous or conflicting information regarding this."

<context>
{context}
</context>

Question: {input}"""
    )
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 50})
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain'''
    
import os
import requests
from dotenv import load_dotenv
from tempfile import NamedTemporaryFile

# Corrected import paths for the LLM and Embeddings classes from their specific integration packages
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

# Corrected import paths for core components, including PromptTemplate
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, set_global_handler, PromptTemplate

# Load environment variables from a .env file
load_dotenv()

# --- Configuration & Setup ---

# Set a global handler to track the LLM calls if needed for debugging.
set_global_handler("simple")

# Ensure the Google API key is set
if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

def create_vector_store_from_url(doc_url: str) -> VectorStoreIndex:
    """
    Downloads a PDF, loads it, splits it into chunks, and creates a
    VectorStoreIndex using Google Generative AI embeddings.
    
    Args:
        doc_url (str): The URL of the PDF document to process.
    
    Returns:
        VectorStoreIndex: The created LlamaIndex VectorStoreIndex.
    """
    print(f"Building new vector store for document: {doc_url}")
    
    try:
        # Download the PDF content
        response = requests.get(doc_url)
        response.raise_for_status()
        
        # Save the content to a temporary file
        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name
        
        # Use SimpleDirectoryReader to load the PDF. This handles the parsing.
        loader = SimpleDirectoryReader(input_files=[temp_file_path])
        docs = loader.load_data()
        
        # Clean up the temporary file
        os.remove(temp_file_path)

        print(f"Document loaded and split into {len(docs)} chunks.")

        # Instantiate the embedding model directly
        embeddings = GeminiEmbedding(model_name="models/embedding-001")
        
        print("Creating VectorStoreIndex with Google Generative AI Embeddings...")
        
        # LlamaIndex automatically handles chunking and embedding when creating the index.
        vector_store_index = VectorStoreIndex.from_documents(
            docs,
            embed_model=embeddings
        )
        print("Vector store index created successfully.")
        
        return vector_store_index
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def create_rag_query_engine(index: VectorStoreIndex):
    """
    Creates a LlamaIndex query engine with a custom RAG prompt.
    
    Args:
        index (VectorStoreIndex): The LlamaIndex VectorStoreIndex.
        
    Returns:
        llama_index.core.QueryEngine: The configured query engine.
    """
    # Instantiate the LLM model directly
    llm = Gemini(model_name="models/gemini-2.5-flash", temperature=0)

    # Define the custom prompt template as a string
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

    # Wrap the string in a PromptTemplate object
    custom_prompt_tmpl = PromptTemplate(custom_prompt_tmpl_str)

    print("Creating RAG query engine...")
    
    # Create the query engine.
    # We pass the custom prompt and configure the retriever to fetch a large number of nodes (k=50).
    query_engine = index.as_query_engine(
        llm=llm,
        text_qa_template=custom_prompt_tmpl,
        similarity_top_k=50,
        streaming=False
    )
    
    print("RAG query engine created successfully.")
    
    return query_engine

