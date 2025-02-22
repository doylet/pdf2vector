import chromadb
import os
import shutil
import PyPDF2
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI()
# ✅ Add CORS Middleware to allow Chrome Extensions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change to your extension ID for security)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (POST, GET, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
)

UPLOAD_DIR = os.getenv("UPLOAD_DIR")

os.makedirs(UPLOAD_DIR, exist_ok=True)

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),  # This is the default and can be omitted
)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="gpt_completions")


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file"""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text


def chunk_text(text, chunk_size=500):
    """Split text into smaller chunks of `chunk_size` tokens"""
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)

    chunks = [tokens[i: i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    
    decoded_chunks = [tokenizer.decode(chunk) for chunk in chunks]
    return decoded_chunks


def generate_embedding(text: str):
    """Convert text into an embedding vector."""
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return response["embeddings"][0]


def store_pdf_in_vector_db(pdf_path):
    """Extract text from PDF, chunk it, generate embeddings, and store in ChromaDB"""
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)

    for i, chunk in enumerate(chunks):
        embedding = generate_embedding(chunk)

        # Store in ChromaDB
        collection.add(
            ids=[f"{pdf_path}-chunk-{i}"],
            embeddings=[embedding],
            metadatas=[{"pdf_name": pdf_path, "text": chunk}]
        )
    
    print(f"Stored {len(chunks)} chunks from {pdf_path} in the vector database.")

def generate_gpt_completion(prompt: str):
    """Generate a GPT completion for a given prompt."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def search_pdf(query):
    """Find the most relevant chunks based on a search query"""
    query_embedding = generate_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    
    return results


def search_similar_responses(query: str):
    """Find the most similar stored completions based on a query."""
    query_embedding = generate_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    return results


def __test__generate_gpt_completion():
    prompt = "What is the meaning of life?"
    completion = generate_gpt_completion(prompt)
    print(completion)


@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # Save PDF locally
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process and store in vector database
    store_pdf_in_vector_db(file_path)

    return {"message": f"{file.filename} saved successfully!", "file_path": file_path}

@app.get("/search/")
def search_pdf_api(query: str):
    results = search_pdf(query)
    return results
