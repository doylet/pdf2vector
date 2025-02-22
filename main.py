import chromadb
import os
import shutil
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
from fastapi import FastAPI, UploadFile, File

app = FastAPI()
load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),  # This is the default and can be omitted
)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="gpt_completions")


def generate_gpt_completion(prompt: str):
    """Generate a GPT completion for a given prompt."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def generate_embedding(text: str):
    """Convert text into an embedding vector."""
    response = client.embeddings.create(
        model="text-embedding-3-large",
        inputs=[text]
    )
    return response["embeddings"][0]


def store_completion_in_vector_db(prompt: str):
    """Generate a GPT completion, embed it, and store in ChromaDB."""
    completion = generate_gpt_completion(prompt)
    embedding = generate_embedding(completion)
    
    # Store in ChromaDB
    collection.add(
        ids=[f"completion-{datetime.utcnow().isoformat()}"],
        embeddings=[embedding],
        metadatas=[{"prompt": prompt, "completion": completion}]
    )
    print(f"Stored GPT completion for: {prompt}")


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
    file_path = f"{UPLOAD_DIR}{file.filename}"

    # Save PDF locally
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process and store in vector database
    store_pdf_in_vector_db(file_path)

    return {"message": f"{file.filename} processed and stored."}

@app.get("/search/")
def search_pdf_api(query: str):
    results = search_pdf(query)
    return results


if __name__ == '__main__':
    # __test__generate_gpt_completion()