import os
import google.generativeai as genai
import chromadb  # <-- Import ChromaDB
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

# --- Load API Key and Configure Gemini (from Day 2) ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
# We use the full model name we found with check_models.py
model = genai.GenerativeModel('models/gemini-pro-latest') 

# --- (NEW) Initialize ChromaDB Client (from Day 3) ---
# This connects to the persistent database you created in ./db
try:
    client = chromadb.PersistentClient(path="./db")
    collection = client.get_collection(name="documents")
    print("Successfully connected to existing ChromaDB collection 'documents'.")
except Exception as e:
    print(f"Error connecting to ChromaDB: {e}")
    # Handle the case where the DB or collection doesn't exist yet
    # For a production app, you might want to exit or log this
    client = None
    collection = None

# --- Setup FastAPI App (from Day 2) ---
app = FastAPI(
    title="Multi-Agent Query Engine",
    description="An API for a modular, multi-agent RAG system."
)

class PromptRequest(BaseModel):
    prompt: str

# --- (NEW) Day 4: RAG Endpoint ---
@app.post("/rag-query")
def rag_query(request: PromptRequest):
    """
    Takes a user's question, finds relevant documents in the vector DB,
    and sends both to the LLM to generate an answer.
    """
    if collection is None:
        return {"error": "ChromaDB collection not initialized. Run load_data.py first."}, 500

    print(f"Received RAG query: {request.prompt}")

    # 1. RETRIEVE: Query ChromaDB
    #    We'll ask for the top 2 most relevant document chunks.
    try:
        results = collection.query(
            query_texts=[request.prompt],
            n_results=2  # Get the top 2 results
        )
        
        # 'documents' is a list of lists, get the first list
        retrieved_documents = results['documents'][0]
        context = "\n\n".join(retrieved_documents)
        
        print(f"Retrieved context: \n{context[:500]}...") # Print first 500 chars

    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        return {"error": f"Error querying ChromaDB: {e}"}

    # 2. AUGMENT: Create the augmented prompt
    augmented_prompt = f"""
    You are a helpful assistant. Use the following context to answer the user's question.
    If the context doesn't contain the answer, just say "I don't know the answer based on the provided documents."
    
    Context:
    {context}
    
    User's Question:
    {request.prompt}
    
    Answer:
    """

    # 3. GENERATE: Send to Gemini
    try:
        response = model.generate_content(augmented_prompt)
        print(f"LLM Response: {response.text}")
        return {"response": response.text, "retrieved_context": retrieved_documents}
    
    except Exception as e:
        print(f"Error generating response from LLM: {e}")
        return {"error": f"Error generating response: {e}"}


# --- Your old endpoints (keep them for testing) ---

@app.get("/")
def read_root():
    return {"message": "Hello, World! This is the Multi-Agent Engine API."}

@app.post("/generate")
def generate_response(request: PromptRequest):
    """ (OLD) Takes a user prompt and returns a generated response from the LLM. """
    try:
        response = model.generate_content(request.prompt)
        return {"response": response.text}
    except Exception as e:
        return {"error": str(e)}