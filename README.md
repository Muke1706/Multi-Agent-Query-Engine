# 🤖 Multi-Agent Query Engine

## Introduction

This project implements a modular, multi-agent system capable of answering user queries by intelligently routing them to the appropriate information source – either a private document collection or a live web search. It utilizes Retrieval-Augmented Generation (RAG) for document-based questions and leverages external tools for real-time data. The system is built using modern Python frameworks like FastAPI, Langchain, Langgraph, and ChromaDB, and is containerized with Docker for easy deployment.

---

## Architecture

The system follows a router-agent pattern orchestrated by Langgraph:

1.  **Router:** Receives the user query and uses an LLM (Gemini) to determine if the query requires information from local documents (`document_search`) or the web (`web_search`).
2.  **Agents:**
    * **DocumentAgent:** Performs RAG using a ChromaDB vector store (populated with Project Management documents) and Gemini to answer questions based on the provided context.
    * **WebSearchAgent:** Uses the Tavily Search API to find real-time information online.
3.  **Synthesizer:** Takes the raw output from the selected agent and uses an LLM (Gemini) to generate a clean, final answer for the user. It also demonstrates conceptual understanding of GraphRAG by attempting to extract key relationships.

---

## Features

* **Intelligent Routing:** Automatically determines the best information source (local documents vs. web).
* **Retrieval-Augmented Generation (RAG):** Answers questions based on content from private documents (PDFs loaded into ChromaDB).
* **Web Search Capability:** Fetches real-time information using the Tavily API.
* **Answer Synthesis:** Generates clear, concise answers from raw agent outputs.
* **Conceptual GraphRAG:** Demonstrates relationship extraction from text.
* **Dockerized:** Packaged for easy setup and execution using Docker.

---

## Technologies Used

* **Orchestration:** Langchain, Langgraph
* **LLM:** Google Gemini API (`gemini-pro-latest`)
* **Vector Database:** ChromaDB (Persistent Client)
* **Embeddings (Local):** Hugging Face Sentence Transformers (`all-MiniLM-L6-v2`)
* **Web Search:** Tavily Search API
* **Containerization:** Docker
* **UI (Demo):** Streamlit
* **Language:** Python 3.11

---

## Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Muke1706/Multi-Agent-Engine.git
    cd Multi-Agent-Engine
    ```

2.  **Create a `.env` file:** In the root directory, create a `.env` file and add your API keys:
    ```env
    GOOGLE_API_KEY=YOUR_GEMINI_API_KEY
    TAVILY_API_KEY=YOUR_TAVILY_API_KEY
    ```

3.  **Build the Docker image:** Make sure Docker Desktop is running and run:
    ```bash
    docker build -t multi-agent-engine .
    ```

---

## How to Run

1.  **Run the agent script directly (via Docker):**
    * This will run the `agent_graph.py` script once with the default test question.
    ```bash
    docker run --rm -it --env-file .env multi-agent-engine
    ```

2.  **Run the Streamlit UI (Locally):**
    * Ensure you have the Python requirements installed locally (`pip install -r requirements.txt`).
    * Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
    *(Note: The Streamlit app currently requires local Python setup. Dockerizing Streamlit involves more complex configurations not covered in this basic setup.)*

---

## Future Improvements

* Implement parallel agent execution for queries requiring both document and web search.
* Add error handling and retries within agent nodes.
* Integrate a proper graph database (e.g., Neo4j) for full GraphRAG capabilities.
* Add a `VisionAgent` for multimodal (image) queries.
* Fully containerize the Streamlit UI with the backend.
