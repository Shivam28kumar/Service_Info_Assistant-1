# Service Information Assistant (Agentic RAG)

A comprehensive AI assistant built for "TechGadgets," designed to ingest company documentation, answer questions using Retrieval-Augmented Generation (RAG), and perform calculations using Agentic tools.

This project demonstrates an end-to-end implementation of **GenAI**, **Vector Databases**, **LLM Orchestration**, and **Agentic Logic** using a completely free tech stack (Hugging Face + LangChain).

---

## 🏗 Architecture Overview

The system is built on a modular architecture separating Data Ingestion, Retrieval, and Agentic Orchestration.

### Data Flow
1.  **Ingestion Layer (`ingestion.py`):** 
    *   Loads raw Markdown documents (`.md`) from the `data/` directory.
    *   Splits text into semantic chunks (size: 500 chars, overlap: 50) using `RecursiveCharacterTextSplitter`.
2.  **Memory Layer (`embeddings.py`):**
    *   Generates vector embeddings using the **Hugging Face `all-MiniLM-L6-v2`** model.
    *   Stores vectors in a local **ChromaDB** persistent directory.
3.  **RAG Layer (`rag_pipeline.py`):**
    *   Retrieves top-k relevant documents based on semantic similarity.
    *   Augments the LLM prompt with retrieved context.
4.  **Agent Layer (`agent.py`):**
    *   Uses a **ReAct (Reasoning + Acting)** agent structure.
    *   **Router Logic:** Decides whether to use the `Company_Knowledge_Base` (RAG) tool or the `Pricing_Calculator` (Python) tool based on user intent.
5.  **Interface Layer (`api.py`):**
    *   Exposes the Agent via a **FastAPI** REST endpoint.

### Technology Stack
*   **Orchestration:** LangChain (Core, Community, HuggingFace)
*   **LLM:** Hugging Face Serverless Inference API (Mistral-7B / Zephyr-7b) openai/gpt-oss-120b--used this one 
*   **Vector Database:** ChromaDB (Local)
*   **Embeddings:** Sentence-Transformers (all-MiniLM-L6-v2)
*   **API Framework:** FastAPI & Uvicorn

---

## 🚀 Setup Instructions

### Prerequisites
*   Python 3.10 or higher
*   A free [Hugging Face Account](https://huggingface.co/) & Access Token

### 1. Clone & Configure Environment
```bash
# Clone the repository (if using git)
git clone <repository-url>
cd service_info_assistant

# Create a Virtual Environment
python -m venv .venv

# Activate Virtual Environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1


2. Install Dependencies
pip install -r requirements.txt


3. Configure API Keys
#Create a file named .env in the root directory and add your Hugging Face token:
HUGGINGFACEHUB_API_TOKEN=hf_YourGeneratedTokenHere

4. Initialize the Knowledge Base
Run the ingestion and embedding scripts to build the vector database:

# Step 1: Process documents
python ingestion.py

# Step 2: Generate embeddings and save to ChromaDB
python embeddings.py

🏃‍♂️ Usage
Option A: Run the API (Recommended)
Start the FastAPI server:
python api.py

The server will start at http://127.0.0.1:8000.


🧠 Assumptions & Limitations
Assumptions
Data Structure: Assumes input data is in Markdown format located in a local data/ folder.
Pricing Model: The calculator tool assumes a fixed formula ($2000 setup + $1500/month). Real-world logic would be dynamic.
Single Turn: The current implementation treats every query as independent; it does not hold conversation history memory.
Limitations
Rate Limits: The Free Hugging Face Inference API has rate limits. Heavy usage may result in 503 errors.
Context Window: The chunk size (500 chars) is optimized for granular retrieval but may miss context in very long documents.
Agent Reliability: Small open-source models (7B parameters) can sometimes struggle with strict JSON formatting required for complex Agent tool use, though prompt engineering mitigates this.

🔮 Future Improvements
If given more time, the following improvements would be prioritized:
Production Database: Migrate from local ChromaDB to a cloud vector store like Pinecone or Weaviate for scalability.
Conversational Memory: Add ConversationBufferMemory to the agent so it can answer follow-up questions ("How much is that in Euros?").
Robust Error Handling: Implement "Retries" with exponential backoff for API calls to handle network flakiness.
Dockerization: Containerize the application using Docker to ensure consistent environments across development and production.
Advanced Tools: Integrate a Calendar API (like Google Calendar) to allow the agent to book consultation appointments directly.


Project Structure

service_info_assistant/
├── data/               # Source Markdown documents
├── chroma_db/          # Persisted Vector Database
├── .env                # API Keys (GitIgnored)
├── ingestion.py        # Document loading logic
├── embeddings.py       # Vector generation logic
├── rag_pipeline.py     # RAG Chain definition
├── agent.py            # Agent & Tool definition
├── api.py              # FastAPI Server
├── config.py           # Configuration variables
└── requirements.txt    # Python dependencies