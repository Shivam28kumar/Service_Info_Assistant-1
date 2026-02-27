import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
# We are importing the functions we just wrote in the previous step!
from ingestion import load_documents, chunk_documents, DATA_DIR

# Configuration
DB_DIR = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2" # This is a very fast, free Hugging Face model

def create_vector_db():
    print("1. Loading and chunking documents from data folder...")
    docs = load_documents(DATA_DIR)
    chunks = chunk_documents(docs)
    
    if not chunks:
        print("No chunks found. Exiting.")
        return None

    print(f"2. Initializing HuggingFace Embedding Model: '{EMBEDDING_MODEL}'...")
    print("(Note: This might take a minute the first time as it downloads the free model)")
    
    # Load the free Hugging Face embedding model
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("3. Converting text to numbers (embeddings) and saving to ChromaDB...")
    # Create the database, embed the chunks, and save it to the DB_DIR folder
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=DB_DIR
    )
    
    print(f"\n✅ SUCCESS! Vector database created and saved to the '{DB_DIR}' folder.")
    return vector_db

if __name__ == "__main__":
    create_vector_db()