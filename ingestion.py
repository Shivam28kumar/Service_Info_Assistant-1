import os
from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Configuration
DATA_DIR = "data"
CHUNK_SIZE = 500      # How many characters per chunk
CHUNK_OVERLAP = 50    # Overlap to prevent cutting sentences in half

def load_documents(data_dir: str):
    """Loads markdown and text documents from the data directory."""
    documents = []
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Directory '{data_dir}' not found. Please create it and add documents.")
        return documents

    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        
        # Load Markdown files
        if filename.endswith(".md"):
            # We use TextLoader here for simplicity with markdown files
            loader = TextLoader(filepath, encoding='utf-8')
            documents.extend(loader.load())
            print(f"Loaded: {filename}")
            
        # Load TXT files
        elif filename.endswith(".txt"):
            loader = TextLoader(filepath, encoding='utf-8')
            documents.extend(loader.load())
            print(f"Loaded: {filename}")
            
    return documents

def chunk_documents(documents):
    """Splits documents into smaller semantic chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""] # Tries to split by paragraph first, then line, then word
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

if __name__ == "__main__":
    print("--- Starting Data Ingestion ---")
    docs = load_documents(DATA_DIR)
    
    if docs:
        chunks = chunk_documents(docs)
        print(f"\nSuccess! Loaded {len(docs)} document(s).")
        print(f"Split them into {len(chunks)} smaller chunks.")
        print("\n--- Sample of Chunk #1 ---")
        print(chunks[0].page_content)
        print("--------------------------")
    else:
        print("No documents were loaded. Please check your data folder.")