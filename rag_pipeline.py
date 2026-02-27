import os
from dotenv import load_dotenv

# 1. Integration Packages
# 1. Integration Packages
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
# from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_chroma import Chroma

# 2. Core Components
from langchain_core.prompts import PromptTemplate

# 3. Modern Chains (Now located in langchain_classic in v1.x)
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Load environment variables
load_dotenv()

# Configuration
DB_DIR = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "HuggingFaceH4/zephyr-7b-beta"

def setup_rag_pipeline():
    # --- Check for API Key ---
    sec_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not sec_key:
        print("❌ CRITICAL ERROR: HUGGINGFACEHUB_API_TOKEN not found in .env file.")
        print("Please add it to your .env file.")
        return None

    print("1. Loading AI Memory (ChromaDB)...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        print(f"Error loading ChromaDB: {e}")
        return None

    print(f"2. Connecting to Cloud LLM ({LLM_MODEL})...")
    try:
        # Step A: Initialize the endpoint with the 'conversational' task
        base_llm = HuggingFaceEndpoint(
            repo_id=LLM_MODEL,
            task="conversational",  # <-- This fixes the Novita provider error
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.03,
        )
        
        # Step B: Wrap it so LangChain correctly formats standard text into a chat structure
        llm = ChatHuggingFace(llm=base_llm)
        
    except Exception as e:
        print(f"Error connecting to Hugging Face: {e}")
        return None

    print("3. Creating Prompt Template...")
    # Instructions for the AI
    template = """
    You are a helpful assistant for 'TechGadgets'.
    Use the retrieved context below to answer the question accurately.
    If the answer is not in the context, say "I don't have information on that."

    Context:
    {context}

    Question: {input}
    Answer:
    """
    
    prompt = PromptTemplate(
        template=template, 
        input_variables=["context", "input"]
    )

    print("4. Building Retrieval Chain...")
    # Create the chain that injects documents into the prompt
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create the final retrieval chain that connects the retriever to the document chain
    qa_chain = create_retrieval_chain(retriever, combine_docs_chain)
    
    return qa_chain

if __name__ == "__main__":
    print("--- STARTING RAG PIPELINE ---")
    rag_chain = setup_rag_pipeline()
    
    if rag_chain:
        question = "What is the estimated cost for a 3-month digital marketing campaign?"
        print(f"\n❓ Question: {question}")
        print("⏳ Thinking...")
        
        try:
            # invoke() uses "input" as the key for modern retrieval chains
            response = rag_chain.invoke({"input": question})
            
            print("\n🤖 AI Answer:")
            print(response["answer"].strip())
            
            print("\n📄 Source:")
            if response.get("context"):
                source = response["context"][0]
                print(f"From: {source.metadata.get('source', 'Unknown')}")
        except Exception as e:
            print(f"Error running chain: {e}")