import os
from dotenv import load_dotenv


# 1. Integration Packages
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_chroma import Chroma
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
# 2. Agent & Tool Packages
from langchain_core.tools import tool, create_retriever_tool
from langchain_classic.agents import AgentExecutor, create_structured_chat_agent
from langchain_classic import hub

# Load environment variables
load_dotenv()

# Configuration
DB_DIR = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# LLM_MODEL = "HuggingFaceH4/zephyr-7b-beta"
LLM_MODEL = "openai/gpt-oss-120b"

# ==========================================
# 🛠️ THE "HANDS": DEFINING OUR CUSTOM TOOLS
# ==========================================

@tool
def custom_pricing_calculator(months: int) -> str:
    """Use this tool to calculate a custom price estimate when a user asks for a specific campaign duration that is not 3 months. 
    Input should be an integer representing the number of months."""
    
    # Our fictional math logic
    base_monthly_rate = 2000
    setup_fee = 1000
    total = (base_monthly_rate * int(months)) + setup_fee
    
    return f"Tool Output: The calculated custom cost for {months} months is ${total} (includes a ${setup_fee} setup fee)."

# ==========================================

def setup_agentic_pipeline():
    sec_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not sec_key:
        print("❌ CRITICAL ERROR: HUGGINGFACEHUB_API_TOKEN missing.")
        return None

    print("1. Loading AI Memory (ChromaDB)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    print(f"2. Connecting to Cloud LLM ({LLM_MODEL})...")
    base_llm = HuggingFaceEndpoint(
        repo_id=LLM_MODEL,
        task="conversational",
        max_new_tokens=512,
        temperature=0.1, 
        repetition_penalty=1.03,
        # THE BRAKES: This forces the AI to stop talking and wait for the Python tool
        stop_sequences=["Observation:", "Observation:\n", "\nObservation", "}\n{"]  #stop_sequences=["Observation:", "}\n{"], the moment the LLM tries to hallucinate the tool's observation block, LangChain will instantly cut off its microphone, take the JSON action it generated, pass it to your actual Python calculator
    )
    llm = ChatHuggingFace(llm=base_llm)

    print("3. Equipping Tools...")
    # Tool 1: We convert our RAG pipeline into a tool
    retriever_tool = create_retriever_tool(
        retriever,
        name="techgadgets_service_search",
        description="Search internal documents for TechGadgets services, standard pricing, and FAQs. Use this to find out what AI services are offered."
    )
    
    # Group the tools together
    tools = [retriever_tool, custom_pricing_calculator]

    print("4. Building the Agent...")
    
    # 1. Pull the standard tool-calling prompt
    prompt = hub.pull("hwchase17/openai-tools-agent")
    
    # 2. OVERRIDE the System Message (The crucial part)
    # We explicitly forbid the model from using anything except OUR tools
    prompt.messages[0].prompt.template = (
        "You are a helpful assistant for TechGadgets. "
        "IMPORTANT: You have ONLY TWO tools: 'techgadgets_service_search' and 'custom_pricing_calculator'. "
        "DO NOT attempt to use 'repo_browser', 'web_search', or any other tool. "
        "If you don't know an answer, use 'techgadgets_service_search' to find it in the company documents."
    )
    
    # 3. Create the agent with this new strict prompt
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # Create the executor
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True
    )
    
    return agent_executor

if __name__ == "__main__":
    print("--- STARTING AGENT PIPELINE ---")
    agent_executor = setup_agentic_pipeline()
    
    if agent_executor:
        # We are asking a multi-part question that requires BOTH tools
        question = "What AI services do you offer, and how much would a custom 7-month digital marketing campaign cost?"
        print(f"\n❓ Question: {question}\n")
        
        try:
            # We use invoke with "input" just like our modern chain
            response = agent_executor.invoke({"input": question})
            print("\n🤖 Final Answer:")
            print(response["output"].strip())
        except Exception as e:
            print(f"Error running agent: {e}")