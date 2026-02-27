import streamlit as st
from agent import setup_agentic_pipeline

# 1. Page Configuration
st.set_page_config(page_title="TechGadgets AI", page_icon="🤖", layout="centered")
st.title("🤖 TechGadgets Service Assistant")
st.write("Ask me about our AI services, FAQs, or get a custom digital marketing quote!")

# 2. Load the Agent (Cached)
# @st.cache_resource prevents the AI/Database from reloading every time you type a message
@st.cache_resource
def load_agent():
    return setup_agentic_pipeline()

agent_executor = load_agent()

# 3. Initialize Chat History
# Streamlit re-runs the script on every click, so we save history in 'session_state'
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am the TechGadgets AI Assistant. How can I help you today?"}
    ]

# 4. Display Previous Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 5. The Chat Input Box
if prompt := st.chat_input("Ask about services or pricing..."):
    
    # Show user message instantly
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and show AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking... searching documents and running calculators..."):
            try:
                # Send the user prompt to the exact same agent you tested in the terminal
                response = agent_executor.invoke({"input": prompt})
                answer = response["output"]
                
                # Display the answer and save it to history
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"Something went wrong: {e}")