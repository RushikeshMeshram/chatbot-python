from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.messages import HumanMessage,AIMessage
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.title("🤖 Chatbot")

temperature = st.sidebar.slider(
    "Temperature",
    0.0,
    1.0,
    0.7
)

max_tokens = st.sidebar.slider(
    "Max tokens",
    50,
    1000,
    256
)

# Create a model endpoint
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    temperature=temperature,
    max_new_tokens=max_tokens,
    task="text-generation"
)

# Wrap it in ChatHuggingFace
chat_model = ChatHuggingFace(llm=llm)

if "messages" not in st.session_state:
    st.session_state.messages = [] 

for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    else:
        with st.chat_message("assistant"):
            st.write(message.content)

user_input = st.chat_input("Ask me anything")

if user_input:
    st.session_state.messages.append(
        HumanMessage(content=user_input)
    )

    try:
        response = chat_model.invoke(st.session_state.messages)
    except Exception as e:
        st.error(f"Something went wrong: {e}")

    response = chat_model.invoke(st.session_state.messages)

    # st.session_state.history.append({"role": "assistant", "content": response.content})
    st.session_state.messages.append(AIMessage(content=response.content))
    st.rerun()  # Rerun the app to display the updated chat history