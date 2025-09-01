from langchain_huggingface import HuggingFaceChat
from dotenv import load_dotenv
import streamlit as st
import os

# Load .env variables
load_dotenv()

# Get your HF API key from environment
api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Initialize DeepSeek as a Chat model
llm = HuggingFaceChat(
    repo_id="deepseek-ai/DeepSeek-V3.1",
    huggingfacehub_api_token=api_key,
    model_kwargs={
        "temperature": 0.7,
        "max_new_tokens": 512
    }
)

st.header("Research Tool")
user_input = st.text_input("Enter your prompt here")

if st.button("Summarize"):
    result = llm.invoke(user_input)
    st.write(result)




