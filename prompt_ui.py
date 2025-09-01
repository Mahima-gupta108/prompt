from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
load_dotenv()
api_key = "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" 
llm=HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.1",
    task="text-generation",
    huggingfacehub_api_token=api_key 
    )
model=ChatHuggingFace(llm=llm)
st.header("research tool")
user_input=st.text_input("enter your prompt here")
if st.button('summarize'):
    result=model.invoke(user_input)
    st.write(result.content)
