import streamlit as st 
import os
import openai
# from PyPDF2 import PdfReader
from llama_index import VectorStoreIndex, SimpleDirectoryReader , Document
from llama_index.embeddings import HuggingFaceEmbedding 
from llama_index import ServiceContext
from llama_index.llms import OpenAI

openai.api_key = st.secrets.openai_key #
# openai.api_key=os.envget("openai_key")  #st.secrets.openai_key

st.title("📝 RAG - Demo ")

with st.sidebar:
    uploaded_file=st.file_uploader("Upload the reference file to retrieve information",type=("pdf"))

if uploaded_file:
    if "messages" not in st.session_state.keys(): # Initialize the chat messages history
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask me a question about any info within the document!"}
        ]

    
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.3, system_prompt="""You are an expert on the uploaded document and your job is to answer 
              all questions. Assume that all questions are related to the document. Keep your answers accurate and based on 
                   facts retrieved from the document – do not hallucinate features.""")

    documents = SimpleDirectoryReader(input_files=uploaded_file).load_data()  #.read().decode()
    service_context = ServiceContext.from_defaults(llm=llm) 
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)

    if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

    if prompt :=st.text_input("Ask me any information from the document you uploaded",placeholder="Your query here",disabled=not uploaded_file):
        st.session_state.messages.append({"role": "user", "content": prompt})


# If last message is not from assistant, generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chat_engine.chat(prompt)
                st.write(response.response)
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message)
