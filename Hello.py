import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#import google.generativeai as palm
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
OPENAI_API_KEY= st.secrets.openai_api

with st.sidebar:
    pdf_docs = st.file_uploader("Upload Files and Click on the Process Button", accept_multiple_files=True)
    if st.button("Process"):
        with st.spinner("Processing"):
            text=""
            for pdf in pdf_docs:
                pdf_reader= PdfReader(pdf)
                for page in pdf_reader.pages:
                    text+= page.extract_text()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
chunks = text_splitter.split_text(text)

embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)
vector_store = FAISS.from_texts(chunks, embedding=embeddings)

llm = ChatOpenAI(openai_api_key = OPENAI_API_KEY)
memory = ConversationBufferMemory(memory_key = "chat_history", return_messages=True)
conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)


if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.chat_engine = llm.as_chat_engine(chat_mode="condense_question", verbose=True)

# if prompt := st.chat_input("Mention your requirements, based on which I can suggest materials",placeholder="Your processing queries here"): 

if prompt :=st.text_input("How can i help you with your material processing query?",placeholder="Your Question Here"):
    st.session_state.messages.append({"role": "user", "content": prompt})

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)
