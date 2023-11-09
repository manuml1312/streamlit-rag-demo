import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

openai.api_key= st.secrets.openai_api

def text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=20)
    chunks = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings(openai_api_key = openai.api_key)
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    return vector_store

def model(vector_store):
    llm = ChatOpenAI(openai_api_key = openai.api_key)
    memory = ConversationBufferMemory(memory_key = "chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)
    return conversation_chain

def user_input(u_q):
    response = st.session_state.conversation({'question': u_q})
    st.session_state.chatHistory = response['chat_history']
    for i, message in enumerate(st.session_state.chatHistory):
        if i%2 == 0:
            st.write("Human: ", message.content)
        else:
            st.write("Bot: ", message.content)


st.set_page_config("Retrieve Info")
st.header("Covestro LLM Powered Chatbot")

# if "conversation" not in st.session_state:
#     st.session_state.conversation.append({"role":"user","content":
# if "chatHistory" not in st.session_state:
#     st.session_state.chatHistory = None

with st.sidebar:
    st.title("SoothsayerAnalytics")
    pdf_docs = st.file_uploader("Upload Files and Click on the Process Button", accept_multiple_files=True)
    if st.button("Process"):
        with st.spinner("Processing"):
            vector_store = text(pdf_docs)

if not pdf_docs:
    st.write("Upload a document before accessing the chatbot")

if user_question :=st.text_input("Learn about our Products and Materials",placeholder="Enter your query here"):
    st.session_state.conversation.append({"role": "user", "content": user_question})
# user_question = st.text_input("Learn about our Products and Materials",placeholder="Enter your query here")

if user_question:
    user_input(user_question)
with st.spinner("Thinking"):
    st.session_state.conversation = model(vector_store)
    st.success("Done")
