import streamlit as st
from langchain.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
os.environ['OPENAI_API_KEY']=st.secrets.openai_api
OPENAI_API_KEY = st.secrets.openai_api
import re
import streamlit as st
import tempfile
import os

def get_pdf_text(pdf_docs):
    text = ""
    if pdf_docs is not None:  # Ensure a file is uploaded
        try:
            # Save the uploaded file to a temporary location
            temp_dir = tempfile.TemporaryDirectory()
            temp_path = os.path.join(temp_dir.name, pdf_docs.name)
            with open(temp_path, "wb") as f:
                f.write(pdf_docs.read())

            # Load the PDF using PDFMinerLoader
            loader = PDFMinerLoader(temp_path)
            text = loader.load()
        except Exception as e:
            st.error(f"Error loading PDF: {e}")
        finally:
            temp_dir.cleanup()  # Clean up temporary directory

    return text

import re

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    
    # Ensure that _separator is a valid regex pattern
    _separator = text_splitter._separators
    if not _separator:
        st.error("Invalid separator pattern in RecursiveCharacterTextSplitter.")
        return []

    # Ensure that text is a string
    if not isinstance(text, (str, bytes)):
        st.error("Invalid text format. Expected string or bytes-like object.")
        return []

    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings.update_forward_refs()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def get_conversational_chain(vector_store):
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)
    return conversation_chain

def user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chatHistory = response['chat_history']
    for i, message in enumerate(st.session_state.chatHistory):
        if i % 2 == 0:
            st.write("Human: ", message.content)
        else:
            st.write("Bot: ", message.content)

def main():
    st.set_page_config("Chat with Multiple PDFs")
    st.header("LLM Powered Chatbot")
    user_question = st.text_input("Ask a Question from the uploaded file")
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = None
    
    if user_question:
        user_input(user_question)

    # Display "Clear Chat" button
    if st.button("Clear Chat"):
        st.session_state.chatHistory = None

    with st.sidebar:
        st.title("SoothsayerAnalytics")
        pdf_docs = st.file_uploader("Upload Files and Click on the Process Button", type=["pdf"])

        if st.button("Process"):
            if pdf_docs is not None:
                with st.spinner("Processing"):
                    raw_text = get_pdf_text(pdf_docs)
                    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, request_timeout=120)
                    text_chunks = get_text_chunks(raw_text)
                    vector_store = get_vector_store(text_chunks)
                    st.session_state.conversation = get_conversational_chain(vector_store)
                    st.success("Done")
            else:
                st.warning("Please upload a PDF file.")

if __name__ == "__main__":
    main()
