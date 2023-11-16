import streamlit as st
import tempfile
from langchain.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
import os

OPENAI_API_KEY = st.secrets.openai_api

def get_pdf_text(pdf_docs):
    text = ""
    # Save the uploaded PDF to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(pdf_docs.read())
        temp_file_path = temp_file.name

    # Use the temporary file path with PDFMinerLoader
    pdf_reader = PDFMinerLoader(temp_file_path)
    text = pdf_reader.load()

    # Remove the temporary file
    os.remove(temp_file_path)

    return text

def get_text_chunks(text):
    text = str(text)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, request_timeout=120)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def get_conversational_chain(vector_store):
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)
    return conversation_chain

def user_input(user_question, conversation_dict):
    if conversation_dict is not None and conversation_dict['conversation_chain'] is not None:
        response = conversation_dict['conversation_chain'].chat({'question': user_question})
        conversation_dict['chat_history'].extend(response['chat_history'])
        st.session_state.conversation_dict = conversation_dict

def clear_chat():
    st.session_state.conversation_dict['chat_history'] = []
    st.session_state.conversation_dict['conversation_chain'] = get_conversational_chain(get_vector_store([]))
    st.session_state.conversation_dict = None

def main():
    st.set_page_config("Chat with Multiple PDFs")
    st.header("LLM Powered Chatbot")
    user_question = st.text_input("Ask a Question from the uploaded file")
    
    if "conversation_dict" not in st.session_state:
        st.session_state.conversation_dict = {
            'conversation_chain': get_conversational_chain(get_vector_store([])),
            'chat_history': [],
        }

    if user_question:
        user_input(user_question, st.session_state.conversation_dict)

    with st.sidebar:
        st.title("SoothsayerAnalytics")
        pdf_docs = st.file_uploader("Upload Files and Click on the Process Button", type=["pdf"])
        
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vector_store = get_vector_store(text_chunks)
                conversation_chain = get_conversational_chain(vector_store)
                
                st.session_state.conversation_dict = {
                    'conversation_chain': conversation_chain,
                    'chat_history': [],
                }
                
                st.success("Done")

        if st.button("Clear Chat"):
            clear_chat()

if __name__ == "__main__":
    main()
