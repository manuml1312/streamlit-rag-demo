import streamlit as st
from langchain.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
import tempfile
import os

OPENAI_API_KEY = st.secrets.openai_api

def get_pdf_text(pdf_docs):
    text = ""
    # Save the uploaded PDF file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(pdf_docs.read())
        temp_path = temp_file.name

    try:
        # Load the PDF using PDFMinerLoader
        pdf_reader = PDFMinerLoader(temp_path)
        text = pdf_reader.load()
    finally:
        # Remove the temporary file
        os.remove(temp_path)

    return text

def get_text_chunks(text):
    text = str(text)  # Ensure that text is a string
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, request_timeout=120)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

# def get_conversational_chain(vector_store):
#     llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
#     memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#     conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)
#     return conversation_chain

# def user_input(user_question):
#     response = st.session_state.conversation_chain.chat({'question': user_question})
#     st.session_state.chat_history = response['chat_history']
#     for i, message in enumerate(st.session_state.chat_history):
#         role = "Human" if i % 2 == 0 else "Bot"
#         st.write(f"{role}: {message.content}")

def get_conversational_chain(vector_store):
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)
    return conversation_chain

def user_input(user_question):
    conversation_chain = st.session_state.conversation_chain

    # Assuming there is a method like 'process' or 'predict'
    response = conversation_chain.process({'question': user_question})

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
    
    # Initialize session state if not present
    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = None
        st.session_state.chat_history = None

    # Check if conversation_chain is initialized and not None
    if st.session_state.conversation_chain is not None:
        if user_question:
            user_input(user_question)

    with st.sidebar:
        st.title("SoothsayerAnalytics")
        pdf_docs = st.file_uploader("Upload Files and Click on the Process Button", type=["pdf"])
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vector_store = get_vector_store(text_chunks)
                st.session_state.conversation_chain = get_conversational_chain(vector_store)
                st.success("Done")



if __name__ == "__main__":
    main()
