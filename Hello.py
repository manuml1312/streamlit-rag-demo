import streamlit as st
from langchain.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

# Replace 'YOUR_OPENAI_API_KEY' with your actual OpenAI API key
OPENAI_API_KEY = st.secrets.openai_api

# Configure Streamlit page
st.set_page_config(page_title="RAG Application", layout="wide")

def get_pdf_text(pdf_docs):
    text = ""
    if pdf_docs is not None:
        try:
            loader = PDFMinerLoader(pdf_docs)
            text = loader.load()
        except Exception as e:
            st.error(f"Error loading PDF: {e}")
    return text

def get_text_chunks(text):
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

def main():
    st.title("RAG Application with OpenAI")
    user_question = st.text_input("Ask a Question:")
    
    if st.button("Generate Response"):
        if user_question:
            with st.spinner("Processing..."):
                response = st.session_state.conversation({'question': user_question})
                st.session_state.chatHistory = response['chat_history']
                for message in st.session_state.chatHistory:
                    st.write(f"{message.role.capitalize()}: {message.content}")

    with st.sidebar:
        st.title("Document Input")
        pdf_docs = st.file_uploader("Upload PDF Document", type=["pdf"])
        
        if st.button("Process Documents"):
            if pdf_docs is not None:
                with st.spinner("Processing Documents..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vector_store = get_vector_store(text_chunks)
                    st.session_state.conversation = get_conversational_chain(vector_store)
                    st.success("Document Processing Complete")

if __name__ == "__main__":
    main()
