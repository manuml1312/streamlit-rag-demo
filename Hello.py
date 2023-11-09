import streamlit as st
from pathlib import Path
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS,Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

OPENAI_API_KEY= st.secrets.openai_api

LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath('data', 'vector_store')

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)
    retriever = FAISS.from_texts(text_chunks, embedding=embeddings)
    return retriever

def query_llm(retriever,query):
    memory = ConversationBufferMemory(memory_key = "chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(llm=ChatOpenAI(openai_api_key = OPENAI_API_KEY), 
                                                               retriever=retriever.as_retriever(), memory=memory)
    result=qa_chain({"question":query,"chat_history":st.session_state.messages})
    result=result['answer']
    st.session_state.messages.append((query,result))
    return result

# def process_documents():
#     for source_doc in st.session_state.source_docs:
        
    
# def input_fields(): 
    # with st.sidebar:
        # st.session_state.source_docs=st.file_uploader(label="Upload the relevant documents for querying",type="pdf",accept_multiple_files=True)

def boot():
    with st.sidebar:
        pdf_docs=st.file_uploader(label="Upload the relevant documents for querying",type="pdf",accept_multiple_files=True)
        st.button("Process"):
            st.thinker("Processing"):
                text= get_pdf_text(pdf_docs)
                chunks=get_text_chunks(text)
                st.session_state.retriever=get_vector_store(chunks)
    if "messages" not in st.session_state:
        st.session_state.messages=[]
    # for message in st.session_state.messages:
    #     st.chat_message('human').write(message[0])
    #     st.chat_message('ai').write(message[1])
    if query:= st.chat_input():
        st.chat_message("human").write(query)
        response=query_llm(st.session_state.retriever,query)
        st.chat_message("ai").write(response)
    


if __name__ == "__main__":
    boot()
