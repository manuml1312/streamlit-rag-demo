import streamlit as st
from llama_index import VectorStoreIndex, SimpleDirectoryReader, Document
from PyPDF2 import PdfFileReader  # Fixed the typo in PyPDF2
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index import ServiceContext
from llama_index.llms import OpenAI
import openai

# Set OpenAI API key from Streamlit Secrets Manager
openai.api_key = st.secrets.openai_key

st.title("📝 Covestro Material Guide Chatbot ")

if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Mention your queries!"}
    ]

llm = OpenAI(model="gpt-3.5-turbo", temperature=0.3, system_prompt="""You are a chatbot to help users select materials.Answer
there queries about the materials and its uses from the document supplied.Keep the answers technical and in detail dont summarise. Keep your answers accurate and based on 
                   facts – do not hallucinate features.""")

# File uploader for PDF
pdf_file = st.file_uploader("Upload PDF Document", type=["pdf", "txt"])
pdf_document = PdfFileReader(pdf_file)
pdf_content = ""
for page_num in range(pdf_document.numPages):  # Fixed the attribute name
    page = pdf_document.getPage(page_num)  # Fixed the attribute name
    pdf_content += page.extract_text()

service_context = ServiceContext.from_defaults(llm=llm)
index = VectorStoreIndex.from_documents(pdf_content, service_context=service_context)

if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

# If prompt is provided, save it to chat history
prompt = st.text_input("How can I help you today?", placeholder="Your query here", key="user_input")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

# If the last message is not from the assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.spinner("Thinking..."):
        response = st.session_state.chat_engine.chat(prompt)
        st.write(response.response)
        message = {"role": "assistant", "content": response.response}
        st.session_state.messages.append(message)
