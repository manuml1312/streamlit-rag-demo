import streamlit as st
from llama_index import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index import ServiceContext
from llama_index.llms import OpenAI
import openai
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
pdf_file = st.file_uploader("Upload PDF Document", type=["pdf","txt"])

if pdf_file:
    # Read the content of the uploaded PDF in binary mode
    with pdf_file as file:
        pdf_content = file.read()
    document = Document(text=pdf_content, filename=pdf_file.name)
    documents = [document]
else:
    documents = []

service_context = ServiceContext.from_defaults(llm=llm)
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

# If prompt is provided, save it to chat history
prompt = st.text_input("How can I help you today?", placeholder="Your query here", disabled=not documents)
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

# If the last message is not from the assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.spinner("Thinking..."):
        response = st.session_state.chat_engine.chat(prompt)
        st.write(response.response)
        message = {"role": "assistant", "content": response.response}
        st.session_state.messages.append(message)
