import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.llms import OpenAI
from PyPDF2 import PdfReader

# Set OpenAI API key from Streamlit Secrets Manager
openai.api_key = st.secrets.openai_key

st.title("📝 Covestro Material Guide Chatbot ")

# Initialize the chat messages history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Mention your queries!"}]

# Initialize OpenAI model
llm = OpenAI(
    model="gpt-3.5-turbo",
    temperature=0.3,
    system_prompt="""You are a chatbot to help users select materials.Answer
    their queries about the materials and its uses from the document supplied.
    Keep the answers technical and in detail; don't summarize. Keep your answers accurate and based on 
    facts – do not hallucinate features."""
)

# File uploader for PDF
pdf_file = st.file_uploader("Upload PDF Document", type=["pdf", "txt"])

if pdf_file:
    pdf_document = PdfReader(pdf_file)
    pdf_documents = []

    for page_num, page in enumerate(pdf_document.pages):
        page_text = page.extract_text()
        doc = {"doc_id": f"Page {page_num + 1}", "content": page_text}
        pdf_documents.append(doc)

    service_context = ServiceContext.from_defaults(llm=llm)
    index = VectorStoreIndex.from_documents(pdf_documents, service_context=service_context)

    # Initialize the chat engine
    if "chat_engine" not in st.session_state:
        st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

    # User input
    prompt = st.text_input("How can I help you today?", placeholder="Your query here", key="user_input")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate a new response if the last message is not from the assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)
