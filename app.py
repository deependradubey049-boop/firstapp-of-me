import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai

# =========================
# LOAD ENV VARIABLES
# =========================
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("Gemini API key not found.")
    st.stop()

# =========================
# GEMINI CONFIG
# =========================
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# =========================
# STREAMLIT CONFIG
# =========================
st.set_page_config(
    page_title="MediQuery AI",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 MediQuery AI")
st.markdown(
    """
    Trusted Medical RAG Chatbot using Gemini + FAISS

    ⚠️ This chatbot is for informational purposes only and does not provide medical diagnosis.
    """
)

# =========================
# SIDEBAR
# =========================
st.sidebar.header("Upload Medical PDFs")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF Files",
    type=["pdf"],
    accept_multiple_files=True
)

# =========================
# EMBEDDING MODEL
# =========================
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =========================
# PROCESS DOCUMENTS
# =========================
def process_documents(files):
    documents = []

    for uploaded_file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name

    st.info("Please upload one or more medical PDF documents.")
