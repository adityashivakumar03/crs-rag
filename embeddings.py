# Directory Structure:
# - embeddings.py: Handles embedding creation and vector storage
# - app.py: Main application with OpenAI integration and query handling

# embeddings.py
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

#os.makedirs("/Users/hariprasaathvv/Downloads/Good AI/Vercel RAG/Ne")
os.environ["OPENAI_API_KEY"]= ""

def load_and_embed_pdfs(pdf_folder: str, embedding_model: str, faiss_index_file: str):
    """
    Load PDFs, create embeddings, and store them in a FAISS index.

    Args:
        pdf_folder (str): Path to folder containing PDFs.
        embedding_model (str): OpenAI embedding model (e.g., "text-embedding-ada-002").
        faiss_index_file (str): Path to save the FAISS index.
    """
    pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    documents = []

    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model=embedding_model)
    vector_store = FAISS.from_documents(texts, embeddings)

    vector_store.save_local(faiss_index_file)
    print(f"FAISS index saved to {faiss_index_file}")

OPENAI_API_KEY= " "
# Usage:
load_and_embed_pdfs("/Users/hariprasaathvv/Downloads/Good AI/Vercel RAG/Doc", "text-embedding-ada-002", "faiss_index")