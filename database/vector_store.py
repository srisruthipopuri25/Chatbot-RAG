from typing import List
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
import config

def create_vector_store(chunks: List[Document]) -> FAISS:
    """Creates a FAISS vectorstore from document chunks using Gemini embeddings."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        google_api_key=config.GOOGLE_API_KEY
    )
    return FAISS.from_documents(chunks, embeddings)
