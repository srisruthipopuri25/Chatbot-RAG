import streamlit as st
import config
from preprocessing.pdf_loader import load_and_split_pdf
from database.vector_store import create_vector_store
from llm.rag_chain import build_rag_chain

st.set_page_config(page_title="Gemini RAG Demo", layout="wide")
st.title("📄 Gemini RAG Demo")

# Ensure API Key exists
if not config.GOOGLE_API_KEY:
    st.error("Google API Key not found. Please set `GOOGLE_API_KEY` in your `.env` file.")
    st.stop()

# File Uploader
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is None:
    st.info("Upload a PDF to begin.")
    st.stop()

# Processing Pipeline
with st.spinner("Reading and splitting PDF..."):
    chunks = load_and_split_pdf(uploaded_file)
    st.success(f"Created {len(chunks)} chunks.")

with st.spinner("Creating embeddings..."):
    vectorstore = create_vector_store(chunks)

qa_chain = build_rag_chain(vectorstore)

# User Query Interface
st.header("Ask Questions")
question = st.text_input("Enter your question")

if question:
    with st.spinner("Thinking..."):
        response = qa_chain.invoke({"input": question})

    st.subheader("Answer")
    st.write(response["answer"])

    st.subheader("Relevant Chunks")
    docs = vectorstore.similarity_search(question, k=2)

    for i, doc in enumerate(docs):
        st.markdown(f"### Chunk {i+1} (Page {doc.metadata.get('page')})")
        st.info(doc.page_content)
