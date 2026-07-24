import io
from typing import List
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config

def load_and_split_pdf(uploaded_file) -> List[Document]:
    """Reads an uploaded PDF file and returns split document chunks."""
    pdf_stream = io.BytesIO(uploaded_file.read())
    reader = PdfReader(pdf_stream)
    
    documents = []
    for page_number, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={"page": page_number + 1},
                )
            )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    
    return text_splitter.split_documents(documents)
