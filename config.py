import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

EMBEDDING_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "gemini-2.5-flash"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
