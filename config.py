"""Central configuration. Every value can be overridden with an environment variable."""
import os

# Local LLM served by Ollama
LLM_MODEL = os.getenv("LLM_MODEL", "mistral")

# Sentence-embedding model used for the FAISS index
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Retrieval
TOP_K = int(os.getenv("TOP_K", "10"))

# Vector store location
DB_DIR = os.getenv("DB_DIR", "db")
FAISS_INDEX_PATH = os.path.join(DB_DIR, "faiss_index")

# How much of the document each feature is allowed to read (characters)
SUMMARY_CHAR_LIMIT = 100_000
QUIZ_CHAR_LIMIT = 20_000
QUESTIONS_CHAR_LIMIT = 15_000

NO_ANSWER_MESSAGE = (
    "The uploaded documents do not contain sufficient information "
    "to answer this question."
)
