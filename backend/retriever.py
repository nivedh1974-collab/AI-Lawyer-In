# ═══ 1. IMPORTS ═══════════════════════════════════════════════
from langchain_community.embeddings import HuggingFaceEmbeddings   # runs the MiniLM embedding model
from langchain_chroma import Chroma                                # the vector database
from langchain_community.retrievers import BM25Retriever          # keyword search tool (imported but never used)
import pickle                                                      # for reading chunks.pkl (never used)
import os                                                          # never used

# ═══ 2. CONFIG ════════════════════════════════════════════════
CHROMA_DB_PATH = "../chroma_db"                                    # where the saved database is
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"        # must be the same model ingest.py used
CHUNKS_PATH = "../chunks.pkl"                                      # chunk file for keyword search (never used)

# ═══ 3. BUILD THE RETRIEVER ═══════════════════════════════════
def get_retriever():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)  # loads MiniLM, so questions get embedded the same way as the chunks

    vectorstore = Chroma(
        persist_directory=CHROMA_DB_PATH,                           # opens the saved database; does not rebuild it
        embedding_function=embeddings                               # used to embed each incoming question
    )
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 8})   # "find the 8 closest chunks" search

    return vector_retriever                                         # returned to chain.py; plain vector search only
