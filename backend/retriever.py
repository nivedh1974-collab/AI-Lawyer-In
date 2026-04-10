from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
import pickle
import os

CHROMA_DB_PATH = "../chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNKS_PATH = "../chunks.pkl"

def get_retriever():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings
    )
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    return vector_retriever