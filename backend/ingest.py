import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ── CONFIG ──────────────────────────────────────────
LAWS_FOLDER = "../data/laws"
CHROMA_DB_PATH = "../chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# ── STEP 1: LOAD PDFs ────────────────────────────────
def load_pdfs(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            print(f"Loading: {filename}")
            loader = PyPDFLoader(os.path.join(folder_path, filename))
            docs = loader.load()
            # Tag each doc with which law it came from
            for doc in docs:
                doc.metadata["law_name"] = filename.replace(".pdf", "")
            documents.extend(docs)
    print(f"\nTotal pages loaded: {len(documents)}")
    return documents

# ── STEP 2: CHUNK DOCUMENTS ──────────────────────────
def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    chunks = splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")
    return chunks

# ── STEP 3: EMBED + STORE ────────────────────────────
def store_in_chroma(chunks):
    print("\nLoading embedding model (first time = slow, downloads model)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    print("Storing chunks in Chroma vector DB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH
    )
    print(f"\nDone. {len(chunks)} chunks stored in {CHROMA_DB_PATH}")
    # Save chunks for BM25 retriever
    import pickle
    with open("../chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print("Chunks saved for BM25 retriever.")
    return vectorstore

# ── MAIN ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=== INDIAN AI LAWYER — INGESTION PIPELINE ===\n")
    docs = load_pdfs(LAWS_FOLDER)
    chunks = chunk_documents(docs)
    store_in_chroma(chunks)
    print("\n=== INGESTION COMPLETE ===")