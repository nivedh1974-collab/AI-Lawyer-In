# ═══ 1. IMPORTS ═══════════════════════════════════════════════
import os
from langchain_community.document_loaders import PyPDFLoader            # reads a PDF, one page at a time
from langchain_text_splitters import RecursiveCharacterTextSplitter     # cuts text into chunks (plain code, no AI model)
from langchain_community.embeddings import HuggingFaceEmbeddings        # runs the MiniLM embedding model
from langchain_chroma import Chroma                                     # the vector database

# ═══ 2. CONFIG (the settings you can tune) ════════════════════
LAWS_FOLDER = "../data/laws"                                            # where the 14 Act PDFs are
CHROMA_DB_PATH = "../chroma_db"                                         # where the database gets saved
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"              # turns text into 384 numbers
CHUNK_SIZE = 800                                                        # max characters per chunk
CHUNK_OVERLAP = 100                                                     # characters shared between neighbouring chunks

# ═══ 3. STEP 1: LOAD THE PDFs ═════════════════════════════════
def load_pdfs(folder_path):
    documents = []
    for filename in os.listdir(folder_path):                            # go through every file in the folder
        if filename.endswith(".pdf"):                                   # skip anything that is not a PDF
            print(f"Loading: {filename}")
            loader = PyPDFLoader(os.path.join(folder_path, filename))
            docs = loader.load()                                        # one "document" per page; also records the page number
            # Tag each doc with which law it came from
            for doc in docs:
                doc.metadata["law_name"] = filename.replace(".pdf", "") # Act name comes from the file name (this is what the sources panel shows)
            documents.extend(docs)                                      # add this PDF's pages to the full list
    print(f"\nTotal pages loaded: {len(documents)}")
    return documents

# ═══ 4. STEP 2: CUT INTO CHUNKS ═══════════════════════════════
def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,                                          # 800 characters
        chunk_overlap=CHUNK_OVERLAP,                                    # 100 characters repeated at the boundary, so a cut sentence isn't lost
        length_function=len                                             # count length in characters, not tokens
    )
    chunks = splitter.split_documents(documents)                        # tries paragraph breaks first, then lines, sentences, words
    print(f"Total chunks created: {len(chunks)}")                       # each chunk keeps its law_name and page
    return chunks

# ═══ 5. STEP 3: EMBED AND SAVE ════════════════════════════════
def store_in_chroma(chunks):
    print("\nLoading embedding model (first time = slow, downloads model)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)     # loads MiniLM (downloaded the first time)

    print("Storing chunks in Chroma vector DB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,                                               # every chunk is embedded into 384 numbers
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH                                # saved to disk, so the app can load it later
    )
    print(f"\nDone. {len(chunks)} chunks stored in {CHROMA_DB_PATH}")
    # Save chunks for BM25 retriever
    import pickle
    with open("../chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)                                          # a second copy as plain text chunks, for keyword search later
    print("Chunks saved for BM25 retriever.")
    return vectorstore

# ═══ 6. MAIN: RUN THE THREE STEPS IN ORDER ════════════════════
if __name__ == "__main__":                                              # runs only when you run this file directly
    print("=== INDIAN AI LAWYER — INGESTION PIPELINE ===\n")
    docs = load_pdfs(LAWS_FOLDER)                                       # PDFs → pages
    chunks = chunk_documents(docs)                                      # pages → chunks
    store_in_chroma(chunks)                                             # chunks → vectors → saved database
    print("\n=== INGESTION COMPLETE ===")
