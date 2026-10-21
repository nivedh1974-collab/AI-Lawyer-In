

```python
# ═══ 1. IMPORTS ═══════════════════════════════════════════════
import streamlit as st                                              # the web interface
import os                                                           # reads environment variables (API key, model name)
import sys                                                          # lets us add /app to Python's search path
import pickle                                                       # reads chunks.pkl (needed for keyword search)
sys.path.append("/app")                                             # so Python can find files in the Space's /app folder

from langchain_community.embeddings import HuggingFaceEmbeddings    # runs the MiniLM embedding model
from langchain_chroma import Chroma                                 # the vector database
from langchain_groq import ChatGroq                                 # client for the Groq-hosted LLM
from langchain_core.prompts import ChatPromptTemplate               # builds the prompt with rules + context + question
from langchain_core.output_parsers import StrOutputParser           # turns the model reply into plain text
from langchain_community.retrievers import BM25Retriever            # keyword search tool (NEW: now actually used)

# ═══ 2. CONFIG ════════════════════════════════════════════════
CHROMA_DB_PATH = "/app/chroma_db"                                   # where the saved vector database sits in the Space
CHUNKS_PATH = "/app/chunks.pkl"                                     # saved chunks, used to build the keyword index
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"          # must be the same model ingest.py used
REFUSAL = "I cannot answer this from available law documents. Please consult a licensed advocate."  # the fixed refusal sentence

# Retrieval settings. Change these to test different setups.
USE_HYBRID = True                                                   # False = vector-only, the old behaviour (your rollback switch)
VECTOR_K = 8                                                        # chunks returned by vector (meaning) search
BM25_K = 8                                                          # chunks returned by keyword search
FINAL_K = 10                                                        # chunks sent to the model after merging both lists

# ═══ 3. THE PROMPT (the rules the model follows) ══════════════
SYSTEM_PROMPT = """You are an expert Indian lawyer AI assistant. Answer ONLY from the context below. Do not add facts from memory.

Use the new criminal laws: BNS (not IPC), BNSS (not CrPC), BSA (not Indian Evidence Act). Never cite IPC, CrPC or Evidence Act sections unless the context itself mentions them as the old equivalent.

If the context supports an answer, format it like this:
**Short answer:** 1-2 plain sentences.
**Law:** Act name and section number, with the key wording from the context.
**Related provisions:** other sections in the context that matter (definition, exceptions, aggravated forms). Skip if none.
**Procedure:** cognizable/bailable/trial court, only if the context says so. Skip if none.
**Note:** one line saying this is general information, not legal advice.

If the context does not support an answer, reply with exactly this sentence and nothing else, with no headings and no note:
I cannot answer this from available law documents. Please consult a licensed advocate.

Context:
{context}"""

# ═══ 4. MERGE FUNCTION (NEW) ══════════════════════════════════
def merge_results(vector_docs, bm25_docs, limit):
    """Reciprocal rank fusion: a chunk ranked high in either list scores high."""
    scores = {}                                                     # chunk -> combined score
    docs_by_key = {}                                                # chunk -> the actual document
    for docs in (vector_docs, bm25_docs):                           # go through both result lists
        for rank, doc in enumerate(docs):                           # rank 0 = best match in that list
            key = (doc.metadata.get("law_name"), doc.metadata.get("page"), doc.page_content[:100])  # identifies a chunk, so duplicates merge
            scores[key] = scores.get(key, 0) + 1 / (60 + rank + 1)  # higher rank = bigger score; 60 is the standard smoothing constant
            docs_by_key[key] = doc                                  # remember the document for this key
    best = sorted(scores, key=scores.get, reverse=True)[:limit]     # sort by score, keep the top few
    return [docs_by_key[k] for k in best]                           # hand back the documents, best first

# ═══ 5. LOAD EVERYTHING ONCE ══════════════════════════════════
@st.cache_resource                                                  # runs once, then reuses the result (loading is slow)
def load_chain():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)  # loads MiniLM, so questions are embedded like the chunks were
    vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)  # opens the saved database; does not rebuild it
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": VECTOR_K})  # "find the closest chunks by meaning"
    print("DOCS IN DB:", len(vectorstore.get()["ids"]), flush=True) # shows chunk count in the Space logs

    bm25_retriever = None                                           # stays None if keyword search is off or the file is missing
    if USE_HYBRID and os.path.exists(CHUNKS_PATH):                  # only build it if hybrid is on AND chunks.pkl exists
        with open(CHUNKS_PATH, "rb") as f:                          # open the saved chunks file
            chunks = pickle.load(f)                                 # read the chunk list back into memory
        bm25_retriever = BM25Retriever.from_documents(chunks)       # builds the keyword index over all chunks
        bm25_retriever.k = BM25_K                                   # how many keyword matches to return
        print("BM25 READY, chunks:", len(chunks), flush=True)       # confirms it loaded; compare with DOCS IN DB
    else:
        print("BM25 OFF: vector-only mode", flush=True)             # tells you in the logs that it fell back

    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),       # the model that writes the answer (can be changed with an env variable)
        api_key=os.getenv("GROQ_API_KEY"),                          # your Groq key, stored as a Space secret
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),                                  # the rules plus the retrieved context
        ("human", "{input}"),                                       # the user's question
    ])
    chain = prompt | llm | StrOutputParser()                        # prompt -> model -> plain text
    return chain, vector_retriever, bm25_retriever                  # everything the app needs

# ═══ 6. RETRIEVE (NEW) ════════════════════════════════════════
def retrieve(question, vector_retriever, bm25_retriever):
    vector_docs = vector_retriever.invoke(question)                 # search by meaning
    if bm25_retriever is None:                                      # keyword search is off
        return vector_docs                                          # so just use the vector results
    bm25_docs = bm25_retriever.invoke(question)                     # search by exact words
    return merge_results(vector_docs, bm25_docs, FINAL_K)           # combine both lists into one

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)           # joins the chunk texts into one block for the prompt

# ═══ 7. PAGE SETUP ════════════════════════════════════════════
st.set_page_config(page_title="Indian AI Lawyer", page_icon="⚖", layout="centered")  # browser tab title and layout
st.title("⚖ Indian AI Lawyer")                                      # heading on the page
st.caption(
    "Answers come from 14 Indian Acts, including BNS, BNSS, BSA and the Constitution. "
    "General information, not legal advice. Please don't enter personal details."
)

if "messages" not in st.session_state:                              # first visit in this session
    st.session_state.messages = []                                  # start with an empty chat history

for msg in st.session_state.messages:                               # redraw earlier messages on every rerun
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ═══ 8. HANDLE A QUESTION ═════════════════════════════════════
if question := st.chat_input("Ask a legal question..."):            # runs when the user submits a question
    st.session_state.messages.append({"role": "user", "content": question})  # save the question in history
    with st.chat_message("user"):
        st.write(question)                                          # show the question

    with st.chat_message("assistant"):
        with st.spinner("Searching law documents..."):              # shows a spinner while we work
            chain, vector_retriever, bm25_retriever = load_chain()  # gets the cached chain and retrievers
            docs = retrieve(question, vector_retriever, bm25_retriever)  # finds the chunks (now hybrid)
            answer = chain.invoke({"context": format_docs(docs), "input": question})  # model writes the answer

        # Keep the refusal clean: no headings, no note.
        if "cannot answer this from available law documents" in answer.lower():  # model refused
            answer = REFUSAL                                        # swap in the exact clean sentence

        st.write(answer)                                            # show the answer
        with st.expander(f"Sources checked ({len(docs)})"):         # collapsible panel listing the chunks used
            for d in docs:
                page = d.metadata.get("page")                       # page number stored during ingestion
                page_label = f"p. {page + 1}" if isinstance(page, int) else ""  # pages start at 0, so add 1
                st.markdown(f"**{d.metadata.get('law_name', 'Unknown Act')}** {page_label}")  # Act name and page
                st.caption(d.page_content[:300])                    # first 300 characters of the chunk

    st.session_state.messages.append({"role": "assistant", "content": answer})  # save the answer in history
```
