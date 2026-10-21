# ═══ 1. IMPORTS ═══════════════════════════════════════════════
import streamlit as st                                          # the web UI library
import os
import sys

sys.path.append("/app")                                         # lets Python find files in the container's /app folder

from langchain_community.embeddings import HuggingFaceEmbeddings   # MiniLM embedding model
from langchain_chroma import Chroma                                # vector database
from langchain_groq import ChatGroq                                # Groq model client
from langchain_core.prompts import ChatPromptTemplate              # prompt with blanks
from langchain_core.output_parsers import StrOutputParser          # model reply → plain text

# ═══ 2. SETTINGS ══════════════════════════════════════════════
CHROMA_DB_PATH = "/app/chroma_db"                               # database location inside the Docker container
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"      # same model ingest.py used
REFUSAL = "I cannot answer this from available law documents. Please consult a licensed advocate."   # the fixed decline message

# ═══ 3. THE PROMPT ════════════════════════════════════════════
# Rule groups inside the prompt, in order:
#   a. "Answer ONLY from the context... Do not add facts from memory"   → grounding
#   b. "Use BNS (not IPC), BNSS (not CrPC), BSA..."                     → fixes the old-law citation problem
#   c. "Short answer / Law / Related provisions / Procedure / Note"      → the answer format
#   d. "reply with exactly this sentence and nothing else"               → clean refusal
#   e. "Context: {context}"                                              → blank where the 8 chunks go
SYSTEM_PROMPT = """You are an expert Indian lawyer AI assistant.
Answer ONLY from the context below. Do not add facts from memory.
Use the new criminal laws: BNS (not IPC), BNSS (not CrPC), BSA (not Indian Evidence Act).
Never cite IPC, CrPC or Evidence Act sections unless the context itself mentions them as the old equivalent.

If the context supports an answer, format it like this:
**Short answer:** 1-2 plain sentences.
**Law:** Act name and section number, with the key wording from the context.
**Related provisions:** other sections in the context that matter (definition, exceptions, aggravated forms). Skip if none.
**Procedure:** cognizable/bailable/trial court, only if the context says so. Skip if none.
**Note:** one line saying this is general information, not legal advice.

If the context does not support an answer, reply with exactly this sentence and nothing else, with no headings and no note: I cannot answer this from available law documents. Please consult a licensed advocate.

Context: {context}"""


# ═══ 4. LOAD EVERYTHING ONCE ══════════════════════════════════
@st.cache_resource                                              # runs once, then reuses the result, so the model isn't reloaded on every question
def load_chain():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)                      # load MiniLM
    vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)   # open the saved database
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})                        # search tool: top 8 chunks
    print("DOCS IN DB:", len(vectorstore.get()["ids"]), flush=True)                     # log line: shows chunk count (0 means the database is missing)

    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),   # model name from a Space variable, default GPT-OSS 120B
        api_key=os.getenv("GROQ_API_KEY"),                      # secret key stored in Space settings
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),                              # rules + chunks
        ("human", "{input}"),                                   # the user's question
    ])

    chain = prompt | llm | StrOutputParser()                    # pipeline: fill prompt → call model → get text
    return chain, retriever


# ═══ 5. HELPER: CHUNKS → TEXT ═════════════════════════════════
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)       # joins the 8 chunks into one block


# ═══ 6. PAGE HEADER ═══════════════════════════════════════════
st.set_page_config(page_title="Indian AI Lawyer", page_icon="⚖️", layout="centered")
st.title("⚖️ Indian AI Lawyer")
st.caption(                                                     # the small disclaimer under the title
    "Answers come from 14 Indian Acts, including BNS, BNSS, BSA and the Constitution. "
    "General information, not legal advice. Please don't enter personal details."
)

# ═══ 7. CHAT HISTORY ══════════════════════════════════════════
if "messages" not in st.session_state:                          # first visit: start an empty history
    st.session_state.messages = []

for msg in st.session_state.messages:                           # redraw earlier messages on each page refresh
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ═══ 8. HANDLE A NEW QUESTION ═════════════════════════════════
if question := st.chat_input("Ask a legal question..."):        # runs when the user submits
    st.session_state.messages.append({"role": "user", "content": question})   # save the question
    with st.chat_message("user"):
        st.write(question)                                      # show it
    with st.chat_message("assistant"):
        with st.spinner("Searching law documents..."):          # loading indicator
            chain, retriever = load_chain()                     # get the pipeline (cached after first call)

            docs = retriever.invoke(question)                   # SEARCH: top 8 chunks (the retrieval step)
            answer = chain.invoke({"context": format_docs(docs), "input": question})   # GENERATE: model answers from those chunks

            # Keep the refusal clean: no headings, no note.
            if "cannot answer this from available law documents" in answer.lower():    # catches any wrapped refusal
                answer = REFUSAL                                # replace with the exact clean sentence

            st.write(answer)                                    # show the answer

            with st.expander(f"Sources checked ({len(docs)})"):    # collapsible sources panel
                for d in docs:
                    page = d.metadata.get("page")
                    page_label = f"p. {page + 1}" if isinstance(page, int) else ""   # page starts at 0, so add 1
                    st.markdown(f"**{d.metadata.get('law_name', 'Unknown Act')}** {page_label}")   # Act name + page
                    st.caption(d.page_content[:300])            # first 300 characters of the chunk

    st.session_state.messages.append({"role": "assistant", "content": answer})   # save the answer to history
