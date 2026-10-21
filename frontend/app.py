Hugging Face's logo
Hugging Face
 
Models
Datasets
Spaces
Buckets
NEW
Docs
Pricing


 
Spaces:
nivedh1974
/
indian-ai-lawyer


Like
0

App

Files
Community
Settings
indian-ai-lawyer
/
app.py 
nivedh1974's picture
nivedh1974
Update app.py
458f98a
VERIFIED
1 day ago

Download

Raw
History
Blame
Edit
Delete

3.91 kB
import streamlit as st
import os
import sys

sys.path.append("/app")

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

CHROMA_DB_PATH = "/app/chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REFUSAL = "I cannot answer this from available law documents. Please consult a licensed advocate."

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


@st.cache_resource
def load_chain():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    print("DOCS IN DB:", len(vectorstore.get()["ids"]), flush=True)

    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        api_key=os.getenv("GROQ_API_KEY"),
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])

    chain = prompt | llm | StrOutputParser()
    return chain, retriever


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


st.set_page_config(page_title="Indian AI Lawyer", page_icon="⚖️", layout="centered")
st.title("⚖️ Indian AI Lawyer")
st.caption(
    "Answers come from 14 Indian Acts, including BNS, BNSS, BSA and the Constitution. "
    "General information, not legal advice. Please don't enter personal details."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if question := st.chat_input("Ask a legal question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching law documents..."):
            chain, retriever = load_chain()

            docs = retriever.invoke(question)
            answer = chain.invoke({"context": format_docs(docs), "input": question})

            # Keep the refusal clean: no headings, no note.
            if "cannot answer this from available law documents" in answer.lower():
                answer = REFUSAL

            st.write(answer)

            with st.expander(f"Sources checked ({len(docs)})"):
                for d in docs:
                    page = d.metadata.get("page")
                    page_label = f"p. {page + 1}" if isinstance(page, int) else ""
                    st.markdown(f"**{d.metadata.get('law_name', 'Unknown Act')}** {page_label}")
                    st.caption(d.page_content[:300])

    st.session_state.messages.append({"role": "assistant", "content": answer})
