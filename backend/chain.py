# ═══ 1. IMPORTS ═══════════════════════════════════════════════
from langchain_groq import ChatGroq                       # client for the Groq-hosted model
from langchain_core.prompts import ChatPromptTemplate     # builds the prompt with blanks to fill
from langchain_core.output_parsers import StrOutputParser # turns the model reply into plain text
from langchain_core.runnables import RunnablePassthrough, RunnableParallel  # wiring pieces for the chain
from retriever import get_retriever                       # your own function: opens Chroma, returns top-8 search
from dotenv import load_dotenv                            # reads the .env file
import os

# ═══ 2. SETUP: KEY AND MODEL ══════════════════════════════════
load_dotenv("../.env")                                    # loads GROQ_API_KEY from the file one folder up

llm = ChatGroq(
    model="openai/gpt-oss-120b",                          # the model that writes the answer
    api_key=os.getenv("GROQ_API_KEY")                     # no temperature set, so the default 0.7 applies
)

# ═══ 3. THE PROMPT (the rules the model follows) ══════════════
system_prompt = """You are an expert Indian lawyer AI assistant.
Answer ONLY based on the legal documents provided as context.   # <- grounding rule: stops answers from memory
Always cite the specific Act name and section number inline.    # <- citation rule
At the end of your answer, list ALL sources used under a heading called "Sources:" like this:
- BNS 2023, Section 103
- Constitution of India, Article 21

If the context does not support the answer, say:                # <- refusal rule
"I cannot answer this from available law documents. Please consult a licensed advocate."

Context: {context}"""                                    # <- blank: the 8 chunks go here

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),                            # the rules plus the retrieved chunks
    ("human", "{input}")                                  # the user's question goes here
])

# ═══ 4. HELPER: TURN CHUNKS INTO TEXT ═════════════════════════
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)  # joins the 8 chunks into one block of text

# ═══ 5. HELPER: LIST THE SOURCES ══════════════════════════════
def get_sources(docs):
    sources = []
    for doc in docs:
        law = doc.metadata.get("law_name", "Unknown")     # Act name, tagged by ingest.py
        page = doc.metadata.get("page", "")               # page number (starts at 0)
        sources.append(f"{law} (page {page})")
    return list(set(sources))                             # removes duplicates, order is random

# ═══ 6. THE RAG CHAIN (the pipeline) ══════════════════════════
retriever = get_retriever()                               # opens the vector database once

rag_chain = (
    RunnableParallel({                                    # step A: prepare three things at once
        "context": retriever | format_docs,               #   search, then join chunks into text
        "input": RunnablePassthrough(),                   #   pass the question through unchanged
        "docs": retriever                                 #   search again, keeping the raw chunks for sources
    })
    | RunnablePassthrough.assign(                         # step B: add the answer, keep everything else
        answer=prompt | llm | StrOutputParser()           #   fill prompt, call the model, get plain text
    )
)

# ═══ 7. THE FUNCTION THE REST OF THE APP CALLS ════════════════
def ask(question):
    result = rag_chain.invoke(question)                   # run the whole pipeline
    answer = result["answer"]                             # the model's text
    sources = get_sources(result["docs"])                 # the Acts and pages it was given

    sources_text = "\n\n---\n**📚 Source Documents:**\n" + "\n".join(f"- {s}" for s in sources)
    return answer + sources_text                          # answer plus a source list added by the code

# ═══ 8. QUICK TEST ════════════════════════════════════════════
if __name__ == "__main__":                                # runs only if you run this file directly
    print("Testing RAG chain...\n")
    answer = ask("What is the punishment for murder?")
    print(f"Answer: {answer}")
