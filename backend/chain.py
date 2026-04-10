from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from retriever import get_retriever
from dotenv import load_dotenv
import os

load_dotenv("../.env")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

system_prompt = """You are an expert Indian lawyer AI assistant.
Answer ONLY based on the legal documents provided as context.
Always cite the specific Act name and section number inline.
At the end of your answer, list ALL sources used under a heading called "Sources:" like this:
- BNS 2023, Section 103
- Constitution of India, Article 21

If the context does not support the answer, say:
"I cannot answer this from available law documents. Please consult a licensed advocate."

Context: {context}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_sources(docs):
    sources = []
    for doc in docs:
        law = doc.metadata.get("law_name", "Unknown")
        page = doc.metadata.get("page", "")
        sources.append(f"{law} (page {page})")
    return list(set(sources))

retriever = get_retriever()

rag_chain = (
    RunnableParallel({
        "context": retriever | format_docs,
        "input": RunnablePassthrough(),
        "docs": retriever
    })
    | RunnablePassthrough.assign(
        answer=prompt | llm | StrOutputParser()
    )
)

def ask(question):
    result = rag_chain.invoke(question)
    answer = result["answer"]
    sources = get_sources(result["docs"])
    
    sources_text = "\n\n---\n**📚 Source Documents:**\n" + "\n".join(f"- {s}" for s in sources)
    return answer + sources_text

if __name__ == "__main__":
    print("Testing RAG chain...\n")
    answer = ask("What is the punishment for murder?")
    print(f"Answer: {answer}")