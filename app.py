# app.py
from fastapi import FastAPI
from pydantic import BaseModel

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


app = FastAPI()

# Embedding function & vectordb
embedding_fn = OllamaEmbeddings(model="nomic-embed-text:latest")
vectordb = Chroma(
    persist_directory="vectordb",
    embedding_function=embedding_fn
)

retriever = vectordb.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# LLM (ChatOllama)
llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0.0
)

# Prompt template
prompt = ChatPromptTemplate.from_template(
    """Use the following context to answer the question. 
If the answer is not in the context, say 'I don't know.'

Context:
{context}

Question: {question}

Answer:"""
)

# Format documents function
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create RAG chain using LCEL
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)


class QueryIn(BaseModel):
    question: str


@app.post("/ask")
def ask(q: QueryIn):
    # Get answer
    answer = rag_chain.invoke(q.question)
    
    # Get source documents using invoke instead of get_relevant_documents
    source_docs = retriever.invoke(q.question)  # ← CORRECTION ICI
    
    sources = []
    for doc in source_docs:
        md = doc.metadata
        sources.append({
            "source": md.get("source", "Unknown"),
            "page": md.get("page", "N/A")
        })

    return {"answer": answer, "sources": sources}