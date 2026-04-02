"""
Core RAG retrieval chain.
Loaded by api.py on every request (chain is stateless).
"""

from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

CHROMA_PATH     = os.getenv("CHROMA_PATH", "./chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "mistral:7b-instruct-q4_K_M")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful fashion product assistant.
Use only the product information provided below to answer the question.
If no matching product is found, say "I couldn't find matching products."

Context:
{context}

Question: {question}
Answer:""",
)


def get_chain(gender: str | None = None, category: str | None = None) -> RetrievalQA:
    embeddings = HuggingFaceEmbeddings(
        model_name   = EMBEDDING_MODEL,
        model_kwargs = {"device": "cpu"},
        encode_kwargs= {"normalize_embeddings": True},
    )

    vectordb = Chroma(
        persist_directory = CHROMA_PATH,
        embedding_function= embeddings,
        collection_name   = "fashion_products",
    )

    filter_dict: dict = {}
    if gender:
        filter_dict["gender"] = gender
    if category:
        filter_dict["masterCategory"] = category

    retriever = vectordb.as_retriever(
        search_type  = "mmr",
        search_kwargs= {
            "k"     : 5,
            "fetch_k": 20,
            "filter": filter_dict or None,
        },
    )

    llm = Ollama(
        base_url   = OLLAMA_BASE_URL,
        model      = OLLAMA_MODEL,
        temperature= 0.1,
        num_ctx    = 2048,   # lower to 1024 if responses feel slow
    )

    return RetrievalQA.from_chain_type(
        llm               = llm,
        chain_type        = "stuff",
        retriever         = retriever,
        chain_type_kwargs = {"prompt": PROMPT},
        return_source_documents=True,
    )
