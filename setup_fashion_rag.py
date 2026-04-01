"""
Fashion RAG Project Setup Script
Run: python setup_fashion_rag.py
Creates the full project structure with all boilerplate files.
"""

import os

BASE = "fashion_rag"

# ── file contents ──────────────────────────────────────────────────────────────

FILES = {

# ── .env ──────────────────────────────────────────────────────────────────────
".env": """\
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key_here
LANGCHAIN_PROJECT=fashion-rag

REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

CHROMA_PATH=./chroma_db
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
OLLAMA_MODEL=mistral:7b-instruct-q4_K_M
OLLAMA_BASE_URL=http://localhost:11434
""",

# ── requirements.txt ──────────────────────────────────────────────────────────
"requirements.txt": """\
langchain==0.2.16
langchain-community==0.2.16
chromadb==0.5.3
sentence-transformers==3.0.1
pandas==2.2.2
fastapi==0.112.0
uvicorn==0.30.5
streamlit==1.37.0
redis==5.0.8
python-dotenv==1.0.1
langsmith==0.1.98
pydantic==2.8.2
httpx==0.27.0
""",

# ── ingest.py ─────────────────────────────────────────────────────────────────
"ingest.py": """\
\"\"\"
Step 1 — Run once to embed the CSV and build the ChromaDB index.
Usage: python ingest.py --csv fashion.csv
\"\"\"

import argparse
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

load_dotenv()

CHROMA_PATH     = os.getenv("CHROMA_PATH", "./chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
BATCH_SIZE      = 64


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\\t")
    df = df.dropna(subset=["productDisplayName"])
    df["text"] = df.apply(
        lambda r: (
            f"{r['productDisplayName']}. "
            f"Category: {r['masterCategory']} > {r['subCategory']} > {r['articleType']}. "
            f"Colour: {r['baseColour']}. Gender: {r['gender']}. "
            f"Season: {r['season']}. Usage: {r['usage']}."
        ),
        axis=1,
    )
    return df


def build_vector_store(df: pd.DataFrame) -> None:
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    client     = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name="fashion_products",
        metadata={"hnsw:space": "cosine"},
    )

    meta_cols = ["gender", "masterCategory", "subCategory",
                 "articleType", "baseColour", "season", "usage"]

    for i in range(0, len(df), BATCH_SIZE):
        batch      = df.iloc[i : i + BATCH_SIZE]
        embeddings = model.encode(
            batch["text"].tolist(), show_progress_bar=False
        ).tolist()
        collection.add(
            ids        = [str(x) for x in batch["id"].tolist()],
            embeddings = embeddings,
            documents  = batch["text"].tolist(),
            metadatas  = batch[meta_cols].to_dict("records"),
        )
        print(f"  indexed {min(i + BATCH_SIZE, len(df))} / {len(df)}")

    print(f"Done. {len(df)} products stored in {CHROMA_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="fashion.csv", help="Path to TSV/CSV dataset")
    args = parser.parse_args()

    df = load_and_clean(args.csv)
    build_vector_store(df)
""",

# ── rag_chain.py ──────────────────────────────────────────────────────────────
"rag_chain.py": """\
\"\"\"
Core RAG retrieval chain.
Loaded by api.py on every request (chain is stateless).
\"\"\"

from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

load_dotenv()

CHROMA_PATH     = os.getenv("CHROMA_PATH", "./chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "mistral:7b-instruct-q4_K_M")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=\"\"\"You are a helpful fashion product assistant.
Use only the product information provided below to answer the question.
If no matching product is found, say "I couldn't find matching products."

Context:
{context}

Question: {question}
Answer:\"\"\",
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
""",

# ── api.py ────────────────────────────────────────────────────────────────────
"api.py": """\
\"\"\"
FastAPI server — production entry point.
Start: uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1

Keep --workers 1: multiple workers each load Mistral into RAM.
\"\"\"

import hashlib
import json
import time

import redis
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_chain import get_chain
import os

load_dotenv()

app = FastAPI(title="Fashion RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

cache = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=int(os.getenv("REDIS_DB", 0)),
    decode_responses=True,
)


class QueryRequest(BaseModel):
    question: str
    gender  : str | None = None
    category: str | None = None


class QueryResponse(BaseModel):
    answer    : str
    sources   : list[dict]
    cached    : bool
    latency_ms: float


@app.post("/query", response_model=QueryResponse)
async def query_fashion(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    cache_key = hashlib.md5(
        f"{req.question.lower().strip()}{req.gender}{req.category}".encode()
    ).hexdigest()

    # ── Cache hit ──────────────────────────────────────────────────────────
    cached_val = cache.get(cache_key)
    if cached_val:
        data = json.loads(cached_val)
        data["cached"]     = True
        data["latency_ms"] = 0.0
        return data

    # ── RAG retrieval ──────────────────────────────────────────────────────
    t0    = time.perf_counter()
    chain = get_chain(gender=req.gender, category=req.category)
    result= chain.invoke({"query": req.question})
    ms    = round((time.perf_counter() - t0) * 1000, 1)

    response = {
        "answer"    : result["result"],
        "sources"   : [doc.metadata for doc in result["source_documents"]],
        "cached"    : False,
        "latency_ms": ms,
    }

    cache.setex(cache_key, 3600, json.dumps(response))
    return response


@app.get("/health")
def health():
    return {"status": "ok"}


@app.delete("/cache")
def clear_cache():
    cache.flushdb()
    return {"message": "Cache cleared."}
""",

# ── app.py ────────────────────────────────────────────────────────────────────
"app.py": """\
\"\"\"
Streamlit chat UI.
Start: streamlit run app.py
\"\"\"

import requests
import streamlit as st

API_URL = "http://localhost:8000/query"

st.set_page_config(page_title="Fashion Assistant", page_icon="👗", layout="wide")
st.title("👗 Fashion Product Assistant")

# ── Sidebar filters ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    gender = st.selectbox(
        "Gender", ["Any", "Men", "Women", "Boys", "Girls", "Unisex"]
    )
    category = st.selectbox(
        "Category", ["Any", "Apparel", "Footwear", "Accessories", "Personal Care"]
    )
    st.divider()
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ── Chat history ───────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ── Input ──────────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about products, colours, styles…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching products…"):
            try:
                resp = requests.post(
                    API_URL,
                    json={
                        "question": prompt,
                        "gender"  : None if gender == "Any" else gender,
                        "category": None if category == "Any" else category,
                    },
                    timeout=120,
                )
                resp.raise_for_status()
                data   = resp.json()
                answer = data["answer"]
                st.write(answer)

                col1, col2 = st.columns(2)
                col1.caption(f"⏱ {data['latency_ms']} ms")
                col2.caption("⚡ cached" if data["cached"] else "")

                with st.expander("Source products"):
                    for src in data["sources"]:
                        st.json(src)

            except requests.exceptions.ConnectionError:
                answer = "Could not reach the API. Is `uvicorn api:app` running?"
                st.error(answer)
            except Exception as e:
                answer = f"Error: {e}"
                st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
""",

# ── README.md ─────────────────────────────────────────────────────────────────
"README.md": """\
# Fashion RAG System

Production-level Retrieval-Augmented Generation for fashion product search.
Optimised for CPU-only machines (i5 10th Gen, no GPU).

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) installed
- Redis running locally (`redis-server`)

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Pull the quantized LLM (once)
ollama pull mistral:7b-instruct-q4_K_M

# 3. Copy your dataset
cp /path/to/your/dataset.csv fashion.csv

# 4. Build the vector index (once)
python ingest.py --csv fashion.csv

# 5. Start Redis (separate terminal)
redis-server

# 6. Start the API (separate terminal)
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1

# 7. Start the UI (separate terminal)
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Performance tips (i5 CPU)

- Set `num_ctx=1024` in rag_chain.py for ~40% faster responses
- Set `k=3` in the retriever for less context to process
- Redis cache means repeated queries return instantly

## Project structure

```
fashion_rag/
├── .env              <- API keys and config
├── requirements.txt  <- Python dependencies
├── fashion.csv       <- Your dataset (add this yourself)
├── ingest.py         <- Run once: builds ChromaDB index
├── rag_chain.py      <- RAG retrieval logic
├── api.py            <- FastAPI REST server
├── app.py            <- Streamlit chat UI
└── chroma_db/        <- Auto-created by ingest.py
```
""",

}  # end FILES


# ── scaffold ───────────────────────────────────────────────────────────────────

def create_project():
    os.makedirs(BASE, exist_ok=True)
    for filename, content in FILES.items():
        filepath = os.path.join(BASE, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  created  {filepath}")

    # chroma_db placeholder so git doesn't ignore the dir
    chroma_dir = os.path.join(BASE, "chroma_db")
    os.makedirs(chroma_dir, exist_ok=True)
    with open(os.path.join(chroma_dir, ".gitkeep"), "w") as f:
        f.write("")
    print(f"  created  {chroma_dir}/.gitkeep")

    print(f"\nDone! Project created in ./{BASE}/")
    print("Next steps:")
    print(f"  cd {BASE}")
    print("  pip install -r requirements.txt")
    print("  ollama pull mistral:7b-instruct-q4_K_M")
    print("  python ingest.py --csv fashion.csv")


if __name__ == "__main__":
    create_project()
