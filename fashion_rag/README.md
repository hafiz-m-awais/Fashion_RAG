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
cp D:\Side_Projects\Fashion_RAG\fashion_rag\clothes.csv

# 4. Build the vector index (once)
python ingest.py --csv clothes.csv

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
