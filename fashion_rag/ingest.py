"""
Step 1 — Run once to embed the CSV and build the ChromaDB index.
Usage: python ingest.py --csv fashion.csv
"""

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
    df = pd.read_csv(path, sep="\t")
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
