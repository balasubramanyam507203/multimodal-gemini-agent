import os
import fitz  # PyMuPDF
import numpy as np
from typing import List

from config import chunks_collection
import google.generativeai as genai

DOCS_DIR = "data/docs"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200


def embed_text(text: str) -> List[float]:
    """Use Gemini embedding model to create a vector."""
    res = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_document",
    )
    return res["embedding"]


def chunk_text(text: str) -> list:
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def extract_text_from_pdf(path: str) -> str:
    doc = fitz.open(path)
    texts = []
    for page in doc:
        texts.append(page.get_text("text"))
    doc.close()
    return "\n".join(texts)


def clear_collection():
    print("Clearing existing chunks...")
    chunks_collection.delete_many({})


def ingest_docs():
    clear_collection()

    for filename in os.listdir(DOCS_DIR):
        if not filename.lower().endswith(".pdf"):
            continue

        full_path = os.path.join(DOCS_DIR, filename)
        print(f"Processing {full_path} ...")

        text = extract_text_from_pdf(full_path)
        if not text.strip():
            print("  -> No text found, skipping.")
            continue

        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue

            emb = embed_text(chunk)
            doc = {
                "source": filename,
                "chunk_index": i,
                "text": chunk,
                "embedding": emb,
            }
            chunks_collection.insert_one(doc)
        print(f"  -> Stored {len(chunks)} chunks.")

    print("Ingestion complete.")


if __name__ == "__main__":
    ingest_docs()
