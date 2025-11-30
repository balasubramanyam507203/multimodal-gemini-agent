from typing import List, Optional
import numpy as np
from PIL import Image

from config import chunks_collection
import google.generativeai as genai


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def embed_query(query: str) -> List[float]:
    res = genai.embed_content(
        model="models/embedding-001",
        content=query,
        task_type="retrieval_query",
    )
    return res["embedding"]


def retrieve_top_k(query: str, k: int = 5) -> List[dict]:
    """Simple in-Python vector search using MongoDB-stored embeddings."""
    q_vec = np.array(embed_query(query), dtype=float)

    docs = list(chunks_collection.find({}, {"text": 1, "source": 1, "embedding": 1}))
    if not docs:
        return []

    for d in docs:
        emb = np.array(d["embedding"], dtype=float)
        d["score"] = cosine_similarity(q_vec, emb)

    docs.sort(key=lambda x: x["score"], reverse=True)
    return docs[:k]


def build_context(chunks: List[dict]) -> str:
    context_parts = []
    for c in chunks:
        context_parts.append(f"[{c['source']} | score={c['score']:.3f}]\n{c['text']}")
    return "\n\n---\n\n".join(context_parts)


def answer_with_gemini(
    query: str,
    retrieved_chunks: List[dict],
    image_path: Optional[str] = None,
) -> str:
    context = build_context(retrieved_chunks)

    system_prompt = (
        "You are a helpful AI assistant that answers questions about PDF reports "
        "containing charts, diagrams, and tables. Use the provided context from the "
        "documents AND the image (if any) to give a clear, step-by-step answer. "
        "Explain any numbers or trends in a simple way."
    )

    parts = [
        system_prompt,
        "\n\nUser question:\n",
        query,
        "\n\nRelevant document context:\n",
        context,
    ]

    model = genai.GenerativeModel("gemini-1.5-flash")

    if image_path:
        image = Image.open(image_path)
        response = model.generate_content([*parts, image])
    else:
        response = model.generate_content(parts)

    return response.text
