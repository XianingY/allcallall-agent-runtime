from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LlamaIndexRetrievalResult:
    available: bool
    hits: list[dict[str, Any]]
    reason: str = ""


def run_fixture_retrieval(query: str, documents: list[dict[str, str]], top_k: int = 5) -> LlamaIndexRetrievalResult:
    """Eval-only LlamaIndex adapter.

    The production knowledge path remains in Go. This adapter lets interview/eval
    fixtures compare a framework-backed retrieval baseline when llama-index-core is
    installed; otherwise it returns a deterministic lexical fallback with
    available=false so CI does not depend on the optional package.
    """

    # Guard the whole indexing path, not just the import: llama-index-core
    # resolves a default OpenAI embedding model at query time and raises
    # ImportError from deep inside embeddings/utils.py when the optional
    # llama-index-embeddings-openai piece is absent. CI installs only the
    # [dev] extra (no llamaindex), but a full local `uv sync --all-extras`
    # hits exactly that half-installed state; degrade instead of exploding.
    try:
        from llama_index.core import Document, VectorStoreIndex

        nodes = [Document(text=item.get("text", ""), metadata={"title": item.get("title", "")}) for item in documents]
        index = VectorStoreIndex.from_documents(nodes)
        retriever = index.as_retriever(similarity_top_k=top_k)
        results = retriever.retrieve(query)
    except Exception:
        return lexical_fallback(query, documents, top_k, reason="llama_index_not_usable")

    hits = [
        {
            "title": str(item.node.metadata.get("title", "")),
            "score": float(item.score or 0),
            "text": item.node.get_content()[:300],
        }
        for item in results
    ]
    return LlamaIndexRetrievalResult(available=True, hits=hits)


def lexical_fallback(
    query: str,
    documents: list[dict[str, str]],
    top_k: int,
    *,
    reason: str,
) -> LlamaIndexRetrievalResult:
    tokens = {token for token in query.lower().split() if len(token) >= 2}
    scored: list[tuple[int, dict[str, str]]] = []
    for item in documents:
        text = f"{item.get('title', '')} {item.get('text', '')}".lower()
        score = sum(1 for token in tokens if token in text)
        scored.append((score, item))
    scored.sort(key=lambda item: item[0], reverse=True)
    hits = [
        {
            "title": item.get("title", ""),
            "score": float(score),
            "text": item.get("text", "")[:300],
        }
        for score, item in scored[:top_k]
    ]
    return LlamaIndexRetrievalResult(available=False, hits=hits, reason=reason)
