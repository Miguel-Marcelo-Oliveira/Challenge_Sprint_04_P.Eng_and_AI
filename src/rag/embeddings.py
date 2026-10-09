from __future__ import annotations

from langchain_ollama import OllamaEmbeddings

from src.rag.config import EMBEDDING_MODEL, OLLAMA_LOCAL_HOST


def criar_embeddings() -> OllamaEmbeddings:
    """Cria o modelo de embedding exclusivamente no Ollama local."""
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_LOCAL_HOST,
    )
