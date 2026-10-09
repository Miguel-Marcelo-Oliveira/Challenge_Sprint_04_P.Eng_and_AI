from __future__ import annotations

from langchain_chroma import Chroma

from src.rag.config import MIN_RELEVANCE, TOP_K


def criar_retriever(vector_store: Chroma, k: int = TOP_K):
    """Cria o retriever de busca por similaridade."""
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def recuperar_relevantes(
    vector_store: Chroma,
    pergunta: str,
    k: int = TOP_K,
    min_relevance: float = MIN_RELEVANCE,
):
    """Recupera documentos e filtra os que estão abaixo do limiar."""
    pares = vector_store.similarity_search_with_score(pergunta, k=k)
    docs = []

    for doc, distance in pares:
        similarity = max(0.0, min(1.0, 1.0 - float(distance)))
        doc.metadata["retrieval_similarity"] = round(similarity, 4)

        if similarity >= min_relevance:
            docs.append(doc)

    return docs
