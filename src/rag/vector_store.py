from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.rag.config import CHROMA_DIR
from src.rag.embeddings import criar_embeddings


COLLECTION_NAME = "chargegrid_goodwe"


def criar_vector_store(
    documentos: list[Document],
    persist_directory: str | Path = CHROMA_DIR,
):
    """Cria novamente a coleção ChromaDB com os documentos fornecidos."""
    path = Path(persist_directory)
    path.mkdir(parents=True, exist_ok=True)

    embeddings = criar_embeddings()

    db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(path),
    )

    try:
        db.delete_collection()
    except Exception:
        pass

    return Chroma.from_documents(
        documents=documentos,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(path),
    )


def carregar_vector_store(
    persist_directory: str | Path = CHROMA_DIR,
):
    """Abre a coleção persistida e verifica se ela possui documentos."""
    db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=criar_embeddings(),
        persist_directory=str(persist_directory),
    )

    if db._collection.count() == 0:
        raise ValueError("ChromaDB está vazia. Execute: python indexar.py")

    return db
