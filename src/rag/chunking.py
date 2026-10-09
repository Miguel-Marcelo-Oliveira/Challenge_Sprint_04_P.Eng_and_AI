from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.rag.config import CHUNK_OVERLAP, CHUNK_SIZE


def dividir_documentos(documentos: list[Document], chunk_size: int = None, chunk_overlap: int = None) -> list[Document]:
    """Divide os documentos em chunks para a busca semântica."""
    from src.rag.config import CHUNK_SIZE, CHUNK_OVERLAP

    size = chunk_size if chunk_size is not None else CHUNK_SIZE
    overlap = chunk_overlap if chunk_overlap is not None else CHUNK_OVERLAP

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    return splitter.split_documents(documentos)
