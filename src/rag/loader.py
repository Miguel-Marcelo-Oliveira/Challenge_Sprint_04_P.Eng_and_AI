from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document


def carregar_pdf(caminho: str | Path) -> list[Document]:
    """Carrega um PDF e padroniza os metadados de fonte e página."""
    path = Path(caminho)
    docs = PyMuPDFLoader(str(path)).load()

    for doc in docs:
        doc.metadata["source"] = path.name
        doc.metadata["page"] = int(doc.metadata.get("page", 0))

    return docs


def carregar_base(pasta: str | Path) -> tuple[list[Document], list[str]]:
    """Carrega todos os PDFs da knowledge base."""
    pasta = Path(pasta)
    documentos = []
    sem_texto = []

    for pdf in sorted(pasta.glob("*.pdf")):
        docs = carregar_pdf(pdf)
        possui_texto = "".join(
            doc.page_content.strip() for doc in docs
        ).strip()

        if possui_texto:
            documentos.extend(docs)
        else:
            sem_texto.append(pdf.name)

    if not documentos:
        raise ValueError(
            "Nenhum PDF com texto extraível foi encontrado na knowledge_base."
        )

    return documentos, sem_texto
