"""Indexa a knowledge base no ChromaDB persistente.

Execute a partir da raiz do projeto:

    python indexar.py
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


try:
    from src.rag.pipeline import indexar_base

except ModuleNotFoundError as exc:
    pacote = exc.name or "uma dependência"

    raise SystemExit(
        f"ERRO: a dependência '{pacote}' não está instalada.\n"
        "Crie/ative um ambiente virtual e execute: "
        "pip install -r requirements.txt"
    ) from exc


def main() -> int:
    """Executa a indexação e mostra um resumo do resultado."""
    try:
        resultado = indexar_base()

    except EnvironmentError as exc:
        print(f"ERRO DE CONFIGURAÇÃO: {exc}", file=sys.stderr)
        return 1

    except Exception as exc:
        print(
            f"ERRO DURANTE A INDEXAÇÃO: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print(
            "Verifique: Ollama local, modelo de embedding, PDFs em "
            "data/knowledge_base/ e as dependências do requirements.txt.",
            file=sys.stderr,
        )
        return 1

    print(f"Páginas indexadas: {resultado.paginas}")
    print(f"Chunks criados: {resultado.chunks}")

    if resultado.pdfs_sem_texto:
        print("PDFs sem texto extraível (não indexados):")

        for nome in resultado.pdfs_sem_texto:
            print(f"  - {nome}")

    print("ChromaDB persistente criado em data/chroma_db/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
