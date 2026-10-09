from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_ollama import ChatOllama

from src.rag.chunking import dividir_documentos
from src.rag.config import (
    CHROMA_DIR,
    KNOWLEDGE_BASE_DIR,
    MAX_TOKENS,
    OLLAMA_API_KEY,
    OLLAMA_CLOUD_HOST,
    OLLAMA_MODEL,
    PROMPT_VERSION,
    TEMPERATURE,
    TOP_K,
    TOP_P,
    validar_configuracao_llm,
)
from src.rag.loader import carregar_base
from src.rag.prompt_rag import criar_prompt
from src.rag.retriever import criar_retriever, recuperar_relevantes
from src.rag.vector_store import carregar_vector_store, criar_vector_store


@dataclass
class ResultadoIndexacao:
    paginas: int
    chunks: int
    pdfs_sem_texto: list[str]


def formatar_contexto(docs: list[Document]) -> str:
    """Formata os documentos recuperados para serem enviados ao prompt."""
    partes = []

    for i, doc in enumerate(docs, 1):
        fonte = doc.metadata.get("source", "documento desconhecido")
        pagina = int(doc.metadata.get("page", 0)) + 1

        partes.append(
            f"[Trecho {i} — {fonte}, pág. {pagina}]\n"
            f"{doc.page_content}"
        )

    return "\n\n---\n\n".join(partes)


def indexar_base() -> ResultadoIndexacao:
    """Carrega, divide e indexa a knowledge base no ChromaDB."""
    paginas, sem_texto = carregar_base(KNOWLEDGE_BASE_DIR)
    chunks = dividir_documentos(paginas)

    criar_vector_store(chunks, CHROMA_DIR)

    return ResultadoIndexacao(
        paginas=len(paginas),
        chunks=len(chunks),
        pdfs_sem_texto=sem_texto,
    )


def _criar_llm(model: str | None = None):
    """Cria o LLM usando o Ollama Cloud via API.

    O embedding não passa por esta função. Ele é criado exclusivamente
    no Ollama local em ``src.rag.embeddings``.
    """
    validar_configuracao_llm()

    return ChatOllama(
        model=model or OLLAMA_MODEL,
        base_url=OLLAMA_CLOUD_HOST,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        num_predict=MAX_TOKENS,
        client_kwargs={
            "headers": {
                "Authorization": f"Bearer {OLLAMA_API_KEY}",
            },
            "verify": False, # Desativa verificação SSL para o teste de auditoria
        },
    )


def construir_chain_com_contexto(
    model: str | None = None,
    version: str = PROMPT_VERSION,
):
    """Cria um chain que recebe contexto e pergunta diretamente."""
    return criar_prompt(version) | _criar_llm(model) | StrOutputParser()


def construir_chain(
    model: str | None = None,
    version: str = PROMPT_VERSION,
):
    """Cria o chain completo de recuperação + prompt + LLM."""
    retriever = criar_retriever(carregar_vector_store(CHROMA_DIR))
    prompt = criar_prompt(version)
    llm = _criar_llm(model)

    return (
        {
            "contexto": retriever | RunnableLambda(formatar_contexto),
            "pergunta": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


def recuperar_contexto(pergunta: str, k: int = TOP_K):
    """Recupera os trechos mais relevantes para uma pergunta."""
    vector_store = carregar_vector_store(CHROMA_DIR)
    return recuperar_relevantes(vector_store, pergunta, k=k)
