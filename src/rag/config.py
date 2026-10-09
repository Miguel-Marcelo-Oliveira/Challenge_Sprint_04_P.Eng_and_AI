from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[2]
KNOWLEDGE_BASE_DIR = ROOT_DIR / "data" / "knowledge_base"
CHROMA_DIR = ROOT_DIR / "data" / "chroma_db"

# Arquitetura do Sprint 04:
# - embeddings: Ollama LOCAL
# - LLMs: Ollama Cloud via API
OLLAMA_LOCAL_HOST = os.getenv("OLLAMA_LOCAL_HOST", "http://localhost:11434")
OLLAMA_CLOUD_HOST = os.getenv("OLLAMA_CLOUD_HOST", "https://ollama.com")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:120b")
OLLAMA_MODEL_2 = os.getenv("OLLAMA_MODEL_2", "nemotron-3-nano:30b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
TOP_P = float(os.getenv("TOP_P", "0.95"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
TOP_K = int(os.getenv("TOP_K", "3"))
MIN_RELEVANCE = float(os.getenv("MIN_RELEVANCE", "0.20"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
PROMPT_VERSION = os.getenv("RAG_PROMPT_VERSION", "v2.0")


def validar_configuracao_llm() -> None:
    if not OLLAMA_API_KEY:
        raise EnvironmentError(
            "OLLAMA_API_KEY não configurada. Informe a chave do Ollama Cloud no arquivo .env."
        )
    if not 0 <= TEMPERATURE <= 2:
        raise ValueError("TEMPERATURE deve estar entre 0 e 2.")
    if not 0 < TOP_P <= 1:
        raise ValueError("TOP_P deve estar entre 0 e 1.")
    if TOP_K < 1:
        raise ValueError("TOP_K deve ser >= 1.")
    if CHUNK_OVERLAP >= CHUNK_SIZE:
        raise ValueError("CHUNK_OVERLAP deve ser menor que CHUNK_SIZE.")


def validar_configuracao() -> None:
    """Mantém compatibilidade com o restante do projeto."""
    validar_configuracao_llm()
