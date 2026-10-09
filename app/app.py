from __future__ import annotations

import os

import gradio as gr

from src.guardrails.moderation import (
    MENSAGEM_BLOQUEIO,
    eh_tentativa_de_jailbreak,
)
from src.guardrails.scope_validator import (
    MENSAGEM_FORA_DE_ESCOPO,
    esta_fora_de_escopo,
    verificar_risco_profissional,
)
from src.rag.config import TOP_K
from src.rag.pipeline import (
    construir_chain_com_contexto,
    formatar_contexto,
    recuperar_contexto,
)


_chain = None


def obter_chain():
    """Cria o chain apenas quando ele for necessário."""
    global _chain

    if _chain is None:
        _chain = construir_chain_com_contexto()

    return _chain


def responder(mensagem, historico=None):
    """Processa uma pergunta do usuário e retorna a resposta do RAG."""
    mensagem = (mensagem or "").strip()

    if not mensagem:
        return "Digite uma pergunta."

    if eh_tentativa_de_jailbreak(mensagem):
        return MENSAGEM_BLOQUEIO

    risco = verificar_risco_profissional(mensagem)
    if risco:
        return risco

    if esta_fora_de_escopo(mensagem):
        return MENSAGEM_FORA_DE_ESCOPO

    docs = recuperar_contexto(mensagem, k=TOP_K)

    if not docs:
        return "Não encontrei essa informação nos documentos fornecidos."

    contexto = formatar_contexto(docs)
    resposta = obter_chain().invoke(
        {
            "contexto": contexto,
            "pergunta": mensagem,
        }
    )

    fontes = []

    for doc in docs:
        item = (
            f"[Fonte: {doc.metadata.get('source', 'documento')}, "
            f"pág. {int(doc.metadata.get('page', 0)) + 1}]"
        )

        if item not in fontes:
            fontes.append(item)

    return resposta.rstrip() + "\n\nFontes recuperadas:\n" + "\n".join(fontes)


def criar_app():
    """Cria a interface Gradio do assistente."""
    with gr.Blocks(title="ChargeGrid AI Assistant — Sprint 04") as demo:
        gr.Markdown(
            "# ChargeGrid AI Assistant\n"
            "### GoodWe · Sprint 04 — RAG"
        )

        gr.Markdown(
            "Pergunte sobre os documentos GoodWe indexados. "
            "As respostas são fundamentadas no contexto recuperado "
            "e exibem as fontes."
        )

        gr.ChatInterface(
            fn=responder,
            title="Assistente GoodWe",
            examples=[
                "Quais produtos aparecem nos documentos GoodWe?",
                "Quais informações são fornecidas sobre carregamento?",
            ],
        )

        def encerrar_aplicacao():
            demo.close()

        gr.Button(
            "⏹ Encerrar aplicação",
            variant="stop"
        ).click(
            fn=encerrar_aplicacao,
            inputs=None,
            outputs=None
        )

    return demo


if __name__ == "__main__":
    compartilhar = os.getenv("GRADIO_SHARE", "false").lower() == "true"
    criar_app().launch(share=compartilhar)
