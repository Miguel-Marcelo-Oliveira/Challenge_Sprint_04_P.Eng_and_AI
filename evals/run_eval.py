from __future__ import annotations

import json
from pathlib import Path

from evals.eval_set import PERGUNTAS
from evals.manual_eval import score_answer_relevancy, score_faithfulness
from src.rag.pipeline import construir_chain, recuperar_contexto


ROOT = Path(__file__).resolve().parents[1]


def avaliar(model=None, prompt_version=None):
    """Executa a avaliação sobre o conjunto de perguntas."""
    if prompt_version:
        chain = construir_chain(model=model, version=prompt_version)
    else:
        chain = construir_chain(model=model)

    rows = []

    for question in PERGUNTAS:
        docs = recuperar_contexto(question)
        contexts = [doc.page_content for doc in docs]

        if not docs:
            answer = "Não encontrei essa informação nos documentos fornecidos."
        else:
            answer = chain.invoke(question)

        rows.append(
            {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "faithfulness_fallback": score_faithfulness(
                    answer, contexts
                ),
                "answer_relevancy_fallback": score_answer_relevancy(
                    question, answer
                ),
                "sources": [
                    {
                        "source": doc.metadata.get("source"),
                        "page": int(doc.metadata.get("page", 0)) + 1,
                        "similarity": doc.metadata.get("retrieval_similarity"),
                    }
                    for doc in docs
                ],
            }
        )

    return rows


if __name__ == "__main__":
    rows = avaliar()
    output_path = ROOT / "evals" / "iteration_current.json"

    output_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Resultado salvo em {output_path}")
