from __future__ import annotations

import json
import time
from pathlib import Path

from evals.eval_set import PERGUNTAS
from evals.manual_eval import score_answer_relevancy, score_faithfulness
from src.rag.config import (
    MAX_TOKENS,
    OLLAMA_MODEL,
    OLLAMA_MODEL_2,
    TEMPERATURE,
    TOP_K,
    TOP_P,
)
from src.rag.pipeline import construir_chain, recuperar_contexto


ROOT = Path(__file__).resolve().parents[1]


def main():
    """Executa as mesmas perguntas com os dois modelos de LLM."""
    results = []

    modelos = dict.fromkeys([OLLAMA_MODEL, OLLAMA_MODEL_2])

    for model in modelos:
        chain = construir_chain(model=model)
        rows = []

        for question in PERGUNTAS:
            docs = recuperar_contexto(question)
            contexts = [doc.page_content for doc in docs]

            start = time.perf_counter()

            if docs:
                answer = chain.invoke(question)
            else:
                answer = "Não encontrei essa informação nos documentos fornecidos."

            latency = time.perf_counter() - start

            rows.append(
                {
                    "question": question,
                    "answer": answer,
                    "faithfulness_fallback": score_faithfulness(
                        answer, contexts
                    ),
                    "answer_relevancy_fallback": score_answer_relevancy(
                        question, answer
                    ),
                    "latency_s": round(latency, 3),
                }
            )

        results.append(
            {
                "model": model,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "max_tokens": MAX_TOKENS,
                "top_k": TOP_K,
                "rows": rows,
            }
        )

    output_path = ROOT / "evals" / "model_comparison_current.json"
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
