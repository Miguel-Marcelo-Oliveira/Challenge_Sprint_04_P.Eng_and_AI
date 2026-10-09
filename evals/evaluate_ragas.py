from __future__ import annotations

import json
import math
from pathlib import Path

from evals.run_eval import avaliar


ROOT = Path(__file__).resolve().parents[1]


def _media_segura(valor):
    """Converte uma média para float válido."""
    valor = float(valor)

    if not math.isfinite(valor):
        raise ValueError(f"Score inválido: {valor}")

    return valor


def avaliar_com_ragas(rows):
    """
    Tenta avaliar as respostas usando RAGAS.

    Se RAGAS funcionar:
        retorna method = 'ragas'

    Se RAGAS falhar:
        retorna method = 'manual_fallback'
        e registra o erro.
    """

    # ---------------------------------------------------------
    # 1. FALLBACK MANUAL
    # ---------------------------------------------------------

    fallback_f = sum(
        row["faithfulness_fallback"]
        for row in rows
    ) / len(rows)

    fallback_r = sum(
        row["answer_relevancy_fallback"]
        for row in rows
    ) / len(rows)

    fallback_mean = (fallback_f + fallback_r) / 2

    fallback_result = {
        "method": "manual_fallback",
        "averages": {
            "faithfulness": round(fallback_f, 3),
            "answer_relevancy": round(fallback_r, 3),
            "mean": round(fallback_mean, 3),
        },
        "rows": rows,
    }

    # ---------------------------------------------------------
    # 2. TENTATIVA DE RAGAS
    # ---------------------------------------------------------

    try:
        from datasets import Dataset

        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            faithfulness,
        )

        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper

        from langchain_ollama import (
            ChatOllama,
            OllamaEmbeddings,
        )

        # -----------------------------------------------------
        # 3. CONFIGURAÇÃO DO LLM JUIZ
        # -----------------------------------------------------

        ragas_llm = LangchainLLMWrapper(
            ChatOllama(
                model="gemma4:cloud",
                temperature=0,
            )
        )

        # -----------------------------------------------------
        # 4. CONFIGURAÇÃO DOS EMBEDDINGS
        # -----------------------------------------------------

        ragas_embeddings = LangchainEmbeddingsWrapper(
            OllamaEmbeddings(
                model="nomic-embed-text"
            )
        )

        # RAGAS usará o Ollama em vez de tentar
        # utilizar automaticamente outro provedor.
        faithfulness.llm = ragas_llm
        answer_relevancy.llm = ragas_llm
        answer_relevancy.embeddings = ragas_embeddings

        # -----------------------------------------------------
        # 5. DATASET PARA O RAGAS
        # -----------------------------------------------------

        dataset = Dataset.from_dict(
            {
                "question": [
                    row["question"]
                    for row in rows
                ],

                "answer": [
                    row["answer"]
                    for row in rows
                ],

                "contexts": [
                    row["contexts"]
                    for row in rows
                ],
            }
        )

        # -----------------------------------------------------
        # 6. EXECUÇÃO DO RAGAS
        # -----------------------------------------------------

        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
            ],
        )

        dataframe = result.to_pandas()

        ragas_f = _media_segura(
            dataframe["faithfulness"].mean()
        )

        ragas_r = _media_segura(
            dataframe["answer_relevancy"].mean()
        )

        ragas_mean = (ragas_f + ragas_r) / 2

        # -----------------------------------------------------
        # 7. RESULTADO RAGAS
        # -----------------------------------------------------

        return {
            "method": "ragas",

            "averages": {
                "faithfulness": round(ragas_f, 3),
                "answer_relevancy": round(ragas_r, 3),
                "mean": round(ragas_mean, 3),
            },

            "scores": dataframe[
                [
                    "faithfulness",
                    "answer_relevancy",
                ]
            ].to_dict(
                orient="records"
            ),

            "rows": rows,
        }

    # ---------------------------------------------------------
    # 8. QUALQUER ERRO → FALLBACK
    # ---------------------------------------------------------

    except Exception as exc:

        fallback_result["ragas_error"] = (
            f"{type(exc).__name__}: {exc}"
        )

        print(
            "RAGAS falhou. "
            "Usando fallback manual."
        )

        print(
            f"Erro do RAGAS: "
            f"{type(exc).__name__}: {exc}"
        )

        return fallback_result


def main():
    """
    Executa a avaliação atual.

    RAGAS é tentado primeiro.
    Fallback é usado somente se RAGAS falhar.
    """

    rows = avaliar()

    resultado = avaliar_com_ragas(rows)

    output_path = (
        ROOT
        / "evals"
        / "rag_results_current.json"
    )

    output_path.write_text(
        json.dumps(
            resultado,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(
        f"Método utilizado: "
        f"{resultado['method']}"
    )

    print(
        f"Faithfulness: "
        f"{resultado['averages']['faithfulness']:.3f}"
    )

    print(
        f"Answer relevancy: "
        f"{resultado['averages']['answer_relevancy']:.3f}"
    )

    print(
        f"Média: "
        f"{resultado['averages']['mean']:.3f}"
    )

    print(
        f"Resultado salvo em {output_path}"
    )


if __name__ == "__main__":
    main()