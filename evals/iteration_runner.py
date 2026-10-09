from __future__ import annotations

import json
import os
from pathlib import Path

from evals.run_eval import avaliar
from evals.evaluate_ragas import avaliar_com_ragas

from src.rag import config
from src.rag.pipeline import indexar_base


ROOT = Path(__file__).resolve().parents[1]


def run(
    name,
    chunk_size,
    chunk_overlap,
    prompt_version,
):
    """
    Executa uma iteração completa:

    1. Configura chunking.
    2. Configura versão do prompt.
    3. Indexa a base.
    4. Executa o RAG.
    5. Tenta avaliar com RAGAS.
    6. Usa fallback somente se RAGAS falhar.
    """

    print()
    print("=" * 60)
    print(f"EXECUTANDO {name}")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. CONFIGURAÇÃO
    # ---------------------------------------------------------

    os.environ["CHUNK_SIZE"] = str(chunk_size)
    os.environ["CHUNK_OVERLAP"] = str(chunk_overlap)
    os.environ["RAG_PROMPT_VERSION"] = prompt_version

    config.CHUNK_SIZE = chunk_size
    config.CHUNK_OVERLAP = chunk_overlap
    config.PROMPT_VERSION = prompt_version

    # ---------------------------------------------------------
    # 2. INDEXAÇÃO
    # ---------------------------------------------------------

    print(
        f"Chunk size: {chunk_size}"
    )

    print(
        f"Chunk overlap: {chunk_overlap}"
    )

    print(
        f"Prompt: {prompt_version}"
    )

    print("Indexando base...")

    index = indexar_base()

    # ---------------------------------------------------------
    # 3. GERAÇÃO DAS RESPOSTAS
    # ---------------------------------------------------------

    print("Executando perguntas de avaliação...")

    rows = avaliar(
        prompt_version=prompt_version
    )

    # ---------------------------------------------------------
    # 4. AVALIAÇÃO
    # ---------------------------------------------------------

    print("Tentando avaliação RAGAS...")

    avaliacao = avaliar_com_ragas(rows)

    method = avaliacao["method"]

    faith = avaliacao["averages"]["faithfulness"]
    relevancy = avaliacao["averages"]["answer_relevancy"]
    mean = avaliacao["averages"]["mean"]

    # ---------------------------------------------------------
    # 5. RESULTADO DA ITERAÇÃO
    # ---------------------------------------------------------

    resultado = {
        "iteration": name,

        "chunk_size": chunk_size,

        "chunk_overlap": chunk_overlap,

        "prompt_version": prompt_version,

        "pages": index.paginas,

        "chunks": index.chunks,

        "evaluation_method": method,

        "faithfulness": faith,

        "answer_relevancy": relevancy,

        "mean_score": mean,

        "positive_gain": None,

        "gain_vs_previous": None,
    }

    # ---------------------------------------------------------
    # 6. REGISTRAR ERRO DO RAGAS, SE EXISTIR
    # ---------------------------------------------------------

    if "ragas_error" in avaliacao:
        resultado["ragas_error"] = (
            avaliacao["ragas_error"]
        )

    print()
    print(f"Método: {method}")
    print(f"Faithfulness: {faith:.3f}")
    print(f"Answer relevancy: {relevancy:.3f}")
    print(f"Média: {mean:.3f}")

    if method == "ragas":
        print("RAGAS executado com sucesso.")
    else:
        print("RAGAS falhou. Fallback utilizado.")

    return resultado


def calcular_ganhos(resultados):
    """
    Calcula o ganho de cada iteração
    em relação à iteração anterior.
    """

    for i in range(len(resultados)):

        # Primeira iteração não possui comparação anterior.
        if i == 0:
            resultados[i]["gain_vs_previous"] = None
            resultados[i]["positive_gain"] = None
            continue

        atual = resultados[i]["mean_score"]

        anterior = resultados[i - 1]["mean_score"]

        gain = atual - anterior

        resultados[i]["gain_vs_previous"] = round(
            gain,
            3,
        )

        resultados[i]["positive_gain"] = (
            gain > 0
        )

    return resultados


if __name__ == "__main__":

    # ---------------------------------------------------------
    # ITERAÇÕES
    # ---------------------------------------------------------

    results = [

        run(
            "iteracao_1",
            512,
            64,
            "v1.0",
        ),

        run(
            "iteracao_2",
            800,
            100,
            "v2.0",
        ),
    ]

    # ---------------------------------------------------------
    # CÁLCULO DOS GANHOS
    # ---------------------------------------------------------

    results = calcular_ganhos(
        results
    )

    # ---------------------------------------------------------
    # SALVAR JSON
    # ---------------------------------------------------------

    output_path = (
        ROOT
        / "evals"
        / "iterations_current.json"
    )

    output_path.write_text(
        json.dumps(
            results,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 60)
    print("RESULTADO FINAL")
    print("=" * 60)

    print(
        json.dumps(
            results,
            ensure_ascii=False,
            indent=2,
        )
    )

    print()
    print(
        f"Resultado salvo em: "
        f"{output_path}"
    )