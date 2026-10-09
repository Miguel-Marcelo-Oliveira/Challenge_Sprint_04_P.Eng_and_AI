from __future__ import annotations

import re
from collections import Counter


STOPWORDS = set(
    "a o os as um uma de do da dos das e em para com por que qual quais é "
    "são no na nos nas ao aos às ou como seu sua seus suas uma um sobre entre".split()
)


def tokens(text: str) -> list[str]:
    """Extrai tokens relevantes para a avaliação lexical."""
    return [
        token
        for token in re.findall(
            r"[a-záéíóúãõâêôç0-9-]+",
            text.lower(),
        )
        if token not in STOPWORDS and len(token) > 2
    ]


def overlap(a: str, b: str) -> float:
    """Calcula a proporção de tokens de A encontrados em B."""
    tokens_a = Counter(tokens(a))
    tokens_b = Counter(tokens(b))

    if not tokens_a:
        return 0.0

    score = sum((tokens_a & tokens_b).values()) / sum(tokens_a.values())
    return min(score, 1.0)


def score_faithfulness(answer: str, contexts: list[str]) -> float:
    """Calcula um fallback de 0 a 1 para suporte lexical da resposta.

    Quanto maior a proporção de termos relevantes da resposta presente no
    contexto recuperado, maior o grounding. É uma aproximação manual e não
    substitui uma avaliação humana ou o RAGAS oficial.
    """
    context = " ".join(contexts)
    sentences = [
        sentence.strip()
        for sentence in re.split(r"[.!?]\s+", answer)
        if len(tokens(sentence)) >= 3
    ]

    if not sentences:
        return 0.0

    score = sum(overlap(sentence, context) for sentence in sentences)
    return round(score / len(sentences), 3)


def score_answer_relevancy(question: str, answer: str) -> float:
    """Calcula a relevância lexical da resposta em relação à pergunta."""
    return round(overlap(question, answer), 3)
