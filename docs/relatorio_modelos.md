# Relatório de modelos e parâmetros — ChargeGrid / GoodWe · Sprint 04

**Disciplina:** Prompt and Artificial Intelligence · FIAP 2026.2 · **Projeto:** EV Challenge 2026 — GoodWe
**Fonte dos números:** `evals/model_comparison_current.json` (gerado por `python -m evals.compare_models`), mais recálculos feitos sobre esse arquivo e sobre `evals/rag_results_current.json`.

> **Resumo.** Foram comparados dois LLMs no mesmo RAG, com as mesmas 39 perguntas, os mesmos contextos e os mesmos parâmetros. **Na métrica do projeto os dois ficaram empatados** (média 0,571 × 0,576; diferença sem significância estatística). A diferença que importa está no comportamento: o `gemma4:31b` seguiu o formato de citação em 100% das respostas e preferiu recusar a errar; o `gpt-oss:120b` variou o formato de citação, teve uma resposta truncada e **errou o ano de fundação da GoodWe com citação**. Recomendação: `gemma4:31b` como padrão, com a ressalva de que a evidência é limitada (seção 6).

---

## 1. Configuração do experimento

| Item | Valor |
|---|---|
| Provedor dos LLMs | Ollama Cloud (API, `OLLAMA_CLOUD_HOST=https://ollama.com`) |
| Modelo A | `gemma4:31b` |
| Modelo B | `gpt-oss:120b` |
| Embeddings (iguais nos dois) | `nomic-embed-text`, Ollama local |
| Recuperação (igual nos dois) | ChromaDB, `top_k = 3`, limiar de relevância 0,20 |
| Prompt | prompt RAG padrão do projeto (v2.0 na configuração do repositório) |
| Perguntas | 39 (`evals/eval_set.py`) |
| Métrica | fallback manual 0–1: faithfulness e answer relevancy (RAGAS indisponível no ambiente; ver `relatorio_rag.md`, seção 6) |

Os contextos recuperados são os mesmos para os dois modelos: recalculando os scores do `gemma4:31b` com os contextos do `rag_results_current.json`, a diferença em relação ao registrado é 0,0000. Logo, qualquer diferença de score vem da geração, não da recuperação.

> **Observação de configuração.** O `.env.example` lista `OLLAMA_MODEL=gpt-oss:120b` e `OLLAMA_MODEL_2=nemotron-3-nano:30b`, mas o JSON da comparação registra `gemma4:31b` e `gpt-oss:120b`. O `.env.example` deve ser alinhado com o que foi de fato executado.

---

## 2. Parâmetros documentados

Os dois modelos rodaram com **exatamente os mesmos parâmetros** (registrados no JSON):

| Parâmetro | Valor | O que controla | Justificativa |
|---|---|---|---|
| `temperature` | **0** | Aleatoriedade da geração | RAG precisa de respostas estáveis e ancoradas no contexto; Aula 06 (slide 13) usa `temperature=0` "para RAG (menos variação)" e a Aula 07 usa 0 também no juiz. |
| `top_k` (recuperação) | **3** | Nº de chunks enviados ao prompt | Valor da Aula 06 (slide 10, `search_kwargs={"k": 3}`): contexto suficiente sem diluir o prompt. |
| `top_p` | **0,95** | Corte de probabilidade acumulada na amostragem | Com `temperature = 0` a escolha é praticamente determinística, então o `top_p` tem efeito desprezível; foi mantido como valor padrão e documentado. |
| `max_tokens` (`num_predict`) | **512** | Limite de tokens gerados | As respostas têm em média ~240 caracteres (máx. 781); 512 tokens sobram para respostas curtas. Ver o caso de truncamento na seção 5. |

Observação sobre `temperature = 0`: mesmo com ele, execuções repetidas de modelos em nuvem podem variar levemente (a iteração 2 e o `rag_results_current.json` diferiram em 0,006 na média).

---

## 3. Resultados quantitativos

Scores do fallback manual, 39 perguntas (valores como gravados em `model_comparison_current.json`):

| Modelo | Faithfulness | Answer relevancy | **Média** | Faith. (mediana) | Relev. (mediana) |
|---|---:|---:|---:|---:|---:|
| `gemma4:31b` | **0,530** | 0,612 | **0,571** | 0,611 | 0,667 |
| `gpt-oss:120b` | 0,494 | **0,657** | **0,576** | 0,542 | 0,700 |

Latência por pergunta (inclui só a chamada ao LLM; a recuperação não entra na medição):

| Modelo | Média | Mediana | Máxima | Total (39 perguntas) |
|---|---:|---:|---:|---:|
| `gemma4:31b` | 1,03 s | 0,73 s | 3,72 s | 40,2 s |
| `gpt-oss:120b` | 1,09 s | 0,89 s | 3,33 s | 42,4 s |

### Os dois modelos são estatisticamente distintos?
Não. Bootstrap pareado (5.000 reamostragens) da diferença *gemma − gpt-oss* por pergunta:

| Métrica | Diferença média | IC 95% | Perguntas em que gemma > / < / = |
|---|---:|---|---|
| Faithfulness | +0,036 | [−0,018 ; +0,090] | 24 / 11 / 4 |
| Answer relevancy | −0,045 | [−0,153 ; +0,060] | 9 / 12 / 18 |
| Média das duas | −0,005 | [−0,074 ; +0,063] | 19 / 16 / 4 |

Todos os intervalos incluem zero. Com 39 perguntas e uma métrica lexical, **não é possível afirmar que um modelo seja melhor que o outro pelos scores**.

### Sensibilidade ao formato de saída
O `gpt-oss:120b` gerou caracteres tipográficos atípicos (hífen não-quebrável U+2011 em "GW7K‑HCA", espaço fino U+202F antes de unidades e da página) em **34 das 39 respostas**; o `gemma4:31b` em nenhuma. Isso quebra a tokenização da rubrica lexical. Normalizando esses caracteres antes de pontuar, o `gpt-oss:120b` vai de 0,494 / 0,657 / 0,576 para **0,500 / 0,665 / 0,582** (faithfulness / relevancy / média) — uma mudança pequena, que não altera a conclusão de empate, mas mostra que parte da métrica depende de formatação e não de conteúdo.

---

## 4. Comparação qualitativa

| Critério | `gemma4:31b` | `gpt-oss:120b` |
|---|---|---|
| Respostas com citação no formato pedido `[Fonte: documento, pág. X]` | **34 de 34** (100%) | 27 de 35 (77%) |
| Citação em formato diferente do pedido | 0 | 8 de 35: 6 com `【Fonte: …】` e 2 com `【Trecho 3 — documento, pág. 1】` |
| Recusas ("Não encontrei essa informação…") | 5 | 4 |
| Respostas truncadas | 0 | 1 (ver seção 5) |
| Erro factual verificado | nenhum encontrado | 1 (ano de fundação) |
| Caracteres tipográficos atípicos | 0 de 39 respostas | 34 de 39 respostas |

Todas as respostas não-recusa dos dois modelos trazem algum tipo de citação de fonte; a diferença é de **aderência ao formato** que a interface e a avaliação esperam.

---

## 5. Casos concretos (verificados contra os PDFs)

1. **Ano de fundação — erro com citação (`gpt-oss:120b`).** Resposta: "A GoodWe foi fundada em **2021**. [Fonte: GoodWe_Portfolio.pdf, pág. 3]". O Portfólio pág. 3 registra **2010 — Fundação**. O trecho que o retriever entregou ao modelo continha "2021 — Fundada GoodWe USA e Japão", que se refere à abertura de unidades nos EUA e no Japão. O `gemma4:31b` recusou (o chunk certo não foi recuperado). Resultado: **recusar foi correto; responder com segurança foi errado**. É o tipo de erro que a rubrica lexical não pega, porque "fundada" e "2021" estão no contexto.
2. **Lista de modelos HCA G2 — recusa indevida (`gpt-oss:120b`).** Perguntado quais modelos da linha HCA G2 aparecem no portfólio, recusou, embora o contexto recuperado (Manual pág. 1) traga "Modelos cobertos: … GW7K-HCA-20, GW11K-HCA-20 e GW22K-HCA-20". O `gemma4:31b` respondeu corretamente e citou a página.
3. **Métodos de início da recarga — comportamentos opostos.** O `gemma4:31b` recusou; o `gpt-oss:120b` respondeu parcialmente (SEMS Portal, Bluetooth), mas a resposta ficou **truncada em 282 caracteres** ("…o usuário pode"). A hipótese é que o raciocínio interno do `gpt-oss` consome parte do limite de 512 tokens (`max_tokens`); isso **não foi confirmado** e deve ser testado aumentando `max_tokens`.
4. **Perguntas numéricas institucionais** (países, capacidade instalada, funcionários): os dois modelos acertaram (mais de 100 países; 100 GW; ≈5.000 funcionários, conforme Portfólio págs. 6 e 9).

Recusas que **ambos** os modelos deram — cabo AC especificado, aplicativo de configuração local (SolarGo) e LED verde piscando — são falhas de **recuperação** (a informação existe nos PDFs, mas o chunk certo não chegou ao prompt). Isso não depende do modelo.

---

## 6. Conclusão e recomendação

- **Pelos scores, empate.** Média 0,571 × 0,576, com intervalos de confiança que incluem zero.
- **Pelo comportamento, vantagem do `gemma4:31b`** para este caso de uso (grounding + citação obrigatória): formato de citação consistente, nenhuma resposta truncada, nenhum erro factual encontrado e preferência por recusar quando o contexto é insuficiente. A latência é equivalente (1,03 s × 1,09 s em média).
- **Pontos a favor do `gpt-oss:120b`:** maior answer relevancy (0,657 × 0,612) e resposta parcial útil em um caso em que o outro recusou — mas com o erro do ano de fundação e o formato de citação instável.
- **Recomendação:** adotar `gemma4:31b` como `OLLAMA_MODEL` padrão e manter `gpt-oss:120b` como segundo modelo de comparação (é o modelo usado nos notebooks das Aulas 06–08).

**Limites desta conclusão:** 39 perguntas, sem gabarito; métrica lexical (não é RAGAS); verificação manual de correção feita apenas em casos selecionados (seção 5); uma única execução por modelo. Para sustentar a escolha com mais força, repetir a comparação com o RAGAS ou com um *LLM-as-judge* (Aula 07, slide 14), com `max_tokens` maior para o `gpt-oss`, e depois de corrigir os pontos de recuperação descritos em `relatorio_rag.md`.

### Sobre o bônus de multi-provider
O enunciado dá +1 ponto para chamada com **mais de um modelo e mais de um prompt**. O repositório compara dois **modelos** (`compare_models.py`, mesmo prompt, mesmo provedor) e compara duas **versões de prompt** em outro fluxo (`iteration_runner.py`, um modelo só). Não há uma chamada única que combine as duas dimensões, então **este relatório não reivindica o bônus**.