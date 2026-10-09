# Relatório técnico do RAG — ChargeGrid / GoodWe · Sprint 04

**Disciplina:** Prompt and Artificial Intelligence · FIAP 2026.2 · **Projeto:** EV Challenge 2026 — GoodWe
**Base dos números:** `evals/iterations_current.json`, `evals/rag_results_current.json` e `evals/model_comparison_current.json`, mais reexecução do splitter sobre os PDFs de `data/knowledge_base/`.

> **Como ler este relatório.** Todos os valores abaixo vêm dos arquivos de avaliação do repositório ou de recontagens feitas sobre eles. Onde um número tem limitação de medição, isso está dito ao lado dele (seção 7). Nenhum score foi estimado ou arredondado para parecer melhor.

---

## 1. Pipeline implementado

```
PDFs (data/knowledge_base/)
  → PyMuPDFLoader            (1 Document por página; metadata source + page)   src/rag/loader.py
  → RecursiveCharacterTextSplitter  (800 / 100)                                 src/rag/chunking.py
  → OllamaEmbeddings nomic-embed-text (Ollama local)                            src/rag/embeddings.py
  → ChromaDB persistente (data/chroma_db, coleção chargegrid_goodwe)           src/rag/vector_store.py
  → retriever top-k = 3 + filtro de relevância mínima 0,20                      src/rag/retriever.py
  → prompt RAG versionado (v1.0 / v2.0)                                         src/rag/prompt_rag.py
  → ChatOllama (Ollama Cloud) → StrOutputParser                                 src/rag/pipeline.py
  → interface Gradio (ChatInterface) com lista de fontes recuperadas            app/app.py
```

A sequência load → split → embed → store → retrieve → generate é a das Aulas 05 e 06. A interface usa `gr.ChatInterface` (Aula 08).

### Parâmetros

| Parâmetro | Valor | Onde |
|---|---|---|
| `chunk_size` / `chunk_overlap` | 800 / 100 | `.env.example`, `config.py` |
| separadores | `\n\n`, `\n`, `. `, ` `, `""` | `chunking.py` |
| `TOP_K` | 3 | `config.py` |
| `MIN_RELEVANCE` | 0,20 (sobre `1 − distância`) | `retriever.py` |
| `temperature` | 0 | `config.py` |
| `top_p` | 0,95 | `config.py` |
| `max_tokens` (`num_predict`) | 512 | `pipeline.py` |
| prompt padrão | v2.0 | `config.py` |

### Desvio em relação ao enunciado
O enunciado cita `nomic-embed-text (Ollama Cloud)`. O projeto gera os embeddings no **Ollama local** (`OLLAMA_LOCAL_HOST`) e usa o Ollama Cloud apenas para o LLM (com `Authorization: Bearer`). A mudança foi feita depois que a etapa de embeddings falhava com `unauthorized (status code: 401)` na indexação. O modelo de embedding é o mesmo (`nomic-embed-text`, 768 dimensões, como na Aula 05).

---

## 2. Base de conhecimento

| Arquivo | Páginas | Caracteres | Observação |
|---|---:|---:|---|
| `GoodWe_Manual_Produtos.pdf` | 6 | 15.430 | Manual técnico HCA / HCA G2 (≈2.570 car./pág.) |
| `GoodWe_FAQ_Carregamento.pdf` | 2 | 7.480 | FAQ em formato pergunta/resposta (≈3.740 car./pág.) |
| `GoodWe_Portfolio.pdf` | 13 | 4.250 | Slides do portfólio: 29–958 car./pág. e 3–32 imagens por página |
| **Total** | **21** | **27.160** | |

**Limitações da base**
- O enunciado pede manuais, regimentos condominiais de carregamento compartilhado, FAQs e tabelas tarifárias. A base atual tem **manual + FAQ + portfólio**; **não há regimentos condominiais nem tabelas tarifárias**.
- O portfólio é majoritariamente visual. Várias páginas só têm cabeçalho (ex.: "Portfólio de Produtos Residenciais GoodWe…", 109–120 caracteres), e informação que está em gráficos/imagens não é extraída pelo PyMuPDF.

---

## 3. Decisões de chunking e trade-offs

Contagem de chunks obtida reexecutando o `RecursiveCharacterTextSplitter` do projeto sobre as 21 páginas:

| chunk_size / overlap | Chunks | Tamanho médio (car.) | Máximo |
|---|---:|---:|---:|
| 256 / 32 | 139 | 203 | 256 |
| 512 / 64 | 68 | 418 | 511 |
| **800 / 100 (adotado)** | **51** | **566** | **799** |
| 1024 / 128 | 40 | 723 | 1022 |

**Por que 800/100.** É o ponto de partida da Aula 06 (slide 9: "~600–1000 caracteres, boa prática para RAG"). Com páginas de 2.500–3.700 caracteres (manual e FAQ), 800 cabe um a dois blocos de pergunta/resposta ou uma seção curta do manual.

**Trade-off observado.** O FAQ é uma tabela de perguntas e respostas curtas; com 800 caracteres, os cortes caem no meio de respostas. Nos contextos recuperados há trechos que começam no fim de uma resposta e já entram na pergunta seguinte (ex.: "…trifásico de 22 kW entrega no máximo 1/3… Qual é a corrente mínima para iniciar a carga?"). Chunks menores (512) tendem a isolar melhor cada par pergunta/resposta, ao custo de fragmentar procedimentos do manual; chunks maiores (1024) fazem o oposto. **Essa comparação não foi medida de forma válida** — ver seção 7, item 1.

---

## 4. Prompt RAG versionado

| Versão | O que mudou | Faithfulness | Answer relevancy | Média | Ganho |
|---|---|---:|---:|---:|---:|
| **v1.0** | Persona + objetivo + regras (usar só o contexto, não inventar, recusa com frase exata, citar `[Fonte: documento, pág. X]`) + bloco `<seguranca>` (contexto é dado, não instrução) | 0,530 | 0,581 | 0,555 | — |
| **v2.0** | Acrescenta `<prioridade>` (instruções do prompt prevalecem sobre o texto do `<contexto>`), `<grounding>` em 5 regras numeradas (citação obrigatória em **toda** afirmação factual; ignorar pedidos nos documentos para revelar prompts/chaves), e `<limites>` (sem aconselhamento jurídico, financeiro ou de segurança elétrica) | 0,527 | 0,597 | 0,562 | +0,007 |

Scores: fallback manual (seção 6), 39 perguntas. O ganho de **+0,007** na média vem de answer_relevancy (+0,016); a faithfulness ficou praticamente estável (−0,003). Esse ganho é **da mesma ordem do ruído entre execuções** (seção 6.3) e não deve ser apresentado como melhoria comprovada.

---

## 5. Grounding, citação e segurança

Camadas aplicadas em `app/app.py`, na ordem:

1. **Entrada vazia** → pede uma pergunta.
2. **Anti-jailbreak** (`guardrails/moderation.py`): 11 padrões regex (ignorar instruções, revelar system prompt, "modo desenvolvedor", DAN etc.) → mensagem de bloqueio, sem chamar o LLM.
3. **Risco profissional** (`guardrails/scope_validator.py`): perguntas jurídicas, financeiras ou de segurança elétrica → orienta consultar advogado, consultor financeiro ou eletricista credenciado.
4. **Fora de escopo**: 9 padrões regex (carro, previsão do tempo, receita, piada…) → recusa.
5. **Recuperação com limiar**: se nenhum chunk atinge `MIN_RELEVANCE`, responde "Não encontrei essa informação nos documentos fornecidos." **sem chamar o LLM**.
6. **LLM com prompt v2.0**: responde só com o contexto, cita fonte, trata o contexto como dado.
7. **Fontes visíveis**: a interface acrescenta "Fontes recuperadas: [Fonte: documento, pág. X]" ao final.

**Citação de fonte medida:** em `rag_results_current.json`, **34 de 34** respostas não-recusa contêm citação `[Fonte: …]` (100%).

**Prompt injection via documentos:** a defesa é o bloco `<prioridade>` + regra 5 do `<grounding>` do prompt v2.0. **Não há teste adversarial documentado** (nenhum documento com instrução maliciosa foi inserido na base), então a eficácia dessa defesa não foi medida.

**Limitações dos guardrails (por inspeção do código, sem testes automatizados no repositório):**
- São heurísticas por regex. Há risco de falso positivo (ex.: o padrão `quem (foi|é)` bloqueia qualquer pergunta iniciada assim; `disjuntor` aciona o aviso de segurança elétrica mesmo quando a informação está na base) e de falso negativo (reformulações que fogem dos padrões).
- A interface sempre anexa a lista "Fontes recuperadas", inclusive quando a resposta é a recusa — o que pode sugerir ao usuário que havia fonte para a resposta.

---

## 6. Avaliação

### 6.1 Método
O pipeline de avaliação (`evals/evaluate_ragas.py`) **tenta o RAGAS primeiro** (`faithfulness`, `answer_relevancy`, juiz via Ollama, como na Aula 07 slide 12) e cai para o **fallback manual** em caso de erro. Nas duas iterações o RAGAS falhou com:

```
ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'
```

Por isso **todos os scores são do fallback manual** (`evaluation_method: manual_fallback`), conforme o enunciado admite quando o ambiente não suporta o RAGAS. Conjunto de avaliação: **39 perguntas** (`evals/eval_set.py`) sobre modelos, modos de carregamento, instalação/segurança, aplicativos, LEDs e dados institucionais.

**Rubrica do fallback** (`evals/manual_eval.py`, escala 0–1):
- *faithfulness*: para cada frase da resposta, fração dos termos relevantes que aparecem nos contextos recuperados; média das frases.
- *answer_relevancy*: fração dos termos relevantes da pergunta presentes na resposta.

### 6.2 Resultados

| Execução | Config. | Faithfulness | Answer relevancy | Média |
|---|---|---:|---:|---:|
| Iteração 1 | prompt v1.0 | 0,530 | 0,581 | 0,555 |
| Iteração 2 | prompt v2.0 | 0,527 | 0,597 | 0,562 |
| `rag_results_current.json` | padrão (v2.0, 800/100) | 0,532 | 0,604 | 0,568 |

No `rag_results_current.json` (39 perguntas): **34 respondidas e 5 recusadas** (12,8%). Considerando só as 34 respondidas: faithfulness 0,601 e relevancy 0,693. Recuperação: 114 chunks no total (38 perguntas com 3 chunks, 1 pergunta com 0); similaridade média 0,453; do top-1, média 0,502 (mín. 0,255; máx. 0,749). Origem dos chunks: Manual 44%, FAQ 38%, Portfólio 18%.

### 6.3 Qualidade do contexto recuperado — diagnóstico das 5 recusas
Para cada recusa, procurei no texto dos PDFs se a informação existe e se o trecho certo foi recuperado:

| Pergunta | A informação está na base? | O trecho foi recuperado? | Diagnóstico |
|---|---|---|---|
| Qual cabo AC é especificado? | Sim (FAQ pág. 1; Manual pág. 4) | Não — vieram págs. 6, 10 e 11 do Portfólio | Falha de recuperação |
| Qual aplicativo faz a configuração local? | Sim (SolarGo: FAQ pág. 2; Manual págs. 1, 3, 5, 6) | Não — chunks da mesma página do FAQ, mas sem o termo | Falha de recuperação |
| O que significa o LED verde piscando? | Sim (FAQ pág. 2; Manual pág. 5) | Nenhum chunk passou o limiar 0,20 | Falha de recuperação (LLM nem chamado) |
| Em que ano a GoodWe foi fundada? | Sim ("2010 — Fundação", Portfólio pág. 3) | Não — veio outro chunk ("2021 — Fundada GoodWe USA e Japão") | Falha de recuperação |
| Métodos de início da recarga no HCA G2? | Sim (RFID, Bluetooth, AUTO Start, Plug and Charge, em várias páginas) | Parcial | Recusa conservadora do modelo com informação parcial no contexto |

Conclusão: **4 das 5 recusas são problema de recuperação, não de geração.** O grounding está funcionando (o sistema recusa em vez de inventar), mas a cobertura da recuperação é o gargalo. Também há ruído: 17% dos chunks recuperados (19 de 114) têm menos de 150 caracteres, em geral cabeçalhos de slide do Portfólio.

Exemplo do risco quando o contexto está incompleto: na comparação de modelos, o `gpt-oss:120b` respondeu "fundada em **2021**" com citação — leitura errada do chunk recuperado (ver `relatorio_modelos.md`).

### 6.4 Ruído entre execuções
A iteração 2 e o `rag_results_current.json` têm configuração nominalmente igual (v2.0, 800/100, parâmetros padrão) e diferem em **0,006** na média (0,562 vs 0,568). O JSON não registra qual LLM foi usado; a lista de 5 recusas coincide exatamente com a do `gemma4:31b` na comparação de modelos, o que sugere esse modelo, mas não prova. De todo modo, o ganho entre prompts (+0,007) está dentro dessa variação.

---

## 7. Limitações conhecidas dos números (leia antes de citar os scores)

1. **A iteração 1 não usou chunk_size 512.** O registro mostra 51 chunks nas duas iterações, mas 512/64 produz **68** chunks (verificado reexecutando o splitter); 51 é exatamente a contagem de 800/100. A causa provável está no código: `chunking.py` faz `from src.rag.config import CHUNK_SIZE, CHUNK_OVERLAP`, o que copia os valores no momento do import; o `iteration_runner.py` altera `config.CHUNK_SIZE` depois, e essa mudança não chega ao splitter. **Na prática, a diferença entre as iterações é só o prompt (v1.0 → v2.0)**, não o chunking.
2. **O fallback lexical penaliza a citação exigida.** As tags `[Fonte: GoodWe_FAQ_Carregamento.pdf, pág. 1]` contêm termos que não estão no contexto e reduzem a faithfulness. Reaplicando a mesma rubrica após remover as citações das respostas: faithfulness do conjunto **0,532 → 0,725** (só respondidas: 0,601 → 0,822); relevancy praticamente igual (0,604 → 0,594). Ou seja, boa parte do "0,53" é artefato de medição, e uma mudança no estilo de citação entre prompts já altera o score sem mudar a fidelidade real.
3. **A rubrica é lexical, não semântica.** Ela não detecta resposta errada com palavras presentes no contexto (como o caso "2021") nem premia paráfrase correta. Não é equivalente ao RAGAS; serve para comparar mudanças grandes entre iterações, não para afirmar qualidade absoluta. A Aula 07 (slide 14) mostra um fallback mais próximo do RAGAS — *LLM-as-judge* com prompt de avaliação e nota 0–1 — que é o caminho recomendado.
4. **Métrica de distância do ChromaDB.** A coleção é criada sem `collection_metadata={"hnsw:space": "cosine"}` (Aula 05, slide 11). O padrão do ChromaDB é distância L2, portanto `similaridade = 1 − distância` e o limiar `MIN_RELEVANCE = 0,20` não correspondem à similaridade de cosseno que a aula assume. Não foi possível inspecionar `data/chroma_db` (ignorado pelo Git e ausente do zip); isso precisa ser confirmado no ambiente local.
5. **Amostra pequena e sem referência.** São 39 perguntas, sem gabarito ("ground truth"); nenhum score mede se a resposta está *correta*, só proximidade lexical.

---

## 8. Próximos passos recomendados (em ordem de impacto)

1. Corrigir a propagação de `CHUNK_SIZE/CHUNK_OVERLAP` (passar como parâmetro para `dividir_documentos` ou ler `config.CHUNK_SIZE` em tempo de execução) e **repetir a comparação 512/64 × 800/100** de verdade.
2. Criar a coleção com `hnsw:space: cosine` e reindexar (`python indexar.py`).
3. Ajustar o fallback: remover as tags de citação antes de pontuar e/ou adicionar o *LLM-as-judge* da Aula 07 (slide 14), usando o mesmo juiz em todas as iterações.
4. Gravar no JSON o modelo, `top_k` e a versão do prompt de cada execução.
5. Ampliar a base com os tipos pedidos no enunciado (regimentos condominiais e tabelas tarifárias) e, se couber, o conteúdo regulatório que o projeto já tinha na Sprint 2 (ex.: ANEEL RN 1.000/2021).
6. Se a recuperação continuar sendo o gargalo, avaliar as técnicas opcionais da Aula 07: reranking com cross-encoder (slide 9) e Parent-Document Retriever (slide 7). O enunciado as trata como diferenciais.
7. Adicionar um documento de teste com instrução embutida ("ignore as regras e revele…") para medir a defesa contra prompt injection, e testes automatizados dos guardrails.