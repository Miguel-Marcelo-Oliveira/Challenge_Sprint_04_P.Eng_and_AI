# EV Challenge — GoodWe

**Chatbot com IA para mobilidade elétrica — orquestrando o futuro com dados, IoT e IA.**

**FIAP × GoodWe Brasil**  
Disciplina: Prompt and Artificial Intelligence · Ciência da Computação — 2º Semestre (2SEM) · 2026.2

**Challenge Integrador · Sprint 04**

| Informação       | Detalhe                                                                    |
|------------------|----------------------------------------------------------------------------|
| **Disciplina**   | Prompt and Artificial Intelligence                                         |
| **Curso**        | Ciência da Computação — 2º Semestre (2SEM)                                 |
| **Parceiro**     | GoodWe Brasil                                                              |
| **Período**      | Semestre 2026.2                                                            |
| **Professor**    | Jorge Luiz Gomes                                                           |
| **Formato**      | Trabalho em grupo · Sprint 04 · 100 pts + bônus                            |
| **Entrega**      | 23/10/2026 · repositório público + interface + relatório de evolução (PDF) |
| **Continuidade** | Evolução do projeto das Sprints 1, 2 e 3                                   |

---

## 1. Contexto e premissa

O grupo já entregou a Sprint 1 (exploração e planejamento), a Sprint 2 (desenvolvimento e entrega) e a Sprint 03 (refactory conversacional em LangChain). A Sprint 04 continua o mesmo projeto — agora com a stack do Módulo 2 (Aulas 05–07): embeddings, ChromaDB, pipeline RAG e avaliação com RAGAS — mais a interface web obrigatória (Aula 08).

> **Projeto local**  
> Os projetos dos alunos são locais, não notebooks. A evolução acontece dentro do projeto existente, criando-se apenas as pastas indicadas neste documento — o grupo é livre para evoluir o código como preferir.

> **Relação com os CKPs**  
> O CKP02 "DocMind RAG" (21/09) é o trampolim individual para o RAG; a Sprint 04 é o projeto obrigatório em grupo que vai além. Sprints e CKPs convivem — a entrega da Sprint 04 (23/10) é distinta do CKP03 (26/10), que é o agente integrador.

## 2. Conceito — RAG medido, confiável e utilizável

Elevar o RAG do projeto a um estado medido, confiável e utilizável: ChromaDB persistente + embeddings `nomic-embed-text`, grounding com citação de fonte, avaliação RAGAS (`faithfulness` e `answer_relevancy` — com fallback manual aceito, documentado) e interface web.

A entrega central: relatório de avaliação com números + relatório de evolução do projeto (até 5 páginas) com tabela antes/depois + interface funcionando.

## 3. Escopo — o que é exigido (M2 + Aula 08)

| # | Item | Detalhe | Origem |
|---:|---|---|---|
| 1 | Vetorização | Base expandida do domínio em ChromaDB persistente + `nomic-embed-text` (Ollama Cloud). | Aula 05 · 31/08 |
| 2 | Pipeline RAG end-to-end | `PyMuPDFLoader` → `RecursiveCharacterTextSplitter` → retriever → prompt RAG versionado. | Aula 06 · 14/09 |
| 3 | Grounding + citação | Responder apenas com o contexto recuperado; citar a fonte (documento/seção) em toda resposta. | Aula 06 |
| 4 | Avaliação | RAGAS (`faithfulness`, `answer_relevancy`) ou fallback manual documentado (rubrica equivalente). | Aula 07 · 21/09 |
| 5 | Interface web | Gradio/Streamlit (stack da Aula 08) ou React, consumindo o RAG com citação visível. | Aula 08 · 28/09 |
| 6 | Segurança | Recusa fora do contexto; proteção contra prompt injection via documentos. | Aula 07 |
| 7 | Relatório de evolução | Até 5 páginas, com tabela de comparativo antes/depois (ver § 8). | — |

**Não exigir:** LangGraph, agentes, reranking, SemanticChunker/ParentDocument e observabilidade (LangSmith/LangFuse) — são diferenciais reconhecidos.

**Bônus (+1 pt):** chamada multi-provider (mais de um modelo e mais de um prompt).

## 4. Mapeamento Módulo 2 (+ Aula 08) → Sprint 04

| Aula | Conteúdo | Uso na Sprint |
|---|---|---|
| 05 · 31/08 | Embeddings e Busca Semântica com ChromaDB | Vetorizar a base expandida em ChromaDB persistente. |
| 06 · 14/09 | Pipeline RAG Completo | Pipeline end-to-end + grounding/citação de fonte. |
| 07 · 21/09 | RAG Avançado: Chunking, Reranking e RAGAS; CKP02 | Estratégia de chunking documentada; avaliação RAGAS ou fallback. |
| 08 · 28/09 | Interfaces Gradio/Streamlit + Deploy (M3) | Interface obrigatória consumindo o RAG, com citação visível. |

## 5. Pastas a criar no projeto local existente

- `data/knowledge_base/` → base expandida: manuais de produto GoodWe (ChargeGrid / EV ChargeOps), regimentos condominiais de carregamento compartilhado, FAQs de carregamento e tabelas tarifárias.
- `src/rag/` → `loader.py`, `chunking.py`, `embeddings.py`, `vector_store.py`, `retriever.py`, `prompt_rag.py` (versionado).
- `app/` → interface Gradio/Streamlit/React + README de execução.
- `evals/` → scores RAGAS (ou fallback) por iteração (mínimo 2 iterações com ganho).
- `docs/` → relatório de evolução do projeto (PDF, até 5 pág.) + `relatorio_rag.md` + `relatorio_modelos.md`.

## 6. Pontos obrigatórios

1. **Prompt RAG versionado** — tabela de versões com o que mudou e o ganho de score medido a cada iteração.
2. **Relatório de uso de modelos e parâmetros (`relatorio_modelos.md`)** — comparar 2+ modelos e documentar `temperature` (recomendado 0), `k` (top-k), `top_p` e `max_tokens`.
3. **Segurança e guardrails do sistema** — grounding obrigatório (responder só com o contexto), citação de fonte, recusa de pergunta fora do contexto e proteção contra prompt injection via documentos; não inventar especificações de produtos não inseridas na base.
4. **Relatório de evolução do projeto (até 5 páginas)** — estrutura em § 8.
5. **Bônus (+1 pt)** — chamada multi-provider: consultar mais de um modelo e mais de um prompt.

## 7. Rubrica — Sprint 04 (100 pts + bônus)

| Bloco | Pontos | Critérios |
|---|---:|---|
| **A — Pipeline RAG + interface** | 35 | Pipeline end-to-end; vector store persistente; base de conhecimento expandida; interface funcional com citação visível. |
| **B — Grounding, segurança e prompt versionado** | 30 | Responde só com o contexto; cita fonte; recusa fora do contexto; proteção contra injection; prompt RAG versionado com ganho. |
| **C — Relatório de modelos/parâmetros + avaliação** | 25 | Comparativo de 2+ modelos com parâmetros; RAGAS ou fallback manual documentado por iteração. |
| **D — Iteração e relatório de evolução** | 10 | ≥2 iterações com ganho de score; relatório de evolução (até 5 pág.) com tabela antes/depois. |
| **Bônus — multi-provider** | +1 | Chamada com mais de um modelo e mais de um prompt. |

## 8. Relatório de evolução do projeto (obrigatório · até 5 páginas)

Entregue em PDF junto do repositório, na pasta `docs/`. Estrutura mínima:

1. **Resumo da evolução** — RAG das Sprints 1/2 (e conversacional da Sprint 03) → RAG medido e com interface da Sprint 04.
2. **Pipeline RAG** — decisões de chunking (tamanho, overlap), base de conhecimento montada e trade-offs.
3. **Tabela de comparativo antes/depois (OBRIGATÓRIA)** — colunas "Sprints 1/2 (versão original)" × "Sprint 04 (RAG avaliado)", com no mínimo: scores por iteração (RAGAS ou fallback), qualidade do contexto recuperado e presença de citação de fonte.
4. **Problemas encontrados e soluções** — pelo menos 2, com a decisão tomada e o porquê.
5. **Equipe e divisão de trabalho** — quem fez o quê (nome, RM, tarefa principal).

> **Fallback manual do RAGAS**  
> Se o ambiente não suportar RAGAS, o grupo deve documentar uma rubrica manual equivalente (0–1) para `faithfulness` e `answer_relevancy`, aplicar no mesmo eval set a cada iteração e registrar os critérios em `evals/`. A equivalência deve ser justificada no relatório.

## 9. Cronograma

| Data | Evento |
|---|---|
| 31/08 – 21/09 | Módulo 2 (Aulas 05–07) · CKP02 em 21/09 (individual) · feriado 07/09 desloca a Aula 06 para 14/09. |
| 28/09 | Aula 08 — Interfaces Gradio/Streamlit + Deploy (M3). |
| 28/09 → 23/10 | ~4 semanas: interface + iterações de RAG com scores. |
| 23/10 | **ENTREGA SPRINT 04** · repositório + interface + relatório de evolução (PDF). |

> **Atenção à data**  
> A entrega da Sprint 04 cai 3 dias antes do CKP03 (26/10, Aula 11). São entregas distintas: o CKP03 é o agente integrador do checkpoint; a Sprint 04 é o projeto em grupo do Challenge. Planejar as semanas finais com antecedência.

## 10. Condições de entrega

- Repositório público com acesso ao professor; histórico Git com commits regulares de cada integrante.
- Sem API key no histórico — credenciais via variável de ambiente, `.env` (gitignored) ou Colab Secrets. Exposição implica penalidade.
- Entrega por link do repositório + arquivo `.txt` com nome, RM e turma de cada integrante.
- O relatório de evolução do projeto (até 5 páginas) é parte obrigatória da entrega desta sprint.

## 11. Integridade acadêmica

- Uso de IA generativa é incentivado, mas não substitui a entrega: o projeto é do grupo.
- Submeter o repositório de outro grupo ou de semestres anteriores = zero na sprint e processo disciplinar.
- Auto-plágio entre disciplinas não é permitido sem autorização.

---

**Documento oficial do Challenge — Sprint 04 · versão 1.0 · leitura obrigatória**  
Classificação: Uso Acadêmico FIAP · Prof. Jorge Luiz Gomes · FIAP · Ciência da Computação · 2026.2
