# Relatório de Evolução do Projeto — ChargeGrid AI Assistant
**EV Challenge GoodWe | Sprint 04**
**Equipe:** [Insira Nomes/RMs aqui]

---

## 1. Resumo da Evolução
O projeto evoluiu de um assistente de chat simples para um sistema de **RAG (Retrieval-Augmented Generation) Medido e Confiável**. 

*   **Sprints 1 e 2:** Foco em planejamento e implementação de um chatbot básico com prompts iniciais.
*   **Sprint 03:** Refatoração para LangChain, introduzindo encadeamento conversacional e melhoria na gestão de memória.
*   **Sprint 04 (Atual):** Implementação de infraestrutura de dados persistente com **ChromaDB**, vetorização semântica com `nomic-embed-text`, pipeline de grounding rigoroso, interface web via **Gradio** e, crucialmente, um framework de **avaliação quantitativa** (via fallback manual do RAGAS) para medir a fidelidade e relevância das respostas.

---

## 2. Pipeline RAG e Decisões Técnicas

### 2.1 Arquitetura de Dados
O pipeline segue o fluxo: `PDF` $\rightarrow$ `PyMuPDFLoader` $\rightarrow$ `RecursiveCharacterTextSplitter` $\rightarrow$ `ChromaDB` $\rightarrow$ `Retriever (Top-K)` $\rightarrow$ `LLM`.

### 2.2 Decisões de Chunking
Após testes comparativos, adotamos a configuração **800 caracteres com overlap de 100**. 
*   **Trade-off:** Chunks menores (512) isolavam melhor as perguntas do FAQ, mas fragmentavam os procedimentos técnicos do manual. Chunks maiores (1024) mantinham a coesão do manual, mas introduziam ruído ao agrupar múltiplas perguntas do FAQ em um único contexto, confundindo o LLM. O tamanho 800 provou ser o equilíbrio ideal para a natureza mista da base (Manual + FAQ).

### 2.3 Base de Conhecimento
A base foi expandida para incluir:
*   Manuais técnicos de produtos HCA / HCA G2.
*   FAQ de carregamento veicular.
*   Portfólio institucional de produtos.

---

## 3. Tabela Comparativa Antes vs. Depois (OBRIGATÓRIA)

Esta tabela compara a versão original (Sprints 1/2) com a versão avaliada da Sprint 04.

| Critério | Sprints 1/2 (Versão Original) | Sprint 04 (RAG Avaliado) | Impacto/Ganho |
| :--- | :--- | :--- | :--- |
| **Método de Recuperação** | Conhecimento interno do LLM / Contexto fixo | Busca Semântica via ChromaDB (Vetores) | $\uparrow$ Precisão factual e redução de alucinações. |
| **Citação de Fontes** | Inexistente | Citação obrigatória `[Fonte: doc, pág. X]` | $\uparrow$ Auditabilidade e confiança do usuário. |
| **Grounding** | Baixo (LLM inventava detalhes) | Rigoroso (Recusa se não houver evidência) | $\uparrow$ Segurança; 100% de citações nas respostas. |
| **Medição de Qualidade** | Subjetiva (Testes manuais) | Quantitativa (Faithfulness / Relevancy) | $\uparrow$ Capacidade de otimização baseada em dados. |
| **Scores Médios** | N/A (Não medido) | **0,568** (Média Geral) | Base establish para melhorias futuras. |
| **Interface** | Terminal / Notebook | Interface Web (Gradio) | $\uparrow$ Usabilidade e acessibilidade. |

---

## 4. Iterações e Ganho de Performance
Foram realizadas iterações focadas no ajuste do Prompt e na configuração de recuperação:

1.  **Iteração 1 (Prompt v1.0):** Definição de persona e regras básicas de recusa.
2.  **Iteração 2 (Prompt v2.0):** Introdução de blocos de `<prioridade>` e `<grounding>` numerados. Resultou em um ganho de **+0,007** na média de relevância, estabilizando a recusa de perguntas fora de escopo.
3.  **Iteração 3 (Tuning de Recuperação):** Ajuste do limiar de relevância (`MIN_RELEVANCE = 0.20`). Identificou-se que 80% das falhas atuais são de *recuperação* (o documento existe, mas o retriever não o traz) e não de *geração*.

---

## 5. Problemas Encontrados e Soluções

**Problema 1: Alucinações em Datas Institucionais**
*   **Cenário:** O modelo respondia que a GoodWe foi fundada em 2021 (data da fundação da filial USA) em vez de 2010 (global).
*   **Solução:** Implementação do Prompt v2.0 com a instrução de "Citação obrigatória para cada afirmação factual". Isso forçou o modelo a olhar para o chunk específico, evidenciando a falha na recuperação do chunk de 2010 e evitando a resposta errada via recusa.

**Problema 2: Falha na propagação de parâmetros de Chunking**
*   **Cenário:** O `iteration_runner.py` alterava o tamanho do chunk, mas o `chunking.py` continuava usando o valor fixo do import inicial.
*   **Solução:** Refatoração da função `dividir_documentos` para aceitar parâmetros opcionais, permitindo que o motor de avaliação altere a estratégia de corte em tempo real sem reiniciar a aplicação.

---

## 6. Equipe e Divisão de Trabalho

| Nome                                        | RM     | Tarefa Principal                         |
|---------------------------------------------|--------|------------------------------------------|
| Miguel Marcelo Alves Ramos de Oliveira      | 569467 | Implementação do Pipeline RAG e ChromaDB | 
| Felipe de Oliveira Doern                    | 568798 | Engenharia de Prompt e Avaliação RAGAS   |
| Tom Stringasci Albuquerque Coelho de Morais | 568844 | Implementação da Interface Gradio        |
| Eric dos Santos Mendes da Silva             | 569528 | Documentação e Guardrails de Segurança   |
| Ligia de Andrade Matheus                    | 568973 | Curadoria da Base de Conhecimento        |
