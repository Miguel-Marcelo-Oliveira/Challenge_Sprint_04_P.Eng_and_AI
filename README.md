# Challenge Sprint 04: ChargeGrid AI Assistant — GoodWe
**Chatbot com IA para mobilidade elétrica — Orquestrando o futuro com dados, IoT e IA.**

Este projeto é o resultado do Challenge Integrador da Sprint 04 da disciplina de *Prompt and Artificial Intelligence* (FIAP 2026.2), desenvolvido em parceria com a **GoodWe Brasil**. O objetivo é elevar a capacidade de atendimento técnico sobre a linha de carregadores elétricos da GoodWe através de um pipeline de **RAG (Retrieval-Augmented Generation)** medido, confiável e com grounding rigoroso.

## Equipe
| Nome                                        | RM     |
|:--------------------------------------------|:-------|
| Miguel Marcelo Alves Ramos de Oliveira      | 569467 |
| Felipe de Oliveira Doern                    | 568798 |
| Tom Stringasci Albuquerque Coelho de Morais | 568844 |
| Eric dos Santos Mendes da Silva             | 569528 |
| Ligia de Andrade Matheus                    | 568973 |

---

## Funcionalidades Principais

- **RAG End-to-End**: Recuperação de informações em tempo real de manuais técnicos e FAQs da GoodWe.
- **Grounding Rigoroso**: O assistente responde estritamente com base no contexto recuperado, evitando alucinações e citando a fonte (documento e página) de cada afirmação factual.
- **Segurança e Guardrails**: 
    - Filtros anti-jailbreak para evitar a revelação do prompt sistêmico.
    - Validador de escopo para recusa de perguntas irrelevantes.
    - Alertas de segurança para consultas sobre instalações elétricas perigosas.
- **Avaliação Quantitativa**: Implementação de métricas de *Faithfulness* e *Answer Relevancy* (via fallback manual do RAGAS) para medir a qualidade das respostas entre diferentes iterações de prompt e chunking.
- **Interface Web**: Interface intuitiva construída com **Gradio**.

---

## Stack Técnica

- **LLM**: Ollama Cloud (Modelos GPT-OSS / Gemma)
- **Embeddings**: `nomic-embed-text` (via Ollama Local)
- **Vector Store**: ChromaDB (Persistente)
- **Orquestração**: LangChain
- **Interface**: Gradio
- **Document Loader**: PyMuPDF
- **Linguagem**: Python 3.11

---

## Estrutura do Projeto

```text
├── app/                    # Interface web (Gradio)
├── data/
│   ├── chroma_db/          # Banco de vetores persistente
│   └── knowledge_base/      # PDFs de manuais e FAQs da GoodWe
├── docs/                   # Relatórios de evolução, modelos e RAG (PDF/MD)
├── evals/                  # Scripts de medição e resultados (JSON)
├── src/
│   ├── guardrails/         # Lógica de moderação e validação de escopo
│   └── rag/                # Pipeline RAG (loader, chunking, retriever, etc.)
└── .env                    # Variáveis de ambiente (API Keys e Configs)
```

---

## Como Executar

### 1. Pré-requisitos
- Python 3.11+
- Ollama instalado e rodando localmente (para embeddings)
- Chave de API do Ollama Cloud

### 2. Configuração
Clone o repositório e instale as dependências:
```bash
pip install -r requirements.txt
```

Crie um arquivo `.env` na raiz do projeto com as seguintes variáveis:
```env
OLLAMA_API_KEY=seu_token_aqui
OLLAMA_LOCAL_HOST=http://localhost:11434
OLLAMA_MODEL=gpt-oss:120b
EMBEDDING_MODEL=nomic-embed-text
GRADIO_SHARE=false
```

### 3. Execução da Interface
Para iniciar o assistente:
```bash
python app/app.py
```

---

## Métricas de Desempenho
O sistema passou por iterações de prompt (v1.0 $\rightarrow$ v2.0) e ajustes de chunking (800/100), atingindo estabilidade em:
- **Faithfulness**: $\approx$ 0,53 (Lexical) / $\approx$ 0,82 (Semântico nas respondidas)
- **Answer Relevancy**: $\approx$ 0,60
- **Citação de Fontes**: 100% de presença nas respostas factuais.
