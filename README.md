# RAG simples utilizando LangChain, NVidia e Cloudflare Workers AI

## Modelos utilizados

- Embedding - nvidia/nemotron-3-embed-1b
- Chat - @cf/meta/llama-3.1-8b-instruct

_"Porque provedores e modelos diferentes?" Porque não?_

Os volumes do compose apontam para o index gerado pelo FAISS para que o mesmo seja persistido entre reinicializações do container.

FAISS foi utilizado por ser simples de instalar e usar, mas pode ser substituído por outro indexador de vetores, como Milvus, Weaviate, Qdrant, etc.

### Ingestão:
Documento/s -> Chunking -> Embedding -> DB (FAISS) 

### Recuperação:
Query -> Embedding da query -> Busca no DB (FAISS) -> Chat LLM -> Resposta

## Como usar:
1. Clone o repositório
2. Ingestão `docker compose run --rm rag-api python -m app.ingest`
3. Subir a API `docker compose up --build`
4. Testar: ```curl -X POST http://localhost:5100/query \
-H "Content-Type: application/json" \
-d '{"question": "Qual é a política de reembolso?"}'```

Para obter o token do Cloudflare Workers AI:
Acessar sua conta CF > Workers AI > API REST (Usando o API Workers AI REST)

#### Créditos: https://dev.to/gautamvhavle/building-production-rag-systems-from-zero-to-hero-2f1i