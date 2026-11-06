# visual-similarity-engine

Serviço de IA do grupo no Simila (clone do Pinterest em microsserviços, projeto da faculdade): embeddings
CLIP, busca de semelhantes no pgvector, busca por texto e categorias por k-means. Ocupa o lugar do serviço
`similaridade` do professor (`/api/similar`). O contrato com os outros serviços (rotas, eventos, JWT,
envelope, healthz) está no README e em `documentacao/decisoes-e-achados.md` do simila-api-gateway e manda
sobre preferências locais. O produtor dos eventos é o simila-imagens (`../simila-imagens`).

## Stack

| Camada | Escolha |
|---|---|
| Linguagem | Python 3.14, uv, tipagem estrita (mypy strict) |
| HTTP | FastAPI, envelope do grupo, Pydantic |
| IA | openai/CLIP ViT-B/32 (git), torch CPU |
| Banco | PostgreSQL 18 + pgvector (HNSW cosseno), SQLAlchemy 2 async + psycopg 3, Alembic |
| Fila | aio-pika (convenção do grupo, filas `vse.*`) |
| Cache | Redis (opcional: falha não derruba nada) |
| Cluster | scikit-learn MiniBatchKMeans + silhueta |
| Qualidade | ruff, mypy, pytest (integração real, CLIP de verdade, sem mock) |

## Regras do projeto

@.claude/rules/principios.md
@.claude/rules/estilo-codigo.md
@.claude/rules/convencoes-api.md
@.claude/rules/mensageria.md
@.claude/rules/ia-e-desempenho.md
@.claude/rules/seguranca.md
@.claude/rules/testes.md
@.claude/rules/git.md

## Comandos

```bash
docker compose up -d     # pgvector + Redis (broker: o do simila-imagens ou --profile broker)
uv sync
make migrate
make api                 # :8000
make worker
make clusterizar
make check               # ruff + mypy + pytest
make dataset && make importar && make avaliar   # imagens reais e precisão@10 (docs/avaliacao.md)
uv run alembic revision -m "<descricao>"   # nova migração (escrita à mão, como a 0001)
```

## Documentação

`docs/` (arquitetura, API, banco, clusterização, avaliação, pendências). Mudou rota, evento, tabela ou
algoritmo? Atualize o documento correspondente na mesma alteração.
