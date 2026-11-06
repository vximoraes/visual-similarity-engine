# visual-similarity-engine

Serviço de IA do grupo no [Simila](https://gitlab.fslab.dev/arquitetura-de-software/simila/simila-api-gateway)
(clone do Pinterest em microsserviços). Gera **embeddings CLIP** ([openai/CLIP](https://github.com/openai/CLIP),
ViT-B/32) das imagens publicadas, guarda no **pgvector**, responde **imagens semelhantes** e **busca por
texto**, e agrupa as imagens em **categorias com k-means** (imagens novas entram direto no centroide mais
próximo).

Atende em `/api/similar` atrás do gateway (porta `8000`) e consome `image.processed.post` e
`image.deleted.post` do [simila-imagens](https://gitlab.fslab.dev/arquitetura-de-software/simila/simila-imagens).

| Documento | Conteúdo |
|---|---|
| [docs/arquitetura.md](docs/arquitetura.md) | C4, pipeline de embedding, desempenho e decisões |
| [docs/api.md](docs/api.md) | Rotas, exemplos e eventos |
| [docs/banco.md](docs/banco.md) | Modelo do banco vetorial e índices |
| [docs/clusterizacao.md](docs/clusterizacao.md) | Como k, centroides e rótulos são escolhidos |
| [docs/avaliacao.md](docs/avaliacao.md) | Conjunto de imagens reais, importação e qualidade medida (precisão@10) |
| [docs/pendencias.md](docs/pendencias.md) | Decisões em aberto do grupo (busca por texto em português) |

## Stack

Python 3.14, **uv**, **FastAPI**, CLIP + torch (CPU), **PostgreSQL 18 + pgvector** (HNSW), SQLAlchemy 2
async + psycopg 3 + Alembic, **aio-pika** (RabbitMQ), **Redis** (cache), scikit-learn (MiniBatchKMeans),
PyJWT (JWT do Auth), ruff, mypy strict, pytest.

## Rodando

Requisitos: [uv](https://docs.astral.sh/uv/) (baixa o Python 3.14) e Docker.

```bash
cp .env.example .env
docker compose up -d          # Postgres + pgvector e Redis
# RabbitMQ: use o do simila-imagens (mesma porta 5672) para receber os eventos reais,
# ou suba um aqui: docker compose --profile broker up -d
uv sync
make migrate
make api                      # http://localhost:8000/api/similar/healthz
make worker                   # outro terminal
make clusterizar              # recalcula as categorias quando quiser
```

Na primeira execução o CLIP baixa os pesos (~340 MB) para `~/.cache/clip`.

### Com Docker

```bash
docker build -t registry.fslab.dev/lc-t-simila/simila-similar:latest .   # ~2 GB, pesos do CLIP inclusos
docker run ... simila-similar                       # API
docker run ... simila-similar python -m app.worker  # worker
docker exec <container> alembic upgrade head        # migrações
```

## Variáveis de ambiente

Documentadas no [.env.example](.env.example). Principais: `POSTGRES_URL`, `REDIS_URL`, `AMQP_URL`,
`AUTH_JWKS_URL`, `AUTH_ISSUER`, `MEDIA_BASE_URL`, `LOTE_MAX`/`LOTE_ESPERA_MS` (lote de inferência),
`HNSW_EF_SEARCH`, `K_MIN`/`K_MAX`.

## Comandos

| Comando | O que faz |
|---|---|
| `make api` / `make worker` | API com reload e worker |
| `make migrate` | `alembic upgrade head` |
| `make clusterizar` | k-means sobre todos os embeddings e troca da execução ativa |
| `make check` | ruff, mypy e testes: tem que passar antes de cada commit |
| `make format` | Formata com ruff |
| `make dataset` | Baixa o conjunto de avaliação (478 imagens reais) para `assets/` |
| `make importar` | Importa `assets/` direto no banco, sem fila nem storage; gera `assets/indice.json` |
| `make avaliar` | Precisão@10 dos semelhantes e da busca por texto; gera `assets/relatorio.html` |

Os testes usam Postgres, Redis e RabbitMQ reais e o CLIP de verdade (`docker compose up -d` e um broker
na 5672 antes de `make test`). Eles limpam as tabelas no início: depois deles, rode `make importar` de novo
se estiver usando o conjunto de avaliação.

`dataset`, `importar` e `avaliar` aceitam outra pasta com `PASTA=<pasta>` (formato
`<pasta>/<categoria>/<imagem>`). Passo a passo e resultados em [docs/avaliacao.md](docs/avaliacao.md).

## Estrutura

```
app/main.py            FastAPI: lifespan (CLIP, Redis, RabbitMQ), envelope, CORS, healthz
app/routes/            similares, clusters, admin
app/worker.py          consumidores: lote de embeddings, exclusão, pedido de k-means
app/jobs/clusterizar.py  job do k-means
app/jobs/importar.py   importa uma pasta de imagens direto no banco (avaliação)
app/jobs/avaliar.py    precisão@10 e relatório HTML
app/avaliacao.py       métricas de avaliação sem banco
app/clusterizacao.py   algoritmo (k pela silhueta, atribuição, rótulos) sem banco
app/clip_model.py      carregamento e encode de imagem/texto
app/repository.py      consultas (KNN, upsert, clusters)
app/cache.py           Redis
app/rabbit.py          convenção de filas do grupo
app/auth.py            JWT do Auth (JWKS)
alembic/               migrações
scripts/               baixar_dataset.py e montar_manifesto.py; dataset/ tem o manifesto e a revisão manual
assets/                conjunto de avaliação local (não versionado)
```
