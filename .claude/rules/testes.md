# Testes

Integração com dependências reais, **sem mock** (regra do grupo): Postgres + pgvector e Redis do
`docker compose`, RabbitMQ na 5672 (o do simila-imagens ou `--profile broker`) e o CLIP de verdade.

```bash
docker compose up -d
make migrate
make test
```

- `tests/conftest.py` sobe por HTTP um JWKS (`assinar()` gera tokens iguais aos do Auth) e um servidor de
  mídia (`publicar_thumb()`), e limpa as tabelas no início da sessão.
- API: fixture `cliente` (app real com lifespan, `httpx.AsyncClient` + `ASGITransport`) no mesmo event loop
  da sessão, para o pool do SQLAlchemy não cruzar loops.
- Fluxo: `tests/test_fluxo.py` roda o worker real (`principal()`) e publica eventos no broker.
- Algoritmo: `tests/test_clusterizacao.py` testa `app/clusterizacao.py` com vetores sintéticos (rápido).
- Imagens de teste geradas na hora (`imagem_jpeg`). Imagens reais de exemplo ficam em `assets/` (não versionado).
- `make check` limpo antes de considerar pronto.
