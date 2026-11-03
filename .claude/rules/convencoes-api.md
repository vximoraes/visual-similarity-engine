---
description: Convenções das rotas HTTP.
---

# Convenções da API

- Todas as rotas começam com `/api/similar` (o gateway repassa o path inteiro); `openapi_url` em
  `/api/similar/openapi.json`, sem Swagger próprio (o Scalar do gateway lê o schema).
- Resposta de sucesso: `Resposta[T]` (`app/routes/__init__.py`) = `{ data, errors: null }`. Erro: lance
  `HTTPException(status, mensagem)`; os handlers de `app/main.py` montam `{ data: null, errors: [...] }`.
  Validação do FastAPI vira 422 com `path`.
- Parâmetros com limites explícitos (`Query(ge=..., le=...)`, `min_length`/`max_length`).
- Rota protegida: `Depends(usuario_atual)`; só admin: `Depends(exigir_admin)` (`app/auth.py`).
- Paginação no padrão do Posts: `{ items, nextCursor }`, `nextCursor` null na última página.
- Rota nova: router em `app/routes/`, `include_router` em `app/main.py`, teste em `tests/`, linha em `docs/api.md`.
