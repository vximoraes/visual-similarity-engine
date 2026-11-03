---
description: Segurança como requisito funcional. Sempre ativa.
---

# Segurança

- JWT do Auth: `EdDSA` fixo, `iss`, `aud: "simila"`, `exp`, `sub` obrigatórios (`app/auth.py`). Não afrouxe
  para testar: os testes assinam com um JWKS próprio (`tests/conftest.py`).
- O worker baixa imagens de URLs que vêm no evento: só `http(s)`, sem seguir redirect, timeout de 10 s.
  Em produção a URL deve ser do bucket de mídia (`MEDIA_BASE_URL`); não amplie isso para URLs arbitrárias
  vindas do usuário (SSRF).
- SQL só pelo SQLAlchemy com parâmetros. O único `text()` com valor interpolado é o `SET LOCAL
  hnsw.ef_search` com inteiro da configuração; nunca interpole entrada do usuário.
- Erro 500 responde `Erro interno do servidor`; o detalhe fica só no log.
- Segredos só em variável de ambiente; `.env` no `.gitignore`.
- Dependências fixas no `pyproject.toml` + `uv.lock`; o CLIP fixo por commit (`tool.uv.sources`).
