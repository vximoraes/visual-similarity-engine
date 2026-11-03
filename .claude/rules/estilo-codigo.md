---
description: Estilo de código. Sempre ativo.
---

# Estilo de código

- **Híbrido PT/EN**: domínio, regra de negócio, comentários, docstrings, mensagens de erro, docs e logs em
  português (`embedding_imagem`, `vizinhos`, `clusterizar`, `execucao_ativa`). Convenções da área ficam em
  inglês, sem traduzir: módulos e pastas de camada (`routes/`, `repository.py`, `models.py`, `jobs/`,
  `worker.py`, `config.py`), nomes impostos por contrato externo (`imageId`, `ownerId`, `thumbnailUrl`,
  `nextCursor`; claims do JWT; APIs de bibliotecas) e termos técnicos (`embedding`, `thumbnail`, `prefetch`,
  `payload`).
- Tipagem completa; `make typecheck` (mypy strict) limpo. Sem `Any` sem motivo; `type: ignore` só com o
  código do erro e quando a biblioteca não tem tipos.
- ruff (lint + format, 100 colunas). Rode `make format` antes de considerar pronto.
- **Poucos comentários**: nomes que se explicam. Docstring curta só para explicar o porquê ou uma regra não
  óbvia. Nunca o que o código já diz.
- Sem travessão (—) em código, comentários, commits e docs.
- I/O assíncrono (FastAPI, aio-pika, SQLAlchemy async, httpx, redis.asyncio). CPU pesada (CLIP, k-means)
  sempre em `asyncio.to_thread`, nunca direto no event loop.
