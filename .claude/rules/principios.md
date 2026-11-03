---
description: Princípios de engenharia para qualquer mudança.
---

# Princípios

- **KISS e YAGNI vencem SOLID** em conflito. A solução mais simples que resolve o pedido; nenhuma camada
  "para o futuro" (repository/service genérico, provider abstrato) sem dois usos reais.
- **DRY com critério**: lógica repetida em 2+ rotas ou consumidores vira função em `app/` (ex.:
  `repository.py`, `cache.py`). Duplicação incidental (parecida por acaso) não é violação.
- **Uma responsabilidade por função/rota/consumidor.** Handler que mistura parsing, regra e resposta é sinal
  para extrair.
- **O contrato do grupo manda** (README e `decisoes-e-achados.md` do simila-api-gateway): rotas em
  `/api/similar`, envelope `{ data, errors }`, eventos, filas, JWT. Divergir exige alinhar com o grupo e
  registrar em `docs/pendencias.md`.
- Mudança pequena e focada no pedido. Achou inconsistência fora do escopo? Avise antes de mexer.
- Leia o código antes de alterar. Preserve contratos públicos (schema de resposta, payload de evento).
