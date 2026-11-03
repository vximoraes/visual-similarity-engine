---
description: Convenção do RabbitMQ do Simila. Sempre ativa ao mexer em eventos, filas ou consumidores.
---

# Mensageria

Fonte: seção 3 de `decisoes-e-achados.md` do gateway. Implementação: `app/rabbit.py` e `app/worker.py`.

- Exchange `simila` (topic, durável); DLX `simila.dlx` (direct).
- Filas deste serviço: `vse.<routing-key>`, durável, com DLQ `<fila>.dlq`. Prefixo `vse` (não `similar`) para
  não dividir mensagens com o serviço do professor no mesmo broker. Argumentos de fila **nunca mudam**.
- Ack manual. Erro = `nack(requeue=False)` (DLQ). Thumbnail que não existe mais (404) ou imagem com
  tombstone: ack e ignora.
- **Idempotência**: upsert por `imagem_id`; exclusão grava tombstone em `imagens_excluidas` e um
  `image.processed.post` atrasado não recria o vetor.
- Payload: valide `imageId`/`ownerId` (`ler_ids`) e aceite campos extras. Use `thumbnailUrl` quando vier.
- Este serviço não publica eventos do grupo. A fila interna `vse.comandos.clusterizar` é só para a rota admin
  pedir o k-means ao worker.
- Mudou fila ou evento? Atualize `docs/api.md` (seção Eventos).
