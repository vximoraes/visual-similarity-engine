# API do visual-similarity-engine

Prefixo `/api/similar` (o gateway repassa o path inteiro). OpenAPI em `GET /api/similar/openapi.json`
(lido pelo Scalar em `/api/docs/`). Envelope do grupo: `{ data, errors: null }` /
`{ data: null, errors: [{ message, path? }] }`.

| Método | Rota | Auth | Descrição |
|---|---|---|---|
| GET | `/api/similar/images/{imageId}/similares?limite=30` | público | Imagens semelhantes (1 a 100), sem a própria |
| GET | `/api/similar/search-text?q=<texto>&limite=30` | Bearer | Busca por descrição em texto (2 a 200 caracteres), traduzida para inglês antes do CLIP |
| GET | `/api/similar/images/{imageId}` | público | Categoria (cluster) e distância ao centroide |
| GET | `/api/similar/clusters` | público | Categorias da clusterização ativa |
| GET | `/api/similar/clusters/{id}/images?limit=20&cursor=` | público | Imagens da categoria, mais típicas primeiro |
| POST | `/api/similar/admin/clusterizacoes` | Bearer admin | Agenda o recálculo das categorias (202) |
| GET | `/api/similar/healthz` | público | 200/503 `{ status, banco, rabbitmq, redis }` (Redis não derruba o status) |
| GET | `/api/similar/openapi.json` | público | Schema |

## Exemplos

```json
GET /api/similar/images/01a19f66-.../similares?limite=3
{
  "data": [
    { "imageId": "01a19f70-...", "similaridade": 0.9412 },
    { "imageId": "01a19f71-...", "similaridade": 0.9127 },
    { "imageId": "01a19f6a-...", "similaridade": 0.8853 }
  ],
  "errors": null
}
```

Os ids vão para `POST /api/posts/by-images`, que devolve os posts na mesma ordem.

```json
GET /api/similar/clusters
{
  "data": [
    { "id": "01a1...", "rotulo": "comida", "tamanho": 132, "imagemRepresentativaId": "01a1..." },
    { "id": "01a1...", "rotulo": "praia", "tamanho": 87, "imagemRepresentativaId": "01a1..." }
  ],
  "errors": null
}
```

Paginação de `/clusters/{id}/images`: `{ items: [ids], nextCursor }` (padrão do Posts); passe o
`nextCursor` como `cursor` na próxima chamada; `null` na última página.

| Status | Quando |
|---|---|
| 401 | Rota protegida sem token, token inválido ou de outra audiência |
| 403 | Rota admin com `role` diferente de `admin` |
| 404 | Imagem sem embedding (inexistente, excluída ou ainda processando); categoria fora da execução ativa |
| 422 | Parâmetro inválido (ex.: `limite=0`, `q` curto) |
| 503 | JWKS do Auth não respondeu |

## Eventos

| Fila | Routing key | Publicado por | O que faz |
|---|---|---|---|
| `vse.image.processed.post` | `image.processed.post` | simila-imagens | Gera e grava o embedding (usa `thumbnailUrl`; sem ela, monta `MEDIA_BASE_URL/<dono>/post/<id>_thumb.jpg`) |
| `vse.image.deleted.post` | `image.deleted.post` | Posts | Apaga o embedding e grava o tombstone |
| `vse.comandos.clusterizar` | (fila direta) | esta API | Roda o k-means, se houve imagens novas desde a última execução |

Todas com DLQ `<fila>.dlq` na `simila.dlx`. Payload inválido vai para a DLQ; thumbnail que não existe mais
(404) é ignorada com ack.
