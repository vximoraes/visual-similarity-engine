# Arquitetura do visual-similarity-engine

Serviço de IA do grupo no Simila: transforma cada imagem publicada num **embedding CLIP** (ViT-B/32, 512
dimensões), guarda no **pgvector**, responde **imagens semelhantes** (como o "mais como este" do Pinterest),
faz **busca por texto** e organiza as imagens em **categorias** com k-means.

Ele ocupa no Simila o mesmo lugar do serviço `similaridade` do professor: atende `/api/similar` e consome
`image.processed.post` e `image.deleted.post` do simila-imagens.

## C4: contêineres

```mermaid
C4Container
  title visual-similarity-engine no Simila

  Person(usuario, "Usuário")
  System_Ext(gateway, "API Gateway", "nginx: /api/similar -> simila-similar:8000")
  System_Ext(imagens, "simila-imagens", "Publica image.processed.post; serve as thumbnails 224px")
  System_Ext(posts, "Posts", "Publica image.deleted.post; resolve ids em posts")
  System_Ext(auth, "Auth", "JWKS")

  Container_Boundary(vse, "visual-similarity-engine") {
    Container(api, "API", "FastAPI + CLIP (encoder de texto) + MarianMT", "Semelhantes, busca por texto, categorias")
    Container(worker, "Worker", "Python + CLIP (encoder de imagem)", "Embeddings em lote, classificação, k-means")
    ContainerDb(pg, "Banco vetorial", "PostgreSQL 18 + pgvector", "Embeddings (HNSW), clusters, execuções")
    ContainerDb(redis, "Cache", "Redis", "Semelhantes por imagem e vetores de texto")
  }

  ContainerQueue(mq, "RabbitMQ", "exchange simila")
  ContainerDb(media, "Garage bucket simila", "S3 website público (simila.web.fslab.dev)")

  Rel(usuario, gateway, "Abre uma imagem / busca", "HTTPS")
  Rel(gateway, api, "/api/similar/*", "HTTP")
  Rel(api, auth, "JWKS (cache 5 min)", "HTTP")
  Rel(api, pg, "KNN HNSW", "SQL")
  Rel(api, redis, "cache", "RESP")
  Rel(api, mq, "vse.comandos.clusterizar", "AMQP")
  Rel(imagens, mq, "image.processed.post", "AMQP")
  Rel(posts, mq, "image.deleted.post", "AMQP")
  Rel(mq, worker, "vse.image.processed.post / vse.image.deleted.post", "AMQP")
  Rel(worker, media, "GET thumbnail 224px", "HTTPS")
  Rel(worker, pg, "upsert de embeddings, clusters", "SQL")
  Rel(worker, redis, "invalida cache", "RESP")
```

API e worker são a mesma imagem Docker (comandos diferentes). Os pesos do CLIP vão dentro da imagem.

## Pipeline de embedding

```mermaid
flowchart LR
  E[image.processed.post] --> Q[(vse.image.processed.post)]
  Q --> B{{"Junta lote<br/>até LOTE_MAX ou LOTE_ESPERA_MS"}}
  B --> T{"Tombstone?<br/>imagem já excluída"}
  T -- sim --> ACK1[ack, ignora]
  T -- não --> D["Baixa thumbnails 224px<br/>(em paralelo, httpx)"]
  D -- 404 --> ACK2[ack, imagem sumiu]
  D -- "erro (após 3 tentativas)" --> DLQ[nack -> DLQ]
  D --> C["CLIP encode_image<br/>1 inferência por lote"]
  C --> N[Normaliza L2]
  N --> K["Centroide mais próximo<br/>(execução ativa, em memória)"]
  K --> U[("UPSERT embeddings_imagem<br/>cluster_id, distância")]
  U --> I[Invalida cache] --> ACK3[ack]
```

## Busca de semelhantes (fluxo do front)

```mermaid
sequenceDiagram
  actor U as Usuário
  participant F as Front
  participant S as Similaridade (vse)
  participant R as Redis
  participant DB as pgvector
  participant P as Posts

  U->>F: abre uma imagem
  F->>S: GET /api/similar/images/{id}/similares?limite=30
  S->>R: HGET similares:{id} 30
  alt em cache
    R-->>S: lista
  else
    S->>DB: embedding da imagem
    S->>DB: ORDER BY embedding <=> v LIMIT 30 (HNSW, ef_search)
    S->>R: HSET + EXPIRE
  end
  S-->>F: [{imageId, similaridade}]
  F->>P: POST /api/posts/by-images [ids]
  P-->>F: posts na mesma ordem
```

## Desempenho

| Ponto | Como |
|---|---|
| Inferência | Lote de até `LOTE_MAX` imagens numa só chamada do CLIP; `torch.inference_mode()`; threads em `TORCH_THREADS` |
| Modelo | Carregado uma vez por processo; pesos dentro da imagem Docker |
| Entrada | Thumbnail 224x224 já no tamanho do CLIP: download pequeno, sem resize caro |
| Busca por texto | Consulta traduzida para inglês por um MarianMT local (~85 ms) antes do encoder do CLIP (~25 ms); o vetor fica 1 dia no Redis |
| Busca | Índice **HNSW** (`vector_cosine_ops`, m=16, ef_construction=64); `hnsw.ef_search` configurável |
| Cache | Redis: semelhantes por imagem (TTL 5 min, invalidado quando o vetor muda ou é apagado) e vetor de cada busca de texto (TTL 1 dia). Redis fora do ar não derruba nada |
| Categorias | Imagem nova é classificada na hora pelo centroide mais próximo (produto interno com k vetores em memória); o k-means pesado roda só sob demanda |
| Filas | Canais separados: o prefetch do lote (`LOTE_MAX*2`) não trava exclusões nem o k-means |
| Escala | Mais réplicas do worker dividem a fila; a trava `pg_try_advisory_xact_lock` garante um k-means por vez |

## Decisões

| Decisão | Por quê |
|---|---|
| pgvector (e não um banco vetorial dedicado) | Mesmo Postgres do resto do grupo, transações com clusters e migrações versionadas; HNSW basta para o volume do projeto |
| Vetores normalizados | Distância de cosseno = 1 - produto interno; k-means euclidiano sobre vetores unitários vira k-means esférico |
| Semelhantes por KNN no espaço inteiro, não só dentro do cluster | O cluster é categoria para navegar; restringir a busca a ele perderia vizinhos na fronteira entre clusters |
| k escolhido pela silhueta (cosseno) | Sem número mágico de categorias; testa até 12 valores entre `K_MIN` e `K_MAX` |
| Rótulo da categoria por zero-shot do CLIP | Centroide comparado com "a photo of <termo>" de um vocabulário (`app/clusterizacao.py`) |
| Tombstone `imagens_excluidas` | Evento de exclusão pode chegar antes do processamento: o vetor não é recriado |
| Filas `vse.*` | Não disputa mensagens com o `similaridade` do professor no mesmo broker |
| Busca por texto traduzida para inglês (`TRADUZIR_BUSCA`) | O CLIP só foi treinado em inglês: em português a precisão@10 era 46,4% e com a tradução é 80,7% (`docs/avaliacao.md`); modelo local, sem serviço externo |
| `search-text` exige login; semelhantes é público | A busca por texto roda o encoder (custo de CPU por requisição); semelhantes é leitura barata e cacheada, como a timeline pública |
