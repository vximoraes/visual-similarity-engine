# Banco vetorial

PostgreSQL 18 + extensão **pgvector**, database exclusiva (`vse`). Modelos em `app/models.py`; migrações
Alembic em `alembic/versions/`, aplicadas por comando explícito (`make migrate`).

```mermaid
erDiagram
  execucoes_clusterizacao ||--o{ clusters : "gera"
  clusters ||--o{ embeddings_imagem : "classifica"

  execucoes_clusterizacao {
    uuid id PK "uuidv7()"
    text modelo "ViT-B/32"
    text algoritmo "minibatch-kmeans-esferico"
    int k
    int amostras
    float silhueta "cosseno, -1 a 1"
    text status "executando | pronta | falhou"
    text motivo_falha
    bool ativa "índice único parcial: no máximo uma"
    timestamptz criado_em
    timestamptz ativado_em
  }

  clusters {
    uuid id PK
    uuid execucao_id FK
    int indice
    vector_512 centroide "normalizado"
    text rotulo "zero-shot do CLIP"
    int tamanho
    uuid imagem_representativa_id "mais perto do centroide"
  }

  embeddings_imagem {
    uuid imagem_id PK "id do simila-imagens"
    uuid dono_id
    vector_512 embedding "normalizado; índice HNSW cosseno"
    text modelo
    uuid cluster_id FK "null antes do primeiro k-means"
    float distancia_centroide "1 - cosseno"
    timestamptz criado_em
    timestamptz atualizado_em
  }

  imagens_excluidas {
    uuid imagem_id PK "tombstone"
    timestamptz excluida_em
  }
```

Índices:

- `embeddings_imagem_hnsw`: `hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)`, usado
  pelo `ORDER BY embedding <=> :vetor LIMIT k`.
- `embeddings_imagem_cluster (cluster_id, distancia_centroide, imagem_id)`: listagem paginada por categoria.
- `execucoes_uma_ativa`: único parcial em `ativa` (`WHERE ativa`).

Execuções antigas e seus clusters ficam como histórico; apagar uma execução apaga seus clusters
(`CASCADE`) e zera o `cluster_id` das imagens (`SET NULL`).
