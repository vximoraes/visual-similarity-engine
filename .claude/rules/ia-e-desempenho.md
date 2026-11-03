---
description: Regras de CLIP, vetores, k-means e desempenho.
---

# IA e desempenho

- Modelo: `clip.load(CLIP_MODELO, jit=False)`, uma vez por processo (`app/clip_model.carregar`). Nunca
  carregue o modelo por requisição ou por mensagem.
- Todo vetor gravado ou comparado é **normalizado (L2 = 1)** (`normalizar`). Distância = cosseno
  (`<=>`, `vector_cosine_ops`). Mudar o modelo ou a dimensão (512) exige migração e reprocessar tudo: a
  coluna `modelo` registra qual gerou cada vetor.
- Inferência sempre em **lote** e dentro de `torch.inference_mode()`, em `asyncio.to_thread`.
- Entrada da imagem: a thumbnail 224x224 do simila-imagens (center crop, tamanho de treino do CLIP).
- Busca de semelhantes = KNN no espaço inteiro (HNSW). Cluster é categoria de navegação, não filtro da busca.
- k-means: `app/clusterizacao.py` fica **sem banco** (testável isolado); o job `app/jobs/clusterizar.py` faz
  I/O e troca a execução ativa numa transação, com trava `pg_try_advisory_xact_lock`.
- Cache Redis é opcional: toda operação de cache captura a falha e segue sem ele. Invalide `similares:<id>`
  quando o vetor da imagem muda ou é apagado.
- Ajustes de desempenho por env (`LOTE_MAX`, `LOTE_ESPERA_MS`, `TORCH_THREADS`, `HNSW_EF_SEARCH`, TTLs), não
  por constante no código.
