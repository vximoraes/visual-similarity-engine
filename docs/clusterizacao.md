# Categorias: k-means e centroides

Objetivo: cada imagem nova **entra direto numa categoria** (cluster) sem rodar o k-means de novo, e as
categorias se ajustam de tempos em tempos conforme o acervo cresce.

## 1. Treino (job `clusterizar`)

Roda sob demanda: `make clusterizar` ou `POST /api/similar/admin/clusterizacoes` (o worker executa).
Só recalcula se entrou ou mudou alguma imagem desde a última execução ativa.

1. Lê todos os embeddings (vetores unitários de 512 dimensões).
2. Para até 12 valores de **k** entre `K_MIN` e `min(K_MAX, n - 1)`: `MiniBatchKMeans` (n_init=3,
   semente fixa) e **silhueta** com distância de cosseno (amostra de até 5.000 pontos, porque ela é O(n²)).
3. Fica com o k de maior silhueta. Os centroides são **normalizados de novo**: sobre vetores unitários o
   k-means euclidiano equivale ao k-means esférico (cosseno).
4. **Rótulo** de cada centroide: o termo mais parecido de um vocabulário (`VOCABULARIO` em
   `app/clusterizacao.py`), comparando com o texto "a photo of <termo>" pelo encoder de texto do CLIP
   (zero-shot). O termo vai em inglês, o rótulo salvo é o português.
5. **Imagem representativa**: a mais próxima do centroide (capa da categoria).
6. Numa única transação: grava clusters, reclassifica todas as imagens (`cluster_id`, `distancia_centroide`),
   desativa a execução anterior e ativa a nova. Uma trava do Postgres impede dois k-means ao mesmo tempo.

Com menos de 3 imagens (ou se tudo cair num grupo só) a execução fica `falhou` com o motivo, e a anterior
continua ativa.

## 2. Classificação online (worker)

O worker mantém os centroides da execução ativa **em memória** e confere a cada lote se a execução mudou.
Para cada imagem nova: `similaridades = vetor · centroidesᵀ`, cluster = argmax, distância = 1 - máximo.
Custo: k produtos internos de 512 dimensões, desprezível perto do CLIP.

Antes do primeiro k-means, as imagens entram sem categoria (`cluster_id` null).

## 3. Quando recalcular

O centroide fixo "envelhece" quando o acervo muda de perfil (ex.: muitas imagens de um tema novo vão para o
cluster menos ruim). Sinais para rodar de novo: distância média ao centroide subindo e clusters muito
grandes. Sugestão para o MVP: recalcular manualmente depois de cargas grandes de imagens; um agendamento
periódico pode publicar na fila `vse.comandos.clusterizar`.

## Por que não usar o cluster para os semelhantes

Os semelhantes vêm do KNN no espaço inteiro (HNSW). Restringir ao cluster cortaria vizinhos legítimos que
caíram do outro lado da fronteira entre dois clusters. O cluster serve para navegação por categoria
("explorar comida", "explorar praia").
