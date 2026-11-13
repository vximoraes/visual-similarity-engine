# Categorias: k-means e centroides

Objetivo: cada imagem nova **entra direto numa categoria** (cluster) sem rodar o k-means de novo, e as
categorias se ajustam de tempos em tempos conforme o acervo cresce.

## 1. Treino (job `clusterizar`)

Roda sob demanda (`make clusterizar` ou `POST /api/similar/admin/clusterizacoes`, que o worker executa)
ou automaticamente, pedido pelo próprio worker (seção 3). Só recalcula se entrou ou mudou alguma imagem
desde a última execução ativa.

1. Lê todos os embeddings (vetores unitários de 512 dimensões).
2. Para até 12 valores de **k** entre `K_MIN` (10) e `min(K_MAX, n - 1)` (`K_MAX` = 40):
   `MiniBatchKMeans` (n_init=3, semente fixa) e **silhueta** com distância de cosseno (amostra de até
   5.000 pontos, porque ela é O(n²)). Acervo com menos imagens que `K_MIN` testa de 2 a n - 1.
3. Fica com o k de maior silhueta. Os centroides são **normalizados de novo**: sobre vetores unitários o
   k-means euclidiano equivale ao k-means esférico (cosseno).
4. **Rótulo** de cada cluster, por zero-shot do CLIP sobre um vocabulário de 111 termos com cara de
   Pinterest (`VOCABULARIO` em `app/clusterizacao.py`; termo em inglês, rótulo salvo em português):
   - o vetor de cada termo é a média de três frases ("a photo of {}", "a picture of {}",
     "{} aesthetic"), normalizada;
   - cada imagem vota no termo mais parecido com ela, e o cluster fica com o termo de maior fração de
     votos;
   - sem repetir: os pares (cluster, termo) são atendidos do maior para o menor, e quem perdeu o termo
     para outro cluster fica com o próximo. Só repete se o vocabulário acabar.
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
grandes (`GET /api/similar/admin/clusterizacoes/ativa` mostra os dois por cluster).

**Recálculo automático.** Depois de gravar cada lote, o worker conta as imagens novas desde a execução
ativa e publica um pedido em `vse.comandos.clusterizar` quando:

- entraram `CLUSTERIZAR_A_CADA_IMAGENS` (500) imagens novas, ou
- passaram `CLUSTERIZAR_INTERVALO_HORAS` (24) desde a execução ativa e entrou ao menos uma imagem (sem
  imagem nova o k-means daria o mesmo resultado).

O pedido sai uma vez por execução ativa: o worker só pede de novo quando a execução muda ou, se o k-means
falhar, depois de outro intervalo. Pedidos repetidos (de outra réplica ou da rota admin) são inofensivos:
a trava do Postgres e o `precisa_recalcular` descartam o que sobra. Antes do primeiro k-means só vale o
critério de quantidade.

## Por que não usar o cluster para os semelhantes

Os semelhantes vêm do KNN no espaço inteiro (HNSW). Restringir ao cluster cortaria vizinhos legítimos que
caíram do outro lado da fronteira entre dois clusters. O cluster serve para navegação por categoria
("explorar comida", "explorar praia").

## 4. Avaliação com dados reais

Base do professor (3.923 imagens do ImageNet, ver [avaliacao.md](avaliacao.md)): 14 categorias e 1.717
imagens em `outros`. **Pureza** de um cluster = fração da categoria mais comum nele (pelo
`assets/indice.json`); a pureza geral é a média ponderada pelo tamanho. **Pureza dos rótulos** = fração das
2.206 imagens avaliadas (sem `outros`) cujo cluster recebeu um rótulo da categoria delas (ex.: "filhotes" e
"cachorros" valem para cachorros, "barcos" e "caminhões" para veículos). Medido com o mesmo algoritmo do job,
em memória, sobre os embeddings importados.

### Antes: `K_MIN=2`, `K_MAX=30`, 40 termos com uma frase

| k | silhueta | pureza | rótulos |
|---|---|---|---|
| **2** | 0,112 | 44,1% | "artesanato" (55% `outros`) e "bebês" (27% cachorros): nenhum certo |

A silhueta (cosseno) em 512 dimensões fica quase plana para qualquer k a partir de 8 e só se destaca em
k = 2, que é inútil para navegar:

| k | 2 | 4 | 8 | 12 | 20 | 30 | 40 | 60 | 100 |
|---|---|---|---|---|---|---|---|---|---|
| silhueta | **0,112** | 0,091 | 0,064 | 0,066 | 0,071 | 0,069 | 0,061 | 0,072 | 0,052 |
| pureza | 44% | 49% | 56% | 59% | 62% | 70% | 70% | 72% | 77% |

### Depois: `K_MIN=10`, `K_MAX=40`, 111 termos, três frases e votação

| k | silhueta | pureza | pureza dos rótulos | clusters com rótulo certo | rótulos repetidos |
|---|---|---|---|---|---|
| 18 | 0,073 | **62,9%** | **58,7%** | 10 de 18 | nenhum |

Efeito de cada mudança no rótulo, sobre os mesmos 18 clusters:

| Rótulo | Pureza dos rótulos | Repetidos |
|---|---|---|
| Antigo: 40 termos, "a photo of", termo mais parecido com o centroide | 42,0% | sim ("artesanato" 5 vezes) |
| 111 termos, uma frase, centroide | 37,8% | sim |
| 111 termos, média das três frases, centroide | 46,3% | sim |
| Idem, sem repetir (centroide) | 41,5% | não |
| **Idem, votação das imagens, sem repetir (adotado)** | **58,7%** | não |

O vocabulário maior sozinho piora: termos genéricos ("artesanato", "bebês", "festas") atraem o centroide de
clusters misturados. A votação resolve porque cada imagem vota no termo dela, e o cluster de cachorros
pequenos vira "filhotes" em vez de perder para o outro cluster de cachorros.

Clusters do resultado (os 18, do maior para o menor):

| Rótulo | Imagens | Categoria mais comum | Pureza |
|---|---|---|---|
| insetos | 392 | insetos | 29% |
| iluminação | 310 | outros | 67% |
| cachorros | 259 | cachorros | 94% |
| instrumentos musicais | 243 | moda | 35% |
| filhotes | 242 | cachorros | 78% |
| macacos | 233 | primatas | 39% |
| bebês | 230 | outros | 63% |
| pássaros | 221 | pássaros | 77% |
| comida | 220 | comida | 62% |
| móveis | 220 | outros | 68% |
| ferramentas | 213 | outros | 81% |
| artesanato | 189 | outros | 63% |
| pequenos mamíferos | 175 | outros | 91% |
| caminhões | 171 | veículos | 60% |
| templos | 166 | arquitetura | 45% |
| computadores | 166 | outros | 87% |
| esportes | 149 | outros | 65% |
| barcos | 124 | veículos | 54% |

Limites que continuam:

- **O k depende da semente.** Como a silhueta é quase plana, outra semente escolhe outro k da faixa: com
  as sementes 42, 0, 1 e 2 saíram k = 18, 26, 10 e 18 (pureza de 63%, 66%, 57% e 67%). A faixa 10 a 40 é o
  que garante categorias navegáveis; o job usa a semente 42 fixa, então o resultado é reprodutível.
- Clusters de `outros` (ferramentas, eletrônicos, outros mamíferos) recebem rótulos que fazem sentido para
  o conteúdo deles ("computadores", "ferramentas", "pequenos mamíferos") mas não contam como acerto,
  porque `outros` não é categoria. Clusters misturados (moda com 35%) continuam com rótulo fraco.
- Tempo do k-means com as 3.923 imagens: de 6 a 14 s em CPU (12 valores de k com silhueta completa), fora
  a codificação das 333 frases do vocabulário pelo CLIP. Medir a fundo é a etapa de desempenho.
