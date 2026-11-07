# Avaliação com imagens reais

Até aqui o engine só tinha sido testado com quadrados coloridos. Este documento descreve como carregar a
**base de imagens do professor** ("Imagenet Dataset Mini") no banco e quanto o CLIP ViT-B/32 acerta nos
**semelhantes** e na **busca por texto**. As etapas de categorias (k-means) e de desempenho usam a mesma
base e os mesmos comandos.

## Passo a passo

Com o ambiente do README de pé (`docker compose up -d`, `uv sync`, `make migrate`) e o zip da base do
professor baixado (peça ao grupo; não é versionado):

```bash
make imagenet ZIP="<caminho>/Imagenenet Dataset Mini.zip"   # 3.923 imagens em assets/<categoria>/ (~1 s)
make importar    # thumbnail 224 + embedding CLIP em lote + upsert no banco (~2,5 min em CPU)
make avaliar     # precisão@10, assets/relatorio.html e as tabelas deste documento
```

O `make imagenet` confere o sha256 do zip (`193336d3…89af5`) e avisa se a base for outra. Os três comandos
aceitam outra pasta com `PASTA=<pasta>`, desde que ela siga o formato `<pasta>/<categoria>/<imagem>`.

> **O `make test` limpa as tabelas do banco de desenvolvimento** (`tests/conftest.py`). Depois de rodar os
> testes, rode `make importar` de novo. Os ids não mudam, porque vêm do `assets/indice.json`.

## 1. A base do professor

São **3.923 imagens da validação do ImageNet (ILSVRC 2012), de 1.000 classes**: de 1 a 14 imagens por
classe, 4 na mediana. Cada classe é um synset do WordNet (`n02123045` = tabby cat).

Com ~4 imagens por classe, a precisão@10 por classe não mede nada: quase nenhuma imagem tem 10 vizinhas
possíveis da mesma classe. Por isso as classes são **agrupadas em categorias com cara de Pinterest pela
própria hierarquia do WordNet**, que é a estrutura do ImageNet. Uma classe entra numa categoria quando
descende de uma das raízes dela (`dog.n.01` para cachorros, `furniture.n.01` para decoração...).

| Categoria | Classes | Imagens | Exemplos de classes |
|---|---|---|---|
| cachorros | 118 | 433 | Afghan hound, Airedale, beagle, golden retriever |
| veículos | 64 | 241 | airliner, ambulance, canoe, sports car, tractor |
| moda | 49 | 200 | abaya, bikini, cardigan, cowboy boot, kimono |
| pássaros | 59 | 197 | albatross, bald eagle, flamingo, toucan |
| comida | 51 | 184 | bagel, banana, broccoli, pizza, cheeseburger |
| arquitetura | 49 | 166 | barn, castle, church, mosque, palace, viaduct |
| decoração | 33 | 153 | bookcase, china cabinet, rocking chair, refrigerator |
| insetos | 36 | 142 | ant, bee, ladybug, monarch, tarantula |
| répteis | 36 | 121 | African chameleon, alligator lizard, green mamba |
| instrumentos | 26 | 111 | accordion, acoustic guitar, cello, grand piano |
| primatas | 20 | 94 | baboon, capuchin, chimpanzee, gorilla |
| felinos | 13 | 71 | Egyptian cat, tabby, cheetah, lion, leopard |
| peixes | 16 | 57 | anemone fish, eel, goldfish, great white shark |
| paisagens | 10 | 36 | alp, cliff, coral reef, lakeside, seashore, volcano |
| **outros** | 420 | **1.717** | abacus, African elephant, ladle, oscilloscope |

**Avaliadas: 2.206 imagens em 14 categorias.** As 1.717 de `outros` são classes sem categoria (ferramentas,
eletrônicos, outros mamíferos...). Elas **são importadas** e entram na busca como vizinhas possíveis, como
num acervo real, mas não são avaliadas como consulta. Quando aparecem entre os vizinhos de uma imagem
avaliada, contam como erro.

Regras do agrupamento (`scripts/gerar_grupos_imagenet.py`):

1. A classe vai para a primeira categoria da lista que contém um ancestral dela. Só uma classe cai em
   duas: `mobile home` é veículo e construção, e fica em veículos.
2. O WordNet classifica pela definição e não pela foto: `menu` é comida, `seat belt` é roupa, `scoreboard`
   é construção. **22 classes assim foram mandadas para `outros`** (`PARA_OUTROS`, com a lista no script).
3. Categoria com menos de 30 imagens vai inteira para `outros` (hoje nenhuma; a menor, paisagens, tem 36).

O resultado fica versionado em `scripts/imagenet/grupos.csv` (wnid, nome, categoria). O `make imagenet`
usa só esse CSV. Para mudar o agrupamento, edite o script e rode
`uv run --no-project --with nltk python scripts/gerar_grupos_imagenet.py <zip>`.

A imagem é extraída como `assets/<categoria>/<wnid>_<arquivo>.JPEG`: a classe original fica no nome do
arquivo e, portanto, no `indice.json`. As imagens do ImageNet são para uso em pesquisa e ensino, por isso
ficam só em `assets/`, que não é versionado.

## 2. Importação direta

`make importar` (`app/jobs/importar.py`) carrega a pasta sem passar pela fila nem pelo Garage, para não
sujar o bucket compartilhado do grupo. Para cada imagem:

1. Gera a thumbnail **igual à do simila-imagens** (`src/lib/processamento.ts`, sharp): aplica a orientação
   do EXIF, põe a transparência sobre branco, converte para sRGB pelo perfil ICC embutido (Adobe RGB,
   Display P3, CMYK), redimensiona cobrindo 224x224 com corte central (lanczos) e grava em JPEG
   qualidade 85. O CLIP recebe o que receberia de uma foto publicada no app, inclusive os artefatos do JPEG.
2. Calcula os embeddings em lotes de `LOTE_MAX` imagens com `app.clip_model`.
3. Classifica no centroide da clusterização ativa (como o worker) e grava com
   `repository.salvar_embeddings`. Os ids são `uuid7` e o dono é fixo:
   `00000000-0000-7000-8000-00000000a55e` (`DONO_IMPORTACAO`).
4. Grava `<pasta>/indice.json` (`id -> { arquivo, categoria }`) a cada lote.

Rodar de novo **não duplica**: os ids do índice são reaproveitados e o upsert sobrescreve. Imagem que saiu
da pasta, ou que já tinha sido importada e deixou de abrir como imagem, é apagada do banco e do índice
(sem tombstone, porque não é exclusão de post). Arquivo novo que não abre como imagem é ignorado com
aviso.

Na máquina de referência (i7-13620H, 16 threads, torch CPU), a importação das 3.923 imagens leva
**~2,5 min** (tempo total de `time make importar`). Medir desempenho a fundo é a etapa seguinte do engine.

## 3. Método da avaliação

`make avaliar` (`app/jobs/avaliar.py`, métricas em `app/avaliacao.py`):

- **Precisão@10 dos semelhantes**: para cada imagem avaliada, a fração dos 10 vizinhos mais próximos (sem
  ela mesma, entre as 3.923) que são da mesma categoria. Mostra a média geral, a média por categoria e de
  que categorias vêm os vizinhos errados.
- **Precisão@10 da busca por texto**: para cada categoria, uma consulta em português e outra em inglês,
  codificadas pelo mesmo `codificar_textos` da rota `/api/similar/search-text`. Conta quantos dos 10
  primeiros resultados, entre as 3.923, são da categoria.
- Os vizinhos são calculados de forma **exata** (cosseno). A diferença para a rota, que usa o índice HNSW,
  é desprezível: com `ef_search=64` o HNSW devolveu **99,9%** dos mesmos 10 vizinhos do cálculo exato nas
  3.923 consultas. A medição foi feita à mão, comparando `repository.vizinhos` (o que a rota usa) com
  `avaliacao.vizinhos_exatos` para cada imagem do índice.
- `assets/relatorio.html` mostra, por categoria, a melhor e a pior consulta por imagem e as duas consultas
  de texto. Borda verde é acerto e vermelha é erro, com a similaridade de cada vizinho.

## 4. Resultados: semelhantes

**Precisão@10 geral: 72,2%** (2.206 imagens avaliadas em 14 categorias, 3.923 na base).

| Categoria | Imagens | Precisão@10 | Mais confundida com (parcela dos erros) |
|---|---|---|---|
| paisagens | 36 | 38,1% | arquitetura (30,9%), outros (22,9%), veículos (18,4%) |
| arquitetura | 166 | 39,5% | outros (58,5%), veículos (11,4%), decoração (8,8%) |
| decoração | 153 | 46,7% | outros (80,1%), arquitetura (6,9%), comida (4,4%) |
| moda | 200 | 46,8% | outros (75,1%), instrumentos (6,1%), decoração (5,3%) |
| veículos | 241 | 60,7% | outros (71,3%), arquitetura (13,6%), paisagens (5,0%) |
| peixes | 57 | 62,8% | outros (65,6%), répteis (27,8%), paisagens (3,3%) |
| instrumentos | 111 | 65,5% | outros (69,7%), moda (9,9%), decoração (7,0%) |
| comida | 184 | 72,5% | outros (74,3%), arquitetura (7,5%), decoração (6,9%) |
| répteis | 121 | 78,3% | outros (55,5%), peixes (11,4%), pássaros (9,1%) |
| felinos | 71 | 82,7% | outros (56,1%), cachorros (34,1%), primatas (5,7%) |
| insetos | 142 | 84,5% | outros (63,6%), répteis (20,5%), comida (7,3%) |
| pássaros | 197 | 89,3% | outros (42,7%), primatas (25,6%), répteis (16,6%) |
| primatas | 94 | 90,7% | outros (78,2%), répteis (8,0%), pássaros (6,9%) |
| cachorros | 433 | 98,1% | outros (67,9%), felinos (22,6%), pássaros (2,4%) |

Medições complementares (feitas à mão sobre os mesmos vizinhos):

| Medida | Valor | O que mostra |
|---|---|---|
| Precisão@10 contando só os vizinhos que têm categoria | **85,6%** | Entre as categorias, o modelo quase não confunde; a maior parte do erro vem de `outros` |
| Vizinhos que são de `outros` | 18,7% | `outros` é 43,8% da base, mas aparece bem menos que isso: o modelo separa o que é do tema |
| Vizinho mais próximo da **mesma classe** do ImageNet (1.000 classes, acaso de 0,1%) | **35,1%** | Mesmo na granularidade fina (raça de cachorro, espécie de pássaro), 1 em 3 acerta no primeiro |
| Imagens da mesma classe que aparecem no top 10 | 41,3% | |

### Onde erra e por quê

- **Boa parte do "erro" é do agrupamento, não do modelo.** As classes de `outros` que mais aparecem como
  vizinhas são do mesmo tema da categoria: `vault` (abóbada), `thatch` (telhado de palha) e `bell cote` para
  arquitetura; `quilt`, `stove` e `desktop computer` para decoração; `wool`, `mailbag` e `bath towel`
  para moda; `parachute`, `car wheel`, `wing` e `grille` para veículos; `timber wolf` para
  cachorros. Para um usuário, esses vizinhos fazem sentido.
- **Arquitetura (39,5%), decoração (46,7%) e moda (46,8%)** sofrem com a natureza do ImageNet: a classe
  nomeia **um objeto**, mas a foto mostra **uma cena**. A pior consulta de `bow tie` (gravata-borboleta,
  moda) é a foto de uma criança, e os 10 vizinhos são fotos de outras crianças (com gaita, estetoscópio,
  bolhas de sabão). A de `shoe shop` (arquitetura) é uma prateleira cheia de sapatos, e os vizinhos são
  outras pilhas de objetos coloridos (chapéus, caixas, óculos de sol).
- **Paisagens (38,1%)** é a menor categoria (36 imagens), e as fotos dela costumam ter outro sujeito em
  destaque: a pior de `seashore` (beira-mar) é um cachorro na praia, com 10 cachorros de vizinhos; a de `alp`
  é um ciclista na montanha, com bicicletas e snowmobiles de vizinhos. `cliff` (falésia) puxa para
  construções de pedra (`cliff dwelling`, `megalith`), que estão em arquitetura.
- **Peixes → répteis e `outros`** (`dugong`, `sea cucumber`, `brain coral`): o fundo aquático pesa mais que
  o animal.
- **Animais vão bem**: cachorros (98,1%, 118 raças), primatas (90,7%) e pássaros (89,3%) têm o sujeito no
  centro da foto, que é o caso ideal do CLIP. Cachorros e felinos se confundem entre si: um gato
  doméstico (`tabby`) dentro de casa tem terriers pequenos entre os vizinhos.

Para o k-means (etapa seguinte): animais devem formar clusters limpos; arquitetura, decoração e moda devem
se misturar com `outros`. Os rótulos do `VOCABULARIO` não têm felinos, primatas, insetos, répteis, peixes
nem instrumentos: vale conferir se o rótulo automático acerta essas categorias.

## 5. Resultados: busca por texto em português e em inglês

A rota `/search-text` **traduz a consulta para inglês antes do CLIP** (`app/traducao.py`, ligado por
`TRADUZIR_BUSCA=true`, o padrão). A decisão veio desta avaliação:

| Busca por texto | Português | Inglês |
|---|---|---|
| CLIP puro (`TRADUZIR_BUSCA=false`) | 46,4% | **80,7%** |
| **Tradução + CLIP (padrão)** | **80,7%** | 69,3% |

Os dois números vêm do `make avaliar`, que usa o mesmo caminho da rota (`traducao.codificar_consultas`).
Para medir o CLIP puro, rode `TRADUZIR_BUSCA=false make avaliar`.

Com a tradução, o português chega ao nível que o inglês tinha. O preço é que parte das consultas já em
inglês piora, porque o tradutor as trata como se fossem de uma língua românica: `clothes` vira
"eyelashes" (de 80% para 0%) e `furniture` vira "supplies" (de 100% para 10%). Como o app é em português,
o ganho compensa. A alternativa para o inglês está em [pendencias.md](pendencias.md).

### Sem a tradução: por que o português falha

Precisão@10 do CLIP puro, consulta a consulta:

| Categoria | Consulta PT | Precisão PT | Consulta EN | Precisão EN |
|---|---|---|---|---|
| arquitetura | arquitetura | 50,0% | architecture | 30,0% |
| cachorros | cachorro | 100,0% | dog | 100,0% |
| comida | comida | 60,0% | food | 50,0% |
| decoração | móveis | 10,0% | furniture | 100,0% |
| felinos | felino | 90,0% | feline | 90,0% |
| insetos | inseto | **0,0%** | insect | 100,0% |
| instrumentos | instrumento musical | 90,0% | musical instrument | 100,0% |
| moda | roupa | 90,0% | clothes | 80,0% |
| paisagens | paisagem | 20,0% | landscape | 40,0% |
| peixes | peixe | 30,0% | fish | 90,0% |
| primatas | macaco | 100,0% | monkey | 100,0% |
| pássaros | pássaro | **0,0%** | bird | 100,0% |
| répteis | réptil | **0,0%** | reptile | 80,0% |
| veículos | veículo | 10,0% | vehicle | 70,0% |

As consultas (`CONSULTAS` em `app/avaliacao.py`) descrevem o conteúdo das fotos: "felinos" tem gatos e
grandes felinos, "decoração" tem móveis e eletrodomésticos, "primatas" tem macacos. Mesmo em inglês,
`architecture` (30%) e `landscape` (40%) perdem para `outros` do mesmo tema: `dome`, `vault`, `thatch` e
`bell cote` no primeiro; campos de `rapeseed` (canola) no segundo.

O encoder de texto do CLIP usa um tokenizador BPE treinado em legendas em inglês. Palavra em português
só funciona quando é **cognata** de inglês ou aparece em legendas. Fora disso, ela vira pedaços sem sentido
e o vetor cai num ponto genérico do espaço.

| Consulta | Tokens do CLIP | 3 categorias mais frequentes no top 10 |
|---|---|---|
| pássaro | `p` `á` `ss` `aro` | outros (3), moda (2), cachorros (2) |
| passaro | `pas` `sar` `o` | veículos (4), arquitetura (3), comida (1) |
| inseto | `in` `se` `to` | outros (6), paisagens (1), comida (1) |
| peixe | `pe` `ix` `e` | outros (3), peixes (3), comida (2) |
| móveis | `m` `ó` `ve` `is` | comida (4), outros (3), decoração (1) |
| cadeira | `cade` `ira` | comida (4), insetos (2), paisagens (1) |
| réptil | `ré` `p` `til` | outros (4), veículos (2), comida (2) |
| reptil | `rep` `til` | **répteis (9)**, outros (1) |
| macaco | `mac` `aco` | **primatas (10)** |
| felino | `fel` `ino` | **felinos (9)**, outros (1) |
| cachorro | `ca` `chor` `ro` | **cachorros (10)** |

Há dois casos:

- **A palavra não existe para o CLIP**, com ou sem acento: `pássaro` e `passaro`, `inseto`, `peixe`, `cadeira`.
- **O acento quebra um cognato**: `reptil` sem acento vira `rep`, como em *reptile*, e acerta 9 de 10;
  `réptil` com acento vira `ré` + `p` e acerta 0. Já `felino` (*feline*) e `macaco` funcionam.

Sem tradução, a busca em português funcionaria por acaso, dependendo da palavra. Com ela, as 14 consultas
viram o termo certo em inglês (`pássaro` → "bird", `móveis` → "furniture", `réptil` → "reptile").

## 6. Limitações

- **Uma consulta de texto por categoria e idioma**: cada número da tabela de texto anda em passos de 10% e
  depende da palavra escolhida. A média sobre 14 categorias é o número mais confiável. A análise de tokens
  da seção 5 mostra que a diferença entre os idiomas é sistemática e não sorte de uma palavra.
- **O rótulo é da classe do ImageNet, não da foto**: a foto de `seashore` com um cachorro em destaque conta
  como paisagem. Os vizinhos "errados" que aparecem por isso são, na verdade, acertos para quem busca.
- **O agrupamento é uma escolha**: outra divisão das 1.000 classes mudaria os números por categoria. A
  precisão contando só vizinhos com categoria (85,6%) e o acerto da classe exata (35,1%) não dependem dela.
- **O ImageNet não é o acervo do Pinterest**: as fotos são centradas num objeto, e há pouca foto de estilo
  de vida, decoração de ambientes, prints e textos.
