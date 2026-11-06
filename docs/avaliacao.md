# Avaliação com imagens reais

Até aqui o engine só tinha sido testado com quadrados coloridos. Este documento descreve o conjunto de
imagens reais, como carregá-lo no banco e quanto o CLIP ViT-B/32 acerta nos **semelhantes** e na **busca
por texto**. As etapas de categorias (k-means) e de desempenho usam o mesmo conjunto e os mesmos comandos.

## Passo a passo

Com o ambiente do README de pé (`docker compose up -d`, `uv sync`, `make migrate`):

```bash
make dataset     # baixa as 478 imagens do manifesto para assets/<categoria>/ (~100 MB)
make importar    # thumbnail 224 + embedding CLIP em lote + upsert no banco; gera assets/indice.json
make avaliar     # precisão@10, assets/relatorio.html e as tabelas deste documento
```

Os três aceitam outra pasta com `PASTA=<pasta>` (ex.: `make importar PASTA=/tmp/fotos`), desde que ela
siga o formato `<pasta>/<categoria>/<imagem>`.

> **O `make test` limpa as tabelas do banco de desenvolvimento** (`tests/conftest.py`). Depois de rodar os
> testes, rode `make importar` de novo. Os ids não mudam, porque vêm do `assets/indice.json`.

## 1. Conjunto de imagens

**478 imagens em 12 categorias** com cara de Pinterest, de **288 autores** diferentes, todas com licença
livre: 372 CC BY-SA, 55 CC BY, 29 CC0 e 22 em domínio público.

| Categoria | Imagens | Categoria de origem no Wikimedia Commons |
|---|---|---|
| arquitetura | 40 | Quality images of buildings, of skyscrapers |
| arte | 40 | Quality images of paintings (e subcategorias: murais, pintura cristã, miniaturas) |
| cachorros | 40 | Quality images of dogs (e subcategorias) |
| carros | 40 | Quality images of automobiles |
| comida | 40 | Quality images of food |
| decoração | 40 | Quality images of furniture, of chairs, of tables, of storage furniture |
| flores | 40 | Quality images of flowers |
| gatos | 38 | Quality images of cats (e subcategorias; só existem 38 elegíveis) |
| moda | 40 | Quality images of clothing, of shoes |
| montanhas | 40 | Quality images of mountains |
| pássaros | 40 | Quality images of birds (e subcategorias) |
| praia | 40 | Quality images of beaches |

O nome de cada pasta é o rótulo em português do `VOCABULARIO` de `app/clusterizacao.py`. Assim, o rótulo
automático do k-means pode ser comparado direto com a categoria real.

### Por que o Wikimedia Commons

- As categorias **"Quality images"** só têm fotos aprovadas na revisão de qualidade da comunidade do
  Commons: boa resolução, foco e enquadramento.
- **A licença vem por arquivo**, com autor e link. Unsplash e Pexels exigem chave de API para buscar e o
  resultado da busca muda com o tempo. COCO e Open Images não têm categorias como decoração, arte ou praia.
- Os arquivos ficam em URLs públicas do Wikimedia, sem login nem cota.

### Reprodutibilidade

O conjunto é definido pelo **manifesto versionado** `scripts/dataset/manifesto.csv`: uma linha por imagem
com categoria, arquivo, **sha256**, URL, página, autor e licença. O `make dataset`
(`scripts/baixar_dataset.py`) baixa cada linha e confere o sha256. Arquivo que já existe e está íntegro
não é baixado de novo. Num download do zero (~5 s nesta máquina), as 478 imagens bateram com o sha256 do
manifesto.

As URLs são de **miniaturas de 960 px** (`thumb.wikimedia.org`), não do original (que somaria alguns GB).
O Wikimedia pode regerar uma miniatura: a foto continua a mesma, mas os bytes mudam. Nesse caso o script
mantém o arquivo e avisa no resumo ("sha256 difere do manifesto"). Download que falha encerra com código 1.

O manifesto foi gerado por `scripts/montar_manifesto.py`, que **só precisa rodar de novo para mudar o
conjunto**. As regras de seleção dele:

1. Só JPEG, com licença CC0, domínio público, CC BY ou CC BY-SA.
2. Lado menor de pelo menos 640 px e proporção entre 1:2 e 2:1. Num panorama, o center crop de 224x224
   perderia quase toda a foto.
3. Arquivo que está em mais de uma categoria do conjunto fica de fora, porque o rótulo seria ambíguo.
4. **Revisão a olho**: as fotos escolhidas foram conferidas uma a uma e 133 foram reprovadas por não
   representarem a categoria. Exemplos: atletas em competição em "clothing", ruas cheias de casas em
   "automobiles", cadeiras de praia e assentos de estádio em "furniture", penas e mosaicos em "birds". Cada
   reprovação está em `scripts/dataset/revisao_manual.csv` com o motivo.
5. Até 40 por categoria, alternando entre autores para que a série de um fotógrafo não domine a
   categoria. A ordem vem de um hash fixo (semente 42 + pageid), então reprovar uma foto só puxa a próxima
   da fila e não embaralha o resto da seleção.

O `assets/` não é versionado: o manifesto é a fonte, e as atribuições (autor, licença, página) ficam nele.

## 2. Importação direta

`make importar` (`app/jobs/importar.py`) carrega a pasta sem passar pela fila nem pelo Garage, para não
sujar o bucket compartilhado do grupo. Para cada imagem:

1. Gera a thumbnail **igual à do simila-imagens** (`src/lib/processamento.ts`, sharp): aplica a orientação
   do EXIF, põe a transparência sobre branco, converte para sRGB pelo perfil ICC embutido (Adobe RGB,
   Display P3, CMYK), redimensiona cobrindo 224x224 com corte
   central (lanczos) e grava em JPEG qualidade 85. O CLIP recebe o que receberia de uma foto publicada
   no app, inclusive os artefatos do JPEG.
2. Calcula os embeddings em lotes de `LOTE_MAX` imagens com `app.clip_model`.
3. Classifica no centroide da clusterização ativa (como o worker) e grava com
   `repository.salvar_embeddings`. Os ids são `uuid7` e o dono é fixo:
   `00000000-0000-7000-8000-00000000a55e` (`DONO_IMPORTACAO`).
4. Grava `<pasta>/indice.json` (`id -> { arquivo, categoria }`) a cada lote.

Rodar de novo **não duplica**: os ids do índice são reaproveitados e o upsert sobrescreve. Imagem que saiu
da pasta, ou que já tinha sido importada e deixou de abrir como imagem, é apagada do banco e do índice
(sem tombstone, porque não é exclusão de post). Arquivo novo que não abre como imagem é ignorado com
aviso.

Na máquina de referência (i7-13620H, 16 threads, torch CPU), a importação das 478 imagens leva **~19 s**,
contando o carregamento do modelo (tempo total de `time make importar`). Medir desempenho a fundo é a
etapa seguinte do engine.

## 3. Método da avaliação

`make avaliar` (`app/jobs/avaliar.py`, métricas em `app/avaliacao.py`):

- **Precisão@10 dos semelhantes**: para cada imagem, a fração dos 10 vizinhos mais próximos (sem ela
  mesma) que são da mesma categoria. Mostra a média geral, a média por categoria e de que categorias vêm
  os vizinhos errados.
- **Precisão@10 da busca por texto**: para cada categoria, uma consulta em português e outra em inglês,
  codificadas pelo mesmo `codificar_textos` da rota `/api/similar/search-text`. Conta quantos dos 10
  primeiros resultados são da categoria.
- Os vizinhos são calculados de forma **exata** (cosseno) e **só dentro do conjunto**. Assim o número mede o
  modelo, sem depender de imagens de fora do conjunto (que não têm categoria) e sem a aproximação do
  HNSW. A diferença para a rota é desprezível: o HNSW (`ef_search=64`) devolveu **99,6%** dos mesmos 10
  vizinhos do cálculo exato nas 478 consultas. A medição foi feita à mão, comparando
  `repository.vizinhos` (o que a rota usa) com `avaliacao.vizinhos_exatos` para cada imagem do índice.
- `assets/relatorio.html` mostra, por categoria, a melhor e a pior consulta por imagem e as duas consultas
  de texto. Borda verde é acerto e vermelha é erro, com a similaridade de cada vizinho.

## 4. Resultados: semelhantes

**Precisão@10 geral: 83,3%** (478 imagens, 12 categorias). Um palpite aleatório acertaria ~8%.

| Categoria | Imagens | Precisão@10 | Mais confundida com (parcela dos erros) |
|---|---|---|---|
| arquitetura | 40 | 60,5% | montanhas (27,8%), arte (25,3%), decoração (24,7%) |
| moda | 40 | 61,5% | decoração (30,5%), arte (13,0%), praia (9,7%) |
| arte | 40 | 72,8% | decoração (38,5%), arquitetura (20,2%), moda (13,8%) |
| decoração | 40 | 78,2% | praia (23,0%), arquitetura (20,7%), arte (16,1%) |
| comida | 40 | 80,5% | flores (39,7%), decoração (19,2%), moda (16,7%) |
| praia | 40 | 88,0% | montanhas (66,7%), moda (12,5%), arquitetura (8,3%) |
| carros | 40 | 89,0% | decoração (43,2%), moda (22,7%), arquitetura (11,4%) |
| montanhas | 40 | 89,5% | praia (78,6%), arquitetura (19,0%), flores (2,4%) |
| cachorros | 40 | 91,2% | gatos (85,7%), montanhas (2,9%), praia (2,9%) |
| gatos | 38 | 93,2% | decoração (38,5%), arquitetura (19,2%), cachorros (15,4%) |
| flores | 40 | 97,0% | decoração (33,3%), comida (25,0%), gatos (16,7%) |
| pássaros | 40 | 98,5% | gatos (50,0%), flores (50,0%) |

### Onde erra e por quê

- **Arquitetura (60,5%)** é a categoria mais heterogênea. "Buildings" no Commons vai de arranha-céu à noite
  a igreja numa vila de montanha e alameda com neve. A foto é de **cena** e o CLIP a aproxima de outras
  cenas: céu, vale e horizonte puxam para **montanhas**; fachadas e interiores ornamentados puxam para
  **arte** e **decoração**. A pior consulta (10%) é uma alameda de árvores com neve, rotulada como prédio,
  cujos vizinhos são paisagens.
- **Moda (61,5%)** e **decoração (78,2%)** se confundem entre si. As duas têm muita foto de **objeto
  isolado em fundo neutro ou de museu** (vestido em manequim, sapato, cômoda, cadeira) e o CLIP agrupa
  pelo tipo de foto tanto quanto pelo objeto.
- **Arte (72,8%)** perde para **decoração**: pintura com cena de interior (a pior consulta, de 0%, é um
  artesão na oficina) fica perto de foto de interior com móveis. O CLIP olha o conteúdo da cena mais do
  que o estilo, e pintura em moldura dourada também lembra móvel antigo.
- **Praia e montanhas** respondem uma pela outra (67% e 79% dos erros de cada): costa rochosa, mar com
  morro ao fundo e céu ocupando metade da foto aparecem nas duas.
- **Comida → flores** (40% dos erros): close colorido de fruta ou doce em fundo desfocado se parece com
  macro de flor.
- **Cachorros → gatos** (86% dos erros): animal doméstico de pelo, no chão ou no sofá. Mesmo assim os dois
  passam de 90%.
- **Pássaros (98,5%) e flores (97,0%)** são visualmente homogêneos: fundo desfocado e um único sujeito em
  destaque.

Para o k-means (etapa seguinte): espere arquitetura, arte, decoração e moda menos separadas; praia e
montanhas podem cair num cluster só; cachorros e gatos ficam próximos.

## 5. Resultados: busca por texto em português e em inglês

Precisão@10 média: **português 72,5%** e **inglês 96,7%**.

| Categoria | Consulta PT | Precisão PT | Consulta EN | Precisão EN |
|---|---|---|---|---|
| arquitetura | arquitetura | 60,0% | architecture | 80,0% |
| arte | pintura | 90,0% | painting | 90,0% |
| cachorros | cachorro | 90,0% | dog | 100,0% |
| carros | carro | 100,0% | car | 90,0% |
| comida | comida | 90,0% | food | 100,0% |
| decoração | móveis | **0,0%** | furniture | 100,0% |
| flores | flor | 100,0% | flower | 100,0% |
| gatos | gato | 100,0% | cat | 100,0% |
| moda | roupa | 90,0% | clothes | 100,0% |
| montanhas | montanha | 50,0% | mountain | 100,0% |
| praia | praia | 100,0% | beach | 100,0% |
| pássaros | pássaro | **0,0%** | bird | 100,0% |

As consultas descrevem o conteúdo das fotos (`CONSULTAS` em `app/avaliacao.py`): a categoria "decoração"
do conjunto é de móveis, "arte" de pinturas e "moda" de roupas e sapatos.

### Por que o português falha

O encoder de texto do CLIP usa um tokenizador BPE treinado em legendas em inglês. Palavra em português
só funciona quando é **cognata** de inglês ou aparece em legendas (nomes de lugar, marcas). Fora disso, ela
vira pedaços sem sentido e o vetor cai num ponto genérico do espaço. **O acento não é a causa:** `passaro`
sem acento falha do mesmo jeito.

| Consulta | Tokens do CLIP | 3 categorias mais frequentes no top 10 |
|---|---|---|
| pássaro | `p` `á` `ss` `aro` | comida (3), arquitetura (2), gatos (1) |
| passaro | `pas` `sar` `o` | arquitetura (2), praia (2), arte (2) |
| ave | `ave` | comida (6), arte (1), moda (1) |
| foto de um pássaro | `foto` `de` `um` `p` `á` `ss` `aro` | gatos (4), carros (3), arquitetura (1) |
| móveis | `m` `ó` `ve` `is` | comida (5), arquitetura (4), praia (1) |
| cadeira | `cade` `ira` | comida (3), arquitetura (3), praia (2) |
| mesa | `mesa` | arquitetura (2), montanhas (2), praia (2) |
| decoração | `decor` `a` `ç` `ão` | **decoração (7)**, arte (1), comida (1) |
| edifício | `edi` `f` `í` `cio` | **arquitetura (9)**, praia (1) |
| gato | `gato` | **gatos (10)** |
| bird | `bird` | **pássaros (10)** |

O padrão é claro: `decoração` funciona por causa do `decor`, `edifício` por causa do `edi`, e `gato` e
`carro` aparecem em legendas em inglês. Já `pássaro`, `cadeira` e `móveis` não têm parentes em inglês e
falham por completo. Para um app em português, isso significa que a busca por texto funciona por acaso,
dependendo da palavra.

A proposta de solução, com a medição de um encoder de texto multilíngue alinhado ao ViT-B/32 (português
de 72,5% para 94,2% sem reprocessar as imagens), está em [pendencias.md](pendencias.md). A decisão é do
grupo e nada foi implementado.

## 6. Limitações

- **Uma consulta de texto por categoria e idioma**: cada número da tabela de texto anda em passos de 10% e
  depende da palavra escolhida. A média sobre 12 categorias é o número mais confiável. A investigação da
  seção 5 mostra que a diferença entre os idiomas é sistemática e não sorte de uma palavra.
- **Um rótulo por imagem**: uma foto de cachorro na praia é "cachorros"; se o vizinho for uma praia, conta
  como erro. A regra 3 da seleção e a revisão manual reduzem isso, mas não zeram.
- **As categorias do Commons não são o acervo do Pinterest**: há mais foto documental (museu, arquitetura
  histórica) e menos foto de estilo de vida. Para o MVP o conjunto basta; um acervo com fotos de usuários
  reais tende a ter mais selfies, prints e textos, que o conjunto não cobre.
- **Gatos tem 38 imagens**: é tudo o que o Commons tem elegível nessa categoria.
