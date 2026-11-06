# Pendências

Decisões que dependem do grupo e divergências de contrato ainda não resolvidas. Cada item diz o problema,
as opções e a recomendação. **Nada aqui está implementado.**

## Busca por texto em português

**Problema.** O encoder de texto do CLIP ViT-B/32 foi treinado só com legendas em inglês. Na base do
professor, a precisão@10 de `/api/similar/search-text` é **46,4% em português e 80,7% em inglês**. As
consultas `pássaro`, `inseto` e `réptil` acertam 0 de 10, e `móveis` e `veículo` acertam 1. Português só
funciona quando a palavra é cognata de inglês (`felino`, `reptil` sem acento). A medição e a análise dos
tokens estão em [avaliacao.md](avaliacao.md#5-resultados-busca-por-texto-em-português-e-em-inglês).

As duas opções abaixo foram **medidas na base do professor**, num experimento fora do projeto: os modelos
rodaram num ambiente temporário (`uv run --with ...`) e não foram adicionados às dependências. Elas usam as
mesmas 14 consultas de `CONSULTAS` e a mesma `avaliacao.precisao_do_texto`. A latência é a média de 20 a
50 codificações de uma consulta, em CPU.

| Encoder de texto | Português | Inglês | Latência por consulta nova | Pesos a mais |
|---|---|---|---|---|
| CLIP ViT-B/32 (atual) | 46,4% | **80,7%** | ~28 ms | 0 |
| A. Tradução local para inglês + CLIP | **80,7%** | 69,3% | ~113 ms | ~300 MB |
| A'. Média do vetor original com o traduzido | 75,7% | 72,9% | ~113 ms | ~300 MB |
| B. clip-ViT-B-32-multilingual-v1 | 66,4% | **82,9%** | ~9 ms | ~520 MB |

Nenhuma opção é a melhor nos dois idiomas: a tradução leva o português ao nível do inglês atual, mas
estraga parte das consultas que já estavam em inglês; o multilíngue mantém o inglês e melhora o português
só pela metade.

Nas duas opções, **os embeddings de imagem continuam válidos**: muda só o texto, sem migração e sem
reprocessar imagens.

### Opção A: traduzir a consulta para inglês antes do CLIP (recomendada)

Um modelo local de tradução, [`Helsinki-NLP/opus-mt-ROMANCE-en`](https://huggingface.co/Helsinki-NLP/opus-mt-ROMANCE-en)
(MarianMT), traduz a consulta e o resultado vai para o encoder de texto atual.

- **O português chega ao nível do inglês atual**: as 14 consultas foram traduzidas corretamente
  (`pássaro` → "bird", `móveis` → "furniture", `réptil` → "reptile").
- **Consulta já em inglês pode estragar**: o modelo traduz de línguas românicas e trata o inglês como se
  fosse uma delas. `clothes` vira "eyelashes" e `furniture` vira "supplies", e o inglês cai para 69,3%.
  Usar a média do vetor da consulta original com o da traduzida (A') amortece, mas perde parte do ganho
  em português (75,7% e 72,9%). Aplicar a tradução só quando a consulta não estiver em inglês exigiria um
  detector de idioma, que não foi medido.
- **Custo:** ~85 ms a mais na primeira vez que alguém busca um texto. O vetor de cada busca já fica 1 dia no
  Redis (`texto:<modelo>:<sha1>`, `app/cache.py`), então buscas repetidas não pagam de novo.
- Dependências novas: `transformers` e `sentencepiece`, no índice do torch CPU, para não puxar a build com
  CUDA e crescer a imagem Docker em ~1,5 GB. Os pesos vão dentro da imagem, como os do CLIP.
- Risco: palavra ambígua sem contexto ("manga" pode ser fruta ou parte da roupa). Não apareceu nas 14
  consultas, mas vale olhar no uso real.

### Opção B: encoder de texto multilíngue alinhado ao ViT-B/32

[`sentence-transformers/clip-ViT-B-32-multilingual-v1`](https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1)
é um DistilBERT multilíngue (50+ idiomas) treinado por destilação para gerar vetores no mesmo espaço do
encoder de texto do CLIP ViT-B/32. Ele substituiria só `ModeloClip.codificar_textos` na API.

- Mais rápido que o CLIP atual (~9 ms), mas **o português para em 66,4%**: melhora `pássaro`, `inseto` e
  `peixe` (de 0 a 30% para 90 a 100%), mas piora `felino` (90% para 50%) e não resolve `réptil` (0%).
- A chave do cache de texto teria de incluir o encoder, para não misturar vetores dos dois.

### Recomendação

**Opção A**, porque o Simila é um app em português: é a única que leva as buscas em português ao nível
das buscas em inglês de hoje (de 46,4% para 80,7%). O custo de latência fica só na primeira busca de cada
texto, e o rótulo zero-shot do k-means (`VOCABULARIO` em inglês, `app/clusterizacao.py`) não muda. O
preço é a perda nas buscas feitas em inglês. Se o grupo achar que elas serão comuns, o próximo passo é
medir um detector de idioma antes da tradução; se o grupo quiser zero risco para o inglês, a opção B é
a alternativa segura.

Se o grupo aceitar, a implementação cabe numa branch própria:

1. Traduzir a consulta em `buscar_por_texto` (`app/routes/similares.py`) antes de `codificar_textos`, em
   `asyncio.to_thread` e com o modelo carregado uma vez por processo, como o CLIP. A chave do cache de
   texto continua a mesma, porque o vetor final ainda é do encoder do CLIP.
2. Fixar as versões no `pyproject.toml`.
3. Refazer a avaliação (`make avaliar`) e atualizar este documento e o `docs/arquitetura.md`.
