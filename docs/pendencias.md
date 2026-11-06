# Pendências

Decisões que dependem do grupo e divergências de contrato ainda não resolvidas. Cada item diz o problema,
as opções e a recomendação. **Nada aqui está implementado.**

## Busca por texto em português

**Problema.** O encoder de texto do CLIP ViT-B/32 foi treinado só com legendas em inglês. No conjunto de
avaliação, a precisão@10 de `/api/similar/search-text` é **72,5% em português e 96,7% em inglês**.
`pássaro` e `móveis` acertam 0 de 10; `cadeira`, `mesa` e `passaro` (sem acento) também falham. Português
só funciona quando a palavra é cognata de inglês (`decoração`, `edifício`, `gato`). Medição e análise dos
tokens em [avaliacao.md](avaliacao.md#5-resultados-busca-por-texto-em-português-e-em-inglês).

### Opção A: traduzir a consulta para inglês antes do encoder

- **Serviço externo** (Google Translate, DeepL): custo por chamada, mais um segredo, latência de rede em
  toda busca nova e a consulta do usuário saindo para um terceiro. Se o serviço cair, a busca também cai,
  a não ser que haja um fallback.
- **Modelo local** de tradução (ex.: MarianMT `Helsinki-NLP/opus-mt-ROMANCE-en`, ~300 MB): sem terceiro,
  mas são dois modelos em sequência. Consulta curta e sem contexto traduz mal ("manga" é fruta ou parte da
  roupa?) e o erro da tradução soma com o do CLIP.

### Opção B: trocar o encoder de texto por um multilíngue alinhado ao ViT-B/32 (recomendada)

[`sentence-transformers/clip-ViT-B-32-multilingual-v1`](https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1)
é um encoder de texto (DistilBERT multilíngue, 50+ idiomas) treinado por destilação para gerar vetores
**no mesmo espaço** do encoder de texto do CLIP ViT-B/32. Só o texto muda:

- **Os embeddings de imagem continuam válidos**: nada a reprocessar, sem migração. A coluna `modelo` de
  `embeddings_imagem` continua `ViT-B/32`.
- Muda só `ModeloClip.codificar_textos` da API (`app/clip_model.py`). O rótulo zero-shot do k-means
  (`VOCABULARIO` em inglês, `app/clusterizacao.py`) pode continuar no encoder original.
- A chave do cache de texto (`texto:<modelo>:<sha1>`, `app/cache.py`) precisa passar a incluir o encoder
  de texto, para não misturar vetores dos dois.

**Medido no conjunto de avaliação**, num experimento fora do projeto (o modelo não foi adicionado às
dependências). O script carregou o modelo com `sentence-transformers` num ambiente temporário
(`uv run --with sentence-transformers`), codificou as mesmas consultas de `CONSULTAS` e calculou a
precisão com `avaliacao.precisao_do_texto`. A latência é a média de 50 codificações de uma consulta:

| Encoder de texto | Português | Inglês | Latência por consulta (CPU) |
|---|---|---|---|
| CLIP ViT-B/32 (atual) | 72,5% | 96,7% | ~22 ms |
| clip-ViT-B-32-multilingual-v1 | **94,2%** | **97,5%** | ~8 ms |

Com ele, `pássaro` vai de 0% a 100%, `móveis` de 0% a 90%, `cadeira` traz 10 de 10 em decoração e
"cachorro na praia" traz 9 cachorros. Até o inglês melhora um pouco.

**Custos:**

- Dependências novas: `sentence-transformers` e `transformers`, que precisam ficar no índice do torch CPU
  (sem puxar a build com CUDA, para a imagem Docker não crescer ~1,5 GB).
- **+~520 MB** de pesos na imagem Docker e ~500 MB a mais de memória na API.
- A primeira subida da API leva ~18 s a mais (download e leitura dos pesos, medido no experimento). Na
  imagem Docker os pesos já vão dentro, como os do CLIP.

**Recomendação:** opção B. Ela resolve o problema na raiz com a mesma latência (até menor), sem serviço
externo e sem reprocessar imagens. Se o grupo aceitar, a implementação cabe numa branch própria: trocar
`codificar_textos` na API, ajustar a chave do cache, fixar as versões no `pyproject.toml`, refazer a
avaliação (`make avaliar`) e atualizar este documento e o `docs/arquitetura.md`.
