# Pendências

Decisões que dependem do grupo e divergências de contrato ainda não resolvidas. Cada item diz o problema,
as opções e a recomendação.

## Busca por texto em consultas em inglês

**Contexto.** O encoder de texto do CLIP ViT-B/32 só foi treinado com legendas em inglês, e a busca em
português acertava 46,4% na base do professor. **Decidido e implementado:** a rota `/search-text` traduz a
consulta para inglês com um modelo local antes do CLIP (`app/traducao.py`, `TRADUZIR_BUSCA=true`). O
português subiu para 80,7%. Detalhes em [avaliacao.md](avaliacao.md#5-resultados-busca-por-texto-em-português-e-em-inglês).

**O que ficou em aberto.** O tradutor (`Helsinki-NLP/opus-mt-ROMANCE-en`) trata toda consulta como se
fosse de uma língua românica, e parte das que já estão em inglês estraga: `clothes` vira "eyelashes" e
`furniture` vira "supplies". O inglês caiu de 80,7% para 69,3%.

As alternativas abaixo foram medidas na base do professor, num experimento fora do projeto (a opção B
rodou num ambiente temporário, `uv run --with sentence-transformers`). Todas usam as 14 consultas de
`CONSULTAS` e a mesma `avaliacao.precisao_do_texto`. A latência é a média de 20 a 50 codificações de
uma consulta, em CPU.

| Busca por texto | Português | Inglês | Latência por consulta nova | Pesos a mais |
|---|---|---|---|---|
| CLIP puro (`TRADUZIR_BUSCA=false`) | 46,4% | **80,7%** | ~28 ms | 0 |
| **Tradução + CLIP (atual)** | **80,7%** | 69,3% | ~113 ms | ~300 MB |
| Média do vetor original com o traduzido | 75,7% | 72,9% | ~113 ms | ~300 MB |
| B. `clip-ViT-B-32-multilingual-v1` | 66,4% | **82,9%** | ~9 ms | ~520 MB |

Nenhuma opção é a melhor nos dois idiomas. A latência a mais só pesa na primeira busca de cada texto: o
vetor fica 1 dia no Redis.

**Opções para o inglês:**

1. **Deixar como está.** O Simila é um app em português, e o inglês de 69,3% ainda é maior que o
   português de antes (46,4%).
2. **Detector de idioma antes da tradução**: traduzir só o que não estiver em inglês. Em tese junta os
   80,7% do português com os 80,7% do inglês, mas consulta curta ("gato", "dog") é difícil de detectar e
   isso **não foi medido**.
3. **Trocar pela opção B**: um encoder multilíngue
   ([`sentence-transformers/clip-ViT-B-32-multilingual-v1`](https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1))
   no mesmo espaço do ViT-B/32. Mantém o inglês, mas o português para em 66,4%.

**Recomendação:** opção 1 por enquanto. Se aparecer uso real em inglês, medir a opção 2 com
`make avaliar` antes de mudar. Para voltar ao CLIP puro sem deploy de código, basta `TRADUZIR_BUSCA=false`.
