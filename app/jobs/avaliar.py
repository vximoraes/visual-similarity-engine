"""
Mede a qualidade dos semelhantes e da busca por texto no conjunto importado por `importar`.

- Semelhantes: precisão@10 de cada imagem (quantos dos 10 vizinhos são da mesma categoria), média
  geral e por categoria, e as categorias que mais se confundem.
- Texto: para cada categoria, uma consulta em português e outra em inglês ("cachorro" e "dog"),
  codificadas pelo mesmo encoder de texto da rota `/search-text`.

Gera `<pasta>/relatorio.html` (local, não versionado) com exemplos para olhar a olho e imprime as
tabelas em Markdown para `docs/avaliacao.md`.

Rodar: `make avaliar PASTA=assets` (ou `uv run python -m app.jobs.avaliar <pasta>`)
"""

import argparse
import asyncio
import html
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from app import avaliacao, clip_model, repository
from app.db import Sessao
from app.jobs.importar import ler_indice

log = logging.getLogger(__name__)

CONFUSOES_MOSTRADAS = 3


@dataclass(frozen=True)
class Conjunto:
    ids: list[uuid.UUID]
    arquivos: list[str]
    categorias: list[str]
    vetores: clip_model.Vetores


@dataclass(frozen=True)
class Texto:
    categorias: list[str]
    consultas: list[str]
    idioma: list[str]
    precisoes: list[float]
    resultados: NDArray[np.int64]

    def media(self, idioma: str) -> float:
        valores = [p for p, i in zip(self.precisoes, self.idioma, strict=True) if i == idioma]
        return float(np.mean(valores))

    def da_categoria(self, categoria: str, idioma: str) -> int:
        return next(
            i
            for i, (c, lingua) in enumerate(zip(self.categorias, self.idioma, strict=True))
            if c == categoria and lingua == idioma
        )


async def carregar(pasta: Path) -> Conjunto:
    indice = ler_indice(pasta)
    if not indice:
        raise SystemExit(f"Sem {pasta}/indice.json: rode `make importar PASTA={pasta}` antes")
    ordem = sorted(indice.items(), key=lambda par: (par[1]["categoria"], par[1]["arquivo"]))
    ids = [uuid.UUID(i) for i, _ in ordem]
    async with Sessao() as sessao:
        vetores = await repository.embeddings_por_id(sessao, ids)
    faltando = len(ids) - len(vetores)
    if faltando:
        raise SystemExit(
            f"{faltando} imagens do índice não estão no banco (o `make test` limpa as tabelas): "
            f"rode `make importar PASTA={pasta}` de novo"
        )
    return Conjunto(
        ids=ids,
        arquivos=[dados["arquivo"] for _, dados in ordem],
        categorias=[dados["categoria"] for _, dados in ordem],
        vetores=np.stack([vetores[i] for i in ids]),
    )


async def avaliar_texto(conjunto: Conjunto) -> Texto | None:
    """None quando nenhuma categoria da pasta tem consulta em `CONSULTAS`"""
    presentes = sorted(set(conjunto.categorias) - {avaliacao.SEM_CATEGORIA})
    sem_consulta = [c for c in presentes if c not in avaliacao.CONSULTAS]
    if sem_consulta:
        log.warning("Categorias sem consulta de texto em CONSULTAS: %s", ", ".join(sem_consulta))
    categorias, consultas, idiomas = [], [], []
    for categoria in presentes:
        if categoria in avaliacao.CONSULTAS:
            for idioma, consulta in zip(("pt", "en"), avaliacao.CONSULTAS[categoria], strict=True):
                categorias.append(categoria)
                consultas.append(consulta)
                idiomas.append(idioma)

    if not consultas:
        return None

    modelo = await asyncio.to_thread(clip_model.carregar)
    vetores_texto = await asyncio.to_thread(modelo.codificar_textos, consultas)
    precisoes, resultados = avaliacao.precisao_do_texto(
        vetores_texto, categorias, conjunto.vetores, conjunto.categorias
    )
    return Texto(categorias, consultas, idiomas, precisoes, resultados)


def porcentagem(valor: float) -> str:
    return f"{valor * 100:.1f}%".replace(".", ",")


def tabelas_markdown(
    resultado: avaliacao.Avaliacao, texto: Texto | None, conjunto: Conjunto
) -> str:
    quantidade = {c: conjunto.categorias.count(c) for c in resultado.por_categoria}
    linhas = [
        f"Precisão@{resultado.k} geral dos semelhantes: **{porcentagem(resultado.geral)}** "
        f"({sum(quantidade.values())} imagens avaliadas em {len(quantidade)} categorias, "
        f"{len(conjunto.ids)} na base)",
        "",
        "| Categoria | Imagens | Semelhantes | Mais confundida com |",
        "|---|---|---|---|",
    ]
    for categoria, precisao in sorted(resultado.por_categoria.items(), key=lambda p: p[1]):
        erros = resultado.confusoes[categoria]
        total_erros = sum(erros.values()) or 1
        confusoes = ", ".join(
            f"{c} ({porcentagem(n / total_erros)})"
            for c, n in erros.most_common(CONFUSOES_MOSTRADAS)
        )
        linhas.append(
            f"| {categoria} | {quantidade[categoria]} | {porcentagem(precisao)} | "
            f"{confusoes or '-'} |"
        )

    if texto is None:
        return "\n".join(linhas)
    linhas += [
        "",
        f"Busca por texto, precisão@{resultado.k}: português **{porcentagem(texto.media('pt'))}**,"
        f" inglês **{porcentagem(texto.media('en'))}**",
        "",
        "| Categoria | Consulta PT | Precisão PT | Consulta EN | Precisão EN |",
        "|---|---|---|---|---|",
    ]
    for categoria in sorted(set(texto.categorias)):
        pt, en = texto.da_categoria(categoria, "pt"), texto.da_categoria(categoria, "en")
        linhas.append(
            f"| {categoria} | {texto.consultas[pt]} | {porcentagem(texto.precisoes[pt])} | "
            f"{texto.consultas[en]} | {porcentagem(texto.precisoes[en])} |"
        )
    return "\n".join(linhas)


def _miniatura(conjunto: Conjunto, indice: int, esperada: str, legenda: str = "") -> str:
    classe = "certo" if conjunto.categorias[indice] == esperada else "errado"
    arquivo = html.escape(conjunto.arquivos[indice])
    titulo = html.escape(f"{conjunto.categorias[indice]}: {conjunto.arquivos[indice]}")
    return (
        f'<figure class="{classe}"><img src="{arquivo}" title="{titulo}" loading="lazy">'
        f"<figcaption>{html.escape(conjunto.categorias[indice])}{legenda}</figcaption></figure>"
    )


def _linha_de_imagem(conjunto: Conjunto, resultado: avaliacao.Avaliacao, i: int) -> str:
    categoria = conjunto.categorias[i]
    vizinhos = resultado.vizinhos[i]
    similaridades = conjunto.vetores[vizinhos] @ conjunto.vetores[i]
    miniaturas = "".join(
        _miniatura(conjunto, int(v), categoria, f" {s:.2f}")
        for v, s in zip(vizinhos, similaridades, strict=True)
    )
    return (
        f'<div class="linha"><figure class="consulta"><img src="'
        f'{html.escape(conjunto.arquivos[i])}"><figcaption>consulta '
        f"({porcentagem(resultado.por_imagem[i])})</figcaption></figure>{miniaturas}</div>"
    )


def relatorio_html(resultado: avaliacao.Avaliacao, texto: Texto | None, conjunto: Conjunto) -> str:
    """Para cada categoria, a melhor e a pior consulta por imagem e as duas consultas de texto"""
    secoes = []
    for categoria in sorted(resultado.por_categoria):
        membros = [i for i, c in enumerate(conjunto.categorias) if c == categoria]
        notas = resultado.por_imagem[membros]
        melhor, pior = membros[int(notas.argmax())], membros[int(notas.argmin())]
        linhas_texto = []
        for idioma in ("pt", "en"):
            if texto is None or categoria not in avaliacao.CONSULTAS:
                break
            t = texto.da_categoria(categoria, idioma)
            miniaturas = "".join(
                _miniatura(conjunto, int(r), categoria) for r in texto.resultados[t]
            )
            linhas_texto.append(
                f'<div class="linha"><div class="texto">"{html.escape(texto.consultas[t])}"<br>'
                f"{porcentagem(texto.precisoes[t])}</div>{miniaturas}</div>"
            )
        secoes.append(
            f"<section><h2>{html.escape(categoria)}: "
            f"{porcentagem(resultado.por_categoria[categoria])}</h2>"
            f"<h3>Melhor e pior consulta por imagem</h3>"
            f"{_linha_de_imagem(conjunto, resultado, melhor)}"
            f"{_linha_de_imagem(conjunto, resultado, pior)}"
            f"{'<h3>Busca por texto</h3>' if linhas_texto else ''}{''.join(linhas_texto)}"
            "</section>"
        )

    avaliadas = sum(c != avaliacao.SEM_CATEGORIA for c in conjunto.categorias)
    resumo_texto = (
        f"Texto em português: <b>{porcentagem(texto.media('pt'))}</b>, "
        f"em inglês: <b>{porcentagem(texto.media('en'))}</b>."
        if texto
        else ""
    )
    return f"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Relatório de avaliação</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 16px; background: #fafafa; color: #222; }}
  .resumo {{ font-size: 1.1rem; }}
  .linha {{ display: flex; gap: 6px; overflow-x: auto; margin: 8px 0; align-items: flex-start; }}
  figure {{ margin: 0; text-align: center; font-size: 0.75rem; flex: none; }}
  img {{ width: 112px; height: 112px; object-fit: cover; display: block; border: 4px solid; }}
  .certo img {{ border-color: #2e7d32; }}
  .errado img {{ border-color: #c62828; }}
  .consulta img {{ border-color: #1565c0; width: 140px; height: 140px; }}
  .texto {{ width: 140px; flex: none; font-weight: 600; }}
  section {{ border-top: 1px solid #ddd; padding-top: 8px; }}
</style>
</head>
<body>
<h1>Relatório de avaliação</h1>
<p class="resumo">{len(conjunto.ids)} imagens na base, {avaliadas} avaliadas.
Precisão@{resultado.k} dos semelhantes:
<b>{porcentagem(resultado.geral)}</b>.
{resumo_texto}
Borda verde: mesma categoria; vermelha: outra.</p>
{"".join(secoes)}
</body>
</html>
"""


async def executar(pasta: Path) -> tuple[avaliacao.Avaliacao, Texto | None]:
    conjunto = await carregar(pasta)
    try:
        resultado = await asyncio.to_thread(
            avaliacao.precisao_dos_vizinhos, conjunto.vetores, conjunto.categorias
        )
    except ValueError as erro:
        raise SystemExit(str(erro)) from erro
    texto = await avaliar_texto(conjunto)
    (pasta / "relatorio.html").write_text(
        relatorio_html(resultado, texto, conjunto), encoding="utf-8"
    )
    print(tabelas_markdown(resultado, texto, conjunto))
    log.info("Relatório em %s", pasta / "relatorio.html")
    return resultado, texto


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Precisão@10 dos semelhantes e da busca por texto")
    parser.add_argument("pasta", type=Path, nargs="?", default=Path("assets"))
    asyncio.run(executar(parser.parse_args().pasta))
