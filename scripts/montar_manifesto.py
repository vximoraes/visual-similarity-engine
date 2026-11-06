"""
Escolhe as imagens do conjunto de avaliação no Wikimedia Commons e grava o manifesto
(`scripts/dataset/manifesto.csv`) já com o sha256 de cada arquivo baixado.

Só é preciso rodar para mudar o conjunto: o manifesto é versionado e `baixar_dataset.py` refaz o
mesmo conjunto a partir dele. Rodar de novo mais tarde pode escolher outras imagens, porque as
categorias do Commons mudam com o tempo.

Fonte: categorias "Quality images of ..." (fotos revisadas pela comunidade do Commons), só com
licença livre (CC0, domínio público, CC BY, CC BY-SA). Regras de seleção:

- largura e altura de pelo menos 640 px e proporção entre 1:2 e 2:1 (panorama perde demais no
  center crop de 224x224);
- arquivo que aparece em mais de uma categoria do conjunto fica de fora (rótulo ambíguo);
- arquivo reprovado na revisão manual (`scripts/dataset/revisao_manual.csv`) fica de fora;
- até IMAGENS_POR_CATEGORIA por categoria, alternando entre autores (uma série do mesmo fotógrafo
  não domina a categoria), com sorteio de semente fixa.

Rodar: `uv run python scripts/montar_manifesto.py`
"""

import asyncio
import csv
import hashlib
import html
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from baixar_dataset import (
    DESTINO_PADRAO,
    Item,
    baixar_bytes,
    criar_cliente,
    escrever_manifesto,
    gravar,
    sha256,
)
from PIL import Image

API = "https://commons.wikimedia.org/w/api.php"
IMAGENS_POR_CATEGORIA = 40
LARGURA_MINIATURA = 960
LADO_MINIMO = 640
SEMENTE = 42
LICENCA_LIVRE = re.compile(r"^(CC0|CC BY(-SA)? \d|Public domain|PD)", re.I)


@dataclass(frozen=True)
class Fonte:
    categoria_commons: str
    profundidade: int = 0


# Nome da pasta = rótulo do VOCABULARIO de app/clusterizacao.py, para comparar com o k-means
CATEGORIAS: dict[str, list[Fonte]] = {
    "arquitetura": [Fonte("Quality images of buildings"), Fonte("Quality images of skyscrapers")],
    "arte": [Fonte("Quality images of paintings", profundidade=1)],
    "cachorros": [Fonte("Quality images of dogs", profundidade=1)],
    "carros": [Fonte("Quality images of automobiles")],
    "comida": [Fonte("Quality images of food")],
    "decoração": [
        Fonte("Quality images of furniture"),
        Fonte("Quality images of chairs"),
        Fonte("Quality images of tables"),
        Fonte("Quality images of storage furniture"),
    ],
    "flores": [Fonte("Quality images of flowers")],
    "gatos": [Fonte("Quality images of cats", profundidade=1)],
    "moda": [Fonte("Quality images of clothing"), Fonte("Quality images of shoes")],
    "montanhas": [Fonte("Quality images of mountains")],
    "pássaros": [Fonte("Quality images of birds", profundidade=1)],
    "praia": [Fonte("Quality images of beaches")],
}


# Revisão a olho das escolhidas: a categoria do Commons às vezes traz foto que não representa o tema
# (ex.: "clothing" com atletas em competição, "automobiles" com rua cheia de casas)
REVISAO_MANUAL = Path(__file__).resolve().parent / "dataset" / "revisao_manual.csv"


def reprovadas() -> set[int]:
    with REVISAO_MANUAL.open(newline="", encoding="utf-8") as arquivo:
        return {int(linha["pageid"]) for linha in csv.DictReader(arquivo)}


@dataclass(frozen=True)
class Candidato:
    pageid: int
    url: str
    pagina: str
    autor: str
    licenca: str
    licenca_url: str


async def consultar(cliente: httpx.AsyncClient, parametros: dict[str, str]) -> list[dict[str, Any]]:
    """Todas as páginas de uma consulta da API do MediaWiki (segue o `continue`)"""
    respostas: list[dict[str, Any]] = []
    continuar: dict[str, str] = {}
    while True:
        resposta = await cliente.get(
            API, params={"action": "query", "format": "json", **parametros, **continuar}
        )
        resposta.raise_for_status()
        dados = resposta.json()
        respostas.append(dados)
        if "continue" not in dados:
            return respostas
        continuar = dados["continue"]


async def subcategorias(cliente: httpx.AsyncClient, categoria: str, profundidade: int) -> set[str]:
    encontradas = {categoria}
    if profundidade == 0:
        return encontradas
    for dados in await consultar(
        cliente,
        {"list": "categorymembers", "cmtitle": f"Category:{categoria}", "cmtype": "subcat"},
    ):
        for membro in dados["query"]["categorymembers"]:
            nome = membro["title"].removeprefix("Category:")
            encontradas |= await subcategorias(cliente, nome, profundidade - 1)
    return encontradas


def imagem_integra(caminho: Path) -> bool:
    """Reaproveita o arquivo já baixado só se ele abre como imagem (não ficou truncado)"""
    if not caminho.exists():
        return False
    try:
        with Image.open(caminho) as imagem:
            imagem.load()
    except OSError:
        return False
    return True


def texto_simples(valor: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", valor)).split())


def para_candidato(pagina: dict[str, Any]) -> Candidato | None:
    info = (pagina.get("imageinfo") or [None])[0]
    if info is None or info.get("mime") != "image/jpeg":
        return None
    largura, altura = info["width"], info["height"]
    if min(largura, altura) < LADO_MINIMO or not 0.5 <= largura / altura <= 2:
        return None
    meta = info.get("extmetadata", {})
    licenca = meta.get("LicenseShortName", {}).get("value", "")
    if not LICENCA_LIVRE.match(licenca):
        return None
    return Candidato(
        pageid=pagina["pageid"],
        url=info.get("thumburl") or info["url"],
        pagina=info["descriptionurl"],
        autor=texto_simples(meta.get("Artist", {}).get("value", "")) or "desconhecido",
        licenca=licenca,
        licenca_url=meta.get("LicenseUrl", {}).get("value", ""),
    )


async def arquivos(cliente: httpx.AsyncClient, categoria: str) -> list[Candidato]:
    candidatos: list[Candidato] = []
    for dados in await consultar(
        cliente,
        {
            "generator": "categorymembers",
            "gcmtitle": f"Category:{categoria}",
            "gcmtype": "file",
            "gcmlimit": "50",
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata",
            "iiurlwidth": str(LARGURA_MINIATURA),
            "iiextmetadatafilter": "Artist|LicenseShortName|LicenseUrl",
        },
    ):
        for pagina in dados.get("query", {}).get("pages", {}).values():
            if candidato := para_candidato(pagina):
                candidatos.append(candidato)
    return candidatos


def escolher(candidatos: list[Candidato], quantidade: int, semente: int) -> list[Candidato]:
    """
    Alterna entre autores: primeiro a 1ª foto de cada autor, depois a 2ª, e assim por diante. A
    ordem vem de um hash da semente com o pageid, então tirar uma foto (revisão manual) só puxa a
    próxima da fila, sem embaralhar o resto da seleção.
    """

    def sorteio(candidato: Candidato) -> str:
        return hashlib.sha256(f"{semente}:{candidato.pageid}".encode()).hexdigest()

    posicao: dict[int, int] = {}
    por_autor: dict[str, int] = defaultdict(int)
    for candidato in sorted(candidatos, key=sorteio):
        posicao[candidato.pageid] = por_autor[candidato.autor]
        por_autor[candidato.autor] += 1
    return sorted(candidatos, key=lambda c: (posicao[c.pageid], sorteio(c)))[:quantidade]


async def montar(destino: Path) -> list[Item]:
    async with criar_cliente() as cliente:
        por_categoria: dict[str, dict[int, Candidato]] = {}
        for nome, fontes in CATEGORIAS.items():
            categorias_commons: set[str] = set()
            for fonte in fontes:
                categorias_commons |= await subcategorias(
                    cliente, fonte.categoria_commons, fonte.profundidade
                )
            achados: dict[int, Candidato] = {}
            for categoria in sorted(categorias_commons):
                achados |= {c.pageid: c for c in await arquivos(cliente, categoria)}
            por_categoria[nome] = achados

        contagem: dict[int, int] = defaultdict(int)
        for achados in por_categoria.values():
            for pageid in achados:
                contagem[pageid] += 1

        fora = reprovadas()
        itens: list[Item] = []
        for nome, achados in por_categoria.items():
            unicos = [
                c for c in achados.values() if contagem[c.pageid] == 1 and c.pageid not in fora
            ]
            escolhidos = escolher(unicos, IMAGENS_POR_CATEGORIA, SEMENTE)
            print(f"{nome}: {len(escolhidos)} de {len(unicos)} candidatas")
            pasta = destino / nome
            pasta.mkdir(parents=True, exist_ok=True)
            nomes = {f"{c.pageid}.jpg" for c in escolhidos}
            for antigo in pasta.glob("*.jpg"):
                if antigo.name not in nomes:
                    print(f"  removida (saiu da seleção): {nome}/{antigo.name}")
                    antigo.unlink()
            for candidato in escolhidos:
                arquivo = pasta / f"{candidato.pageid}.jpg"
                if imagem_integra(arquivo):
                    conteudo = arquivo.read_bytes()
                else:
                    conteudo = await baixar_bytes(cliente, candidato.url)
                    gravar(arquivo, conteudo)
                itens.append(
                    Item(
                        categoria=nome,
                        arquivo=arquivo.name,
                        sha256=sha256(conteudo),
                        url=candidato.url,
                        pagina=candidato.pagina,
                        autor=candidato.autor,
                        licenca=candidato.licenca,
                        licenca_url=candidato.licenca_url,
                    )
                )
    return itens


def main() -> int:
    itens = asyncio.run(montar(DESTINO_PADRAO))
    escrever_manifesto(itens)
    print(f"Manifesto com {len(itens)} imagens em {len(CATEGORIAS)} categorias")
    return 0


if __name__ == "__main__":
    sys.exit(main())
