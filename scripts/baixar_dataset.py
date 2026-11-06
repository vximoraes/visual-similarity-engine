"""
Monta o conjunto de imagens reais em `assets/<categoria>/` a partir do manifesto versionado
(`scripts/dataset/manifesto.csv`). Cada linha traz a URL, o sha256 esperado, o autor e a licença.
Arquivo já baixado e íntegro não é baixado de novo.

Rodar: `make dataset` (ou `uv run python scripts/baixar_dataset.py [destino]`)
"""

import argparse
import asyncio
import csv
import hashlib
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import httpx

RAIZ = Path(__file__).resolve().parent.parent
MANIFESTO = RAIZ / "scripts" / "dataset" / "manifesto.csv"
DESTINO_PADRAO = RAIZ / "assets"
CAMPOS = ["categoria", "arquivo", "sha256", "url", "pagina", "autor", "licenca", "licenca_url"]

# Política do Wikimedia: User-Agent que identifique o cliente e um contato (o repositório)
USER_AGENT = (
    "visual-similarity-engine/0.1 (https://github.com/vximoraes/visual-similarity-engine) "
    f"httpx/{httpx.__version__}"
)
PARALELO = 4
TENTATIVAS = 4


@dataclass(frozen=True)
class Item:
    categoria: str
    arquivo: str
    sha256: str
    url: str
    pagina: str
    autor: str
    licenca: str
    licenca_url: str

    def caminho(self, destino: Path) -> Path:
        return destino / self.categoria / self.arquivo


class Situacao(StrEnum):
    BAIXADO = "baixado"
    JA_EXISTIA = "já existia"
    DIVERGENTE = "baixado, mas o sha256 difere do manifesto"
    FALHOU = "falhou"


def sha256(conteudo: bytes) -> str:
    return hashlib.sha256(conteudo).hexdigest()


def gravar(caminho: Path, conteudo: bytes) -> None:
    """Grava num temporário e renomeia: um download interrompido não deixa arquivo pela metade"""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    temporario = caminho.with_suffix(".tmp")
    temporario.write_bytes(conteudo)
    temporario.replace(caminho)


def ler_manifesto(caminho: Path = MANIFESTO) -> list[Item]:
    with caminho.open(newline="", encoding="utf-8") as arquivo:
        return [Item(**linha) for linha in csv.DictReader(arquivo)]


def escrever_manifesto(itens: list[Item], caminho: Path = MANIFESTO) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=CAMPOS, lineterminator="\n")
        escritor.writeheader()
        for item in sorted(itens, key=lambda i: (i.categoria, i.arquivo)):
            escritor.writerow({campo: getattr(item, campo) for campo in CAMPOS})


def criar_cliente() -> httpx.AsyncClient:
    return httpx.AsyncClient(headers={"User-Agent": USER_AGENT}, timeout=30, follow_redirects=True)


async def baixar_bytes(cliente: httpx.AsyncClient, url: str) -> bytes:
    """Repete em erro de rede, 429 e 5xx (o servidor de mídia do Wikimedia limita rajadas)"""
    for tentativa in range(TENTATIVAS):
        try:
            resposta = await cliente.get(url)
            if resposta.status_code != 429 and resposta.status_code < 500:
                resposta.raise_for_status()
                return resposta.content
            if tentativa == TENTATIVAS - 1:
                resposta.raise_for_status()
        except httpx.TransportError:
            if tentativa == TENTATIVAS - 1:
                raise
        await asyncio.sleep(2**tentativa)
    raise AssertionError("inalcançável")


async def baixar_item(
    cliente: httpx.AsyncClient, item: Item, destino: Path, limite: asyncio.Semaphore
) -> Situacao:
    caminho = item.caminho(destino)
    if caminho.exists() and sha256(caminho.read_bytes()) == item.sha256:
        return Situacao.JA_EXISTIA
    async with limite:
        try:
            conteudo = await baixar_bytes(cliente, item.url)
        except httpx.HTTPError as erro:
            print(f"falhou: {item.categoria}/{item.arquivo} ({erro})", file=sys.stderr)
            return Situacao.FALHOU
    gravar(caminho, conteudo)
    if sha256(conteudo) != item.sha256:
        # A URL é de miniatura (thumb.wikimedia.org), que o Wikimedia pode regerar: mesma foto com
        # bytes diferentes. Mantém e avisa; o resumo final mostra quantas divergiram
        print(f"aviso: sha256 diferente em {item.categoria}/{item.arquivo}", file=sys.stderr)
        return Situacao.DIVERGENTE
    return Situacao.BAIXADO


async def baixar_todos(itens: list[Item], destino: Path) -> dict[Situacao, int]:
    limite = asyncio.Semaphore(PARALELO)
    async with criar_cliente() as cliente:
        situacoes = await asyncio.gather(*(baixar_item(cliente, i, destino, limite) for i in itens))
    return {s: situacoes.count(s) for s in Situacao}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("destino", nargs="?", type=Path, default=DESTINO_PADRAO)
    args = parser.parse_args()

    itens = ler_manifesto()
    contagem = asyncio.run(baixar_todos(itens, args.destino))
    categorias = len({i.categoria for i in itens})
    print(f"{len(itens)} imagens em {categorias} categorias, em {args.destino}")
    for situacao, total in contagem.items():
        if total:
            print(f"  {situacao}: {total}")
    return 1 if contagem[Situacao.FALHOU] else 0


if __name__ == "__main__":
    sys.exit(main())
