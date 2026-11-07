"""
Monta `assets/<categoria>/` a partir do zip da base do professor ("Imagenet Dataset Mini": 3.923
imagens da validação do ImageNet, 1.000 classes). A categoria de cada classe vem de
`scripts/imagenet/grupos.csv`; classes fora das categorias vão para `assets/outros/`. O arquivo
leva a classe no nome (`<wnid>_<arquivo>`), então o `indice.json` da importação também a guarda.

Confere o sha256 do zip (avisa se for outra versão da base) e só extrai caminhos no formato
`val/<wnid>/<arquivo>.JPEG`: nada de fora de `assets/` é escrito, mesmo com zip adulterado.

Rodar: `make imagenet ZIP=<caminho do zip>` (ou
`uv run python scripts/preparar_imagenet.py <zip> [destino]`)
"""

import argparse
import csv
import hashlib
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
GRUPOS = RAIZ / "scripts" / "imagenet" / "grupos.csv"
DESTINO_PADRAO = RAIZ / "assets"
SHA256_DA_BASE = "193336d3084f8fb4c4d4b05e8e50a3f835f8c9c6d21c6e6a29cb4bb1f2889af5"
ARQUIVO = re.compile(r"^val/(n\d{8})/(ILSVRC2012_val_\d{8}\.JPEG)$")


def ler_grupos(caminho: Path = GRUPOS) -> dict[str, str]:
    with caminho.open(newline="", encoding="utf-8") as arquivo:
        return {linha["wnid"]: linha["grupo"] for linha in csv.DictReader(arquivo)}


def sha256_do_arquivo(caminho: Path) -> str:
    resumo = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1 << 20), b""):
            resumo.update(bloco)
    return resumo.hexdigest()


def extrair(caminho_zip: Path, destino: Path, grupos: dict[str, str]) -> Counter[str]:
    """Extrai cada imagem para destino/<grupo>/<wnid>_<arquivo>; arquivo já extraído é mantido"""
    por_grupo: Counter[str] = Counter()
    with zipfile.ZipFile(caminho_zip) as arquivo:
        for info in arquivo.infolist():
            achado = ARQUIVO.match(info.filename)
            if achado is None:
                continue
            wnid, nome = achado.groups()
            if wnid not in grupos:
                raise SystemExit(f"Classe {wnid} não está em {GRUPOS.name}: base diferente?")
            caminho = destino / grupos[wnid] / f"{wnid}_{nome}"
            por_grupo[grupos[wnid]] += 1
            if caminho.exists() and caminho.stat().st_size == info.file_size:
                continue
            caminho.parent.mkdir(parents=True, exist_ok=True)
            temporario = caminho.with_suffix(".tmp")
            temporario.write_bytes(arquivo.read(info))
            temporario.replace(caminho)
    return por_grupo


def main() -> int:
    parser = argparse.ArgumentParser(description="Extrai a base do professor por categoria")
    parser.add_argument("zip", type=Path)
    parser.add_argument("destino", nargs="?", type=Path, default=DESTINO_PADRAO)
    args = parser.parse_args()

    if sha256_do_arquivo(args.zip) != SHA256_DA_BASE:
        print("aviso: o sha256 do zip não é o da base usada em docs/avaliacao.md", file=sys.stderr)
    por_grupo = extrair(args.zip, args.destino, ler_grupos())
    print(f"{sum(por_grupo.values())} imagens em {args.destino}")
    for grupo, total in sorted(por_grupo.items(), key=lambda par: -par[1]):
        print(f"  {grupo}: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
