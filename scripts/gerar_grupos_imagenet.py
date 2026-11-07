"""
Agrupa as 1.000 classes (synsets) do ImageNet em categorias com cara de Pinterest, pela hierarquia
do WordNet, e grava `scripts/imagenet/grupos.csv` (wnid, nome, grupo).

Só é preciso rodar para mudar o agrupamento: o CSV é versionado e `preparar_imagenet.py` usa só
ele. Precisa do nltk, que não é dependência do projeto:

    uv run --no-project --with nltk python scripts/gerar_grupos_imagenet.py <zip do professor>

Uma classe entra no primeiro grupo de GRUPOS que tiver um ancestral dela (a ordem resolve a classe
que cai em dois grupos: "mobile home" é veículo e construção, fica em veículos). Classe sem
grupo, de grupo com poucas imagens ou listada em PARA_OUTROS vai para "outros": continua na base
como vizinha possível, mas não é avaliada.
"""

import csv
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

import nltk
from nltk.corpus import wordnet as wn

SAIDA = Path(__file__).resolve().parent / "imagenet" / "grupos.csv"
OUTROS = "outros"
MINIMO_DE_IMAGENS = 30
ARQUIVO = re.compile(r"^val/(n\d{8})/[^/]+\.JPEG$")

# Ordem importa: o primeiro grupo que contém a classe vence
GRUPOS: dict[str, list[str]] = {
    "cachorros": ["dog.n.01"],
    "felinos": ["feline.n.01"],
    "pássaros": ["bird.n.01"],
    "peixes": ["fish.n.01"],
    "répteis": ["reptile.n.01"],
    "insetos": ["insect.n.01", "arachnid.n.01"],
    "primatas": ["primate.n.02"],
    "comida": [
        "food.n.01",
        "food.n.02",
        "dish.n.02",
        "edible_fruit.n.01",
        "vegetable.n.01",
        "fungus.n.01",
    ],
    "moda": ["clothing.n.01", "footwear.n.02", "jewelry.n.01"],
    "decoração": ["furniture.n.01", "home_appliance.n.01"],
    "instrumentos": ["musical_instrument.n.01"],
    "veículos": ["vehicle.n.01", "craft.n.02"],
    "arquitetura": ["building.n.01", "structure.n.01"],
    "paisagens": ["geological_formation.n.01"],
}

# O WordNet põe objetos no grupo pela definição, não pela foto: "menu" é comida, "seat belt" é
# roupa, "scoreboard" é construção. Estas classes não representam a categoria numa foto e vão
# para "outros" (nome = primeiro lema do synset)
PARA_OUTROS = {
    "comida": ["menu", "plate", "cup", "hay"],
    "moda": [
        "seat belt",
        "knee pad",
        "crash helmet",
        "football helmet",
        "bulletproof vest",
        "mask",
    ],
    "decoração": ["park bench"],
    "veículos": ["missile", "shopping cart", "barrow"],
    "arquitetura": [
        "brass",
        "coil",
        "honeycomb",
        "grille",
        "plate rack",
        "scoreboard",
        "turnstile",
        "home theater",
    ],
}


def nome_de(wnid: str) -> str:
    return str(wn.synset_from_pos_and_offset("n", int(wnid[1:])).lemma_names()[0]).replace("_", " ")


def contar_por_wnid(caminho_zip: Path) -> Counter[str]:
    with zipfile.ZipFile(caminho_zip) as arquivo:
        return Counter(
            achado.group(1) for nome in arquivo.namelist() if (achado := ARQUIVO.match(nome))
        )


def grupo_de(wnid: str, raizes: dict[str, list[object]]) -> str:
    synset = wn.synset_from_pos_and_offset("n", int(wnid[1:]))
    ancestrais = {h for caminho in synset.hypernym_paths() for h in caminho}
    return next((g for g, rs in raizes.items() if any(r in ancestrais for r in rs)), OUTROS)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    nltk.download("wordnet", quiet=True)
    por_wnid = contar_por_wnid(Path(sys.argv[1]))
    raizes: dict[str, list[object]] = {
        g: [wn.synset(s) for s in synsets] for g, synsets in GRUPOS.items()
    }
    grupos = {wnid: grupo_de(wnid, raizes) for wnid in por_wnid}
    for grupo, nomes in PARA_OUTROS.items():
        for nome in nomes:
            achados = [w for w, g in grupos.items() if g == grupo and nome_de(w) == nome]
            if len(achados) != 1:
                raise SystemExit(f"PARA_OUTROS: {nome!r} não é uma classe única de {grupo}")
            grupos[achados[0]] = OUTROS

    imagens: Counter[str] = Counter()
    for wnid, grupo in grupos.items():
        imagens[grupo] += por_wnid[wnid]
    pequenos = {g for g, n in imagens.items() if n < MINIMO_DE_IMAGENS and g != OUTROS}
    grupos = {wnid: OUTROS if g in pequenos else g for wnid, g in grupos.items()}

    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    with SAIDA.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo, lineterminator="\n")
        escritor.writerow(["wnid", "nome", "grupo"])
        for wnid in sorted(grupos):
            escritor.writerow([wnid, nome_de(wnid), grupos[wnid]])

    final: Counter[str] = Counter()
    for wnid, grupo in grupos.items():
        final[grupo] += por_wnid[wnid]
    for grupo, total in final.most_common():
        print(f"{grupo}: {total} imagens")
    if pequenos:
        print(f"Grupos com menos de {MINIMO_DE_IMAGENS} imagens foram para {OUTROS}: {pequenos}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
