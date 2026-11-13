"""
Algoritmo de categorias (sem banco, para testar isolado).

Os embeddings do CLIP são normalizados, então o k-means sobre eles é o "k-means esférico": a
distância euclidiana entre vetores unitários é monotônica com a de cosseno. Os centroides são
normalizados de novo depois do ajuste, para a atribuição usar produto interno (cosseno).
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

from app.clip_model import Vetores, normalizar

AMOSTRAS_MINIMAS = 3
# Silhueta é O(n²): acima disso, calcula numa amostra
AMOSTRA_SILHUETA = 5_000
CANDIDATOS_MAX = 12


@dataclass(frozen=True)
class Resultado:
    k: int
    silhueta: float
    centroides: Vetores
    rotulos: NDArray[np.int64]
    distancias: NDArray[np.float32]


class AmostrasInsuficientes(Exception):
    pass


def candidatos_de_k(n: int, k_min: int, k_max: int) -> list[int]:
    """Até CANDIDATOS_MAX valores de k entre k_min e min(k_max, n - 1), espaçados"""
    topo = min(k_max, n - 1)
    if topo < 2:
        return []
    inicio = k_min if topo >= k_min else 2
    valores = np.unique(np.linspace(inicio, topo, num=min(CANDIDATOS_MAX, topo - inicio + 1)))
    return [int(v) for v in valores]


def ajustar(vetores: Vetores, k: int, semente: int = 42) -> tuple[Vetores, NDArray[np.int64]]:
    modelo = MiniBatchKMeans(n_clusters=k, n_init=3, random_state=semente, batch_size=1024)
    rotulos = modelo.fit_predict(vetores)
    return normalizar(modelo.cluster_centers_), rotulos.astype(np.int64)


def atribuir(
    vetores: Vetores, centroides: Vetores
) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    """Centroide mais próximo de cada vetor e a distância de cosseno (1 - similaridade)"""
    similaridades = vetores @ centroides.T
    indices = similaridades.argmax(axis=1)
    distancias = 1.0 - similaridades[np.arange(len(vetores)), indices]
    return indices.astype(np.int64), distancias.astype(np.float32)


def clusterizar(vetores: Vetores, k_min: int, k_max: int, semente: int = 42) -> Resultado:
    """Testa alguns k e fica com o de maior silhueta (cosseno)"""
    n = len(vetores)
    candidatos = candidatos_de_k(n, k_min, k_max)
    if n < AMOSTRAS_MINIMAS or not candidatos:
        raise AmostrasInsuficientes(f"{n} imagens não bastam para k >= {k_min}")

    amostra = min(n, AMOSTRA_SILHUETA)
    melhor: Resultado | None = None
    for k in candidatos:
        centroides, rotulos = ajustar(vetores, k, semente)
        if len(np.unique(rotulos)) < 2:
            continue
        nota = float(
            silhouette_score(
                vetores, rotulos, metric="cosine", sample_size=amostra, random_state=semente
            )
        )
        if melhor is None or nota > melhor.silhueta:
            indices, distancias = atribuir(vetores, centroides)
            melhor = Resultado(k, nota, centroides, indices, distancias)

    if melhor is None:
        raise AmostrasInsuficientes("todas as imagens caíram no mesmo grupo")
    return melhor


# Vocabulário para o rótulo automático: o CLIP compara cada imagem com frases sobre cada termo.
# O termo vai em inglês (o CLIP foi treinado em inglês); o rótulo salvo é o português
VOCABULARIO: dict[str, str] = {
    "food": "comida",
    "dessert": "sobremesas",
    "cake": "bolos",
    "bread and baking": "pães",
    "coffee": "café",
    "cocktails and drinks": "drinks",
    "fruit": "frutas",
    "vegetables": "legumes",
    "healthy salad": "saladas",
    "pizza and fast food": "lanches",
    "fashion outfit": "moda",
    "shoes and sneakers": "calçados",
    "handbags": "bolsas",
    "jewelry": "joias",
    "watches": "relógios",
    "hats": "chapéus",
    "sunglasses": "óculos",
    "swimwear": "moda praia",
    "makeup": "maquiagem",
    "nail art": "unhas",
    "hairstyle": "cabelo",
    "interior design": "decoração",
    "living room": "sala de estar",
    "kitchen": "cozinha",
    "bedroom": "quarto",
    "bathroom": "banheiro",
    "furniture": "móveis",
    "lamps and lighting": "iluminação",
    "home appliances": "eletrodomésticos",
    "houseplants": "plantas",
    "garden": "jardim",
    "tableware and dishes": "louças",
    "kitchen utensils": "utensílios de cozinha",
    "tools": "ferramentas",
    "architecture": "arquitetura",
    "castles and palaces": "castelos",
    "churches and temples": "templos",
    "bridges": "pontes",
    "city": "cidade",
    "travel": "viagem",
    "beach": "praia",
    "mountains": "montanhas",
    "lake": "lagos",
    "forest": "florestas",
    "desert": "deserto",
    "ocean": "mar",
    "sunset": "pôr do sol",
    "nature landscape": "natureza",
    "snow and winter": "inverno",
    "dogs": "cachorros",
    "puppies": "filhotes",
    "cats": "gatos",
    "lions and tigers": "felinos",
    "birds": "pássaros",
    "fish": "peixes",
    "insects": "insetos",
    "butterflies": "borboletas",
    "spiders": "aranhas",
    "reptiles": "répteis",
    "snakes": "cobras",
    "turtles": "tartarugas",
    "frogs": "sapos",
    "monkeys": "macacos",
    "bears": "ursos",
    "elephants": "elefantes",
    "horses": "cavalos",
    "farm animals": "animais da fazenda",
    "wild animals": "animais selvagens",
    "marine animals": "animais marinhos",
    "small mammals": "pequenos mamíferos",
    "cars": "carros",
    "sports cars": "carros esportivos",
    "motorcycles": "motos",
    "bicycles": "bicicletas",
    "boats": "barcos",
    "airplanes": "aviões",
    "trains": "trens",
    "trucks": "caminhões",
    "musical instruments": "instrumentos musicais",
    "music": "música",
    "sports": "esportes",
    "fitness and gym": "fitness",
    "soccer": "futebol",
    "camping and outdoors": "camping",
    "diy crafts": "artesanato",
    "sewing and knitting": "costura",
    "painting and art": "arte",
    "drawing and illustration": "ilustração",
    "photography": "fotografia",
    "tattoo": "tatuagem",
    "books": "livros",
    "games": "jogos",
    "toys": "brinquedos",
    "anime": "anime",
    "technology": "tecnologia",
    "computers": "computadores",
    "phones": "celulares",
    "cameras": "câmeras",
    "electronics": "eletrônicos",
    "office and workspace": "escritório",
    "school supplies": "material escolar",
    "science": "ciência",
    "space": "espaço",
    "wedding": "casamento",
    "baby": "bebês",
    "kids": "crianças",
    "people portrait": "retratos",
    "party": "festas",
    "christmas": "natal",
    "flowers": "flores",
    "text and quotes": "frases",
}

FRASES = ("a photo of {}", "a picture of {}", "{} aesthetic")


def vetores_do_vocabulario(
    codificar: Callable[[list[str]], Vetores], termos: list[str] | None = None
) -> Vetores:
    termos = list(VOCABULARIO) if termos is None else termos
    vetores = codificar([frase.format(termo) for termo in termos for frase in FRASES])
    return normalizar(vetores.reshape(len(termos), len(FRASES), -1).mean(axis=1))


def rotular(
    vetores: Vetores, rotulos: NDArray[np.int64], k: int, vetores_vocabulario: Vetores
) -> list[str]:
    """Rótulo de cada cluster: o termo mais votado pelas imagens dele (zero-shot do CLIP)"""
    termos = list(VOCABULARIO.values())
    votos = np.zeros((k, len(termos)))
    np.add.at(votos, (rotulos, (vetores @ vetores_vocabulario.T).argmax(axis=1)), 1)
    fracoes = votos / np.maximum(votos.sum(axis=1, keepdims=True), 1)
    escolhidos: list[int | None] = [None] * k
    usados: set[int] = set()
    for posicao in np.argsort(-fracoes, axis=None, kind="stable"):
        cluster, termo = divmod(int(posicao), len(termos))
        if escolhidos[cluster] is None and termo not in usados:
            escolhidos[cluster] = termo
            usados.add(termo)
    return [
        termos[int(fracoes[c].argmax()) if termo is None else termo]
        for c, termo in enumerate(escolhidos)
    ]
