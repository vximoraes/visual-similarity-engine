"""
Algoritmo de categorias (sem banco, para testar isolado).

Os embeddings do CLIP são normalizados, então o k-means sobre eles é o "k-means esférico": a
distância euclidiana entre vetores unitários é monotônica com a de cosseno. Os centroides são
normalizados de novo depois do ajuste, para a atribuição usar produto interno (cosseno).
"""

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
    if topo < k_min:
        return []
    valores = np.unique(np.linspace(k_min, topo, num=min(CANDIDATOS_MAX, topo - k_min + 1)))
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


# Vocabulário para o rótulo automático: o CLIP compara o centroide com "a photo of <termo>".
# O termo vai em inglês (o CLIP foi treinado em inglês); o rótulo salvo é o português
VOCABULARIO: dict[str, str] = {
    "food": "comida",
    "dessert": "sobremesa",
    "drink": "bebida",
    "fashion and clothing": "moda",
    "shoes": "calçados",
    "jewelry": "joias",
    "makeup": "maquiagem",
    "hairstyle": "cabelo",
    "interior design": "decoração",
    "architecture": "arquitetura",
    "garden": "jardim",
    "nature landscape": "natureza",
    "beach": "praia",
    "mountains": "montanhas",
    "city": "cidade",
    "travel": "viagem",
    "dog": "cachorros",
    "cat": "gatos",
    "wild animals": "animais",
    "birds": "pássaros",
    "flowers": "flores",
    "car": "carros",
    "motorcycle": "motos",
    "sports": "esportes",
    "fitness": "fitness",
    "technology": "tecnologia",
    "painting and art": "arte",
    "illustration": "ilustração",
    "photography": "fotografia",
    "tattoo": "tatuagem",
    "wedding": "casamento",
    "baby": "bebês",
    "people portrait": "retratos",
    "text and quotes": "frases",
    "diy crafts": "artesanato",
    "games": "jogos",
    "anime": "anime",
    "music": "música",
    "books": "livros",
    "space": "espaço",
}


def rotular(centroides: Vetores, vetores_vocabulario: Vetores) -> list[str]:
    """Rótulo de cada centroide: o termo do vocabulário mais parecido (zero-shot do CLIP)"""
    termos = list(VOCABULARIO.values())
    return [termos[i] for i in (centroides @ vetores_vocabulario.T).argmax(axis=1)]
