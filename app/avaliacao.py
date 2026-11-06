"""
Métricas de qualidade da busca (sem banco, para testar isolado).

Precisão@k de uma consulta = fração dos k primeiros resultados que são da categoria esperada. Os
vizinhos são calculados de forma exata (produto interno entre vetores normalizados = cosseno), só
dentro do conjunto avaliado: assim o número mede o modelo, sem a aproximação do HNSW e sem imagens
de fora do conjunto, que não têm categoria.
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from app.clip_model import Vetores

K = 10
# Linhas por bloco no cálculo dos vizinhos: memória O(bloco * n) em vez de O(n²)
BLOCO = 1024

# Consulta de texto por categoria do conjunto: (português, inglês). Descrevem o conteúdo das fotos:
# "decoração" no conjunto é de móveis; "arte", de pinturas; "moda", de roupas e sapatos
CONSULTAS: dict[str, tuple[str, str]] = {
    "arquitetura": ("arquitetura", "architecture"),
    "arte": ("pintura", "painting"),
    "cachorros": ("cachorro", "dog"),
    "carros": ("carro", "car"),
    "comida": ("comida", "food"),
    "decoração": ("móveis", "furniture"),
    "flores": ("flor", "flower"),
    "gatos": ("gato", "cat"),
    "moda": ("roupa", "clothes"),
    "montanhas": ("montanha", "mountain"),
    "pássaros": ("pássaro", "bird"),
    "praia": ("praia", "beach"),
}


@dataclass(frozen=True)
class Avaliacao:
    k: int
    geral: float
    por_categoria: dict[str, float]
    # Precisão de cada imagem, na ordem da entrada
    por_imagem: NDArray[np.float64]
    # Índices dos k vizinhos de cada imagem, do mais parecido para o menos
    vizinhos: NDArray[np.int64]
    # Categoria esperada -> categorias que apareceram por engano entre os vizinhos
    confusoes: dict[str, Counter[str]]


def _maiores(similaridades: NDArray[np.float32], k: int) -> NDArray[np.int64]:
    """Índices das k maiores similaridades de cada linha, em ordem decrescente"""
    if similaridades.shape[1] < k:
        raise ValueError(f"{similaridades.shape[1]} imagens não bastam para precisão@{k}")
    topo = np.argpartition(-similaridades, k - 1, axis=1)[:, :k]
    ordem = np.take_along_axis(similaridades, topo, axis=1).argsort(axis=1)[:, ::-1]
    return np.take_along_axis(topo, ordem, axis=1).astype(np.int64)


def mais_parecidos(consultas: Vetores, base: Vetores, k: int) -> NDArray[np.int64]:
    """Índices dos k vetores da base mais parecidos com cada consulta (cosseno)"""
    return _maiores(consultas @ base.T, k)


def vizinhos_exatos(vetores: Vetores, k: int = K, bloco: int = BLOCO) -> NDArray[np.int64]:
    """k vizinhos de cada vetor no próprio conjunto, sem ele mesmo"""
    if len(vetores) <= k:
        raise ValueError(f"{len(vetores)} imagens não bastam para precisão@{k}")
    blocos = []
    for inicio in range(0, len(vetores), bloco):
        similaridades = vetores[inicio : inicio + bloco] @ vetores.T
        linhas = np.arange(len(similaridades))
        similaridades[linhas, linhas + inicio] = -np.inf
        blocos.append(_maiores(similaridades, k))
    return np.concatenate(blocos)


def precisao_dos_vizinhos(vetores: Vetores, categorias: list[str], k: int = K) -> Avaliacao:
    vizinhos = vizinhos_exatos(vetores, k)
    rotulos = np.array(categorias)
    acertos = rotulos[vizinhos] == rotulos[:, None]
    por_imagem = acertos.mean(axis=1)

    confusoes: dict[str, Counter[str]] = {c: Counter() for c in sorted(set(categorias))}
    for i, categoria in enumerate(categorias):
        confusoes[categoria].update(rotulos[vizinhos[i][~acertos[i]]].tolist())

    return Avaliacao(
        k=k,
        geral=float(por_imagem.mean()),
        por_categoria={c: float(por_imagem[rotulos == c].mean()) for c in sorted(set(categorias))},
        por_imagem=por_imagem,
        vizinhos=vizinhos,
        confusoes=confusoes,
    )


def precisao_do_texto(
    vetores_texto: Vetores,
    esperadas: list[str],
    vetores: Vetores,
    categorias: list[str],
    k: int = K,
) -> tuple[list[float], NDArray[np.int64]]:
    """Precisão@k de cada consulta de texto e os índices dos resultados"""
    resultados = mais_parecidos(vetores_texto, vetores, k)
    rotulos = np.array(categorias)
    precisoes = [
        float((rotulos[resultados[i]] == esperada).mean()) for i, esperada in enumerate(esperadas)
    ]
    return precisoes, resultados
