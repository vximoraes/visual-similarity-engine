import numpy as np
import pytest

from app.clip_model import normalizar
from app.clusterizacao import (
    AmostrasInsuficientes,
    atribuir,
    candidatos_de_k,
    clusterizar,
    rotular,
)


def grupos(centros: int, por_grupo: int, semente: int = 0) -> np.ndarray:
    """Pontos unitários em `centros` direções bem separadas, com um pouco de ruído"""
    gerador = np.random.default_rng(semente)
    direcoes = normalizar(gerador.normal(size=(centros, 512)))
    pontos = np.repeat(direcoes, por_grupo, axis=0) + gerador.normal(
        scale=0.02, size=(centros * por_grupo, 512)
    )
    return normalizar(pontos)


def test_escolhe_o_k_dos_grupos_reais_pela_silhueta() -> None:
    vetores = grupos(centros=4, por_grupo=25)
    resultado = clusterizar(vetores, k_min=2, k_max=10)

    assert resultado.k == 4
    assert resultado.silhueta > 0.5
    assert np.allclose(np.linalg.norm(resultado.centroides, axis=1), 1, atol=1e-5)
    # Cada grupo de 25 pontos cai inteiro no mesmo cluster
    for g in range(4):
        assert len(set(resultado.rotulos[g * 25 : (g + 1) * 25])) == 1


def test_imagem_nova_vai_para_o_centroide_mais_proximo() -> None:
    vetores = grupos(centros=3, por_grupo=20)
    resultado = clusterizar(vetores, k_min=2, k_max=6)
    nova = normalizar(vetores[5:6] + 0.001)

    indices, distancias = atribuir(nova, resultado.centroides)

    assert indices[0] == resultado.rotulos[5]
    assert 0 <= distancias[0] < 0.1


def test_candidatos_respeitam_a_faixa_e_o_numero_de_amostras() -> None:
    assert candidatos_de_k(5, 2, 30) == [2, 3, 4]
    assert len(candidatos_de_k(10_000, 2, 30)) == 12
    assert candidatos_de_k(2, 2, 30) == []


def test_poucas_imagens_nao_clusteriza() -> None:
    with pytest.raises(AmostrasInsuficientes):
        clusterizar(grupos(1, 2), k_min=2, k_max=10)


def test_rotulo_e_o_termo_mais_parecido_com_o_centroide() -> None:
    vocabulario = normalizar(np.eye(40, 512, dtype=np.float32))
    centroides = vocabulario[[3, 17]]
    rotulos = rotular(centroides, vocabulario)
    assert len(rotulos) == 2 and rotulos[0] != rotulos[1]
