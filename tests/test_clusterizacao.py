import numpy as np
import pytest

from app.clip_model import normalizar
from app.clusterizacao import (
    FRASES,
    VOCABULARIO,
    AmostrasInsuficientes,
    atribuir,
    candidatos_de_k,
    clusterizar,
    rotular,
    vetores_do_vocabulario,
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


def test_acervo_menor_que_k_min_testa_de_2_ate_n_menos_1() -> None:
    assert candidatos_de_k(5, 10, 40) == [2, 3, 4]
    assert candidatos_de_k(12, 10, 40) == [10, 11]
    resultado = clusterizar(grupos(centros=3, por_grupo=3), k_min=10, k_max=40)
    assert resultado.k == 3


def test_poucas_imagens_nao_clusteriza() -> None:
    with pytest.raises(AmostrasInsuficientes):
        clusterizar(grupos(1, 2), k_min=2, k_max=10)


VOCABULARIO_SINTETICO = normalizar(np.eye(len(VOCABULARIO), 512, dtype=np.float32))
TERMOS = list(VOCABULARIO.values())


def votos(*termos_por_cluster: list[int]) -> tuple[np.ndarray, np.ndarray]:
    vetores = np.concatenate([VOCABULARIO_SINTETICO[t] for t in termos_por_cluster])
    rotulos = np.concatenate([np.full(len(t), c) for c, t in enumerate(termos_por_cluster)])
    return vetores, rotulos.astype(np.int64)


def test_rotulo_e_o_termo_mais_votado_pelas_imagens_do_cluster() -> None:
    vetores, rotulos = votos([3, 3, 5], [17, 17, 17, 2])
    assert rotular(vetores, rotulos, 2, VOCABULARIO_SINTETICO) == [TERMOS[3], TERMOS[17]]


def test_dois_clusters_nao_recebem_o_mesmo_rotulo() -> None:
    vetores, rotulos = votos([0] * 9 + [1], [0] * 6 + [2] * 4)
    assert rotular(vetores, rotulos, 2, VOCABULARIO_SINTETICO) == [TERMOS[0], TERMOS[2]]


def test_cluster_vazio_tambem_recebe_rotulo() -> None:
    vetores, rotulos = votos([4, 4])
    resultado = rotular(vetores, rotulos, 2, VOCABULARIO_SINTETICO)
    assert resultado[0] == TERMOS[4] and resultado[1] != TERMOS[4]


def test_vetor_do_termo_e_a_media_das_frases_normalizada() -> None:
    gerador = np.random.default_rng(0)
    vetores_das_frases: dict[str, np.ndarray] = {}
    chamadas: list[list[str]] = []

    def codificar(frases: list[str]) -> np.ndarray:
        chamadas.append(frases)
        for frase in frases:
            vetores_das_frases.setdefault(frase, normalizar(gerador.normal(size=(1, 512)))[0])
        return np.stack([vetores_das_frases[f] for f in frases])

    vetores = vetores_do_vocabulario(codificar, ["cat", "dog"])

    assert len(chamadas) == 1
    assert chamadas[0] == [f.format(t) for t in ("cat", "dog") for f in FRASES]
    media_do_cat = np.mean([vetores_das_frases[f.format("cat")] for f in FRASES], axis=0)
    esperado = normalizar(media_do_cat[None, :])[0]
    assert np.allclose(vetores[0], esperado, atol=1e-6)
    assert np.allclose(np.linalg.norm(vetores, axis=1), 1, atol=1e-5)


def test_vocabulario_tem_rotulos_unicos_e_tamanho_de_pinterest() -> None:
    assert 80 <= len(VOCABULARIO) <= 150
    assert len(set(VOCABULARIO.values())) == len(VOCABULARIO)
