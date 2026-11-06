"""Métricas de avaliação com vetores sintéticos (rápido, sem banco e sem CLIP)"""

import csv
from pathlib import Path

import numpy as np
import pytest

from app import avaliacao
from app.clip_model import normalizar


def grupos(centros: list[list[float]], por_grupo: int, ruido: float = 0.05) -> np.ndarray:
    gerador = np.random.default_rng(0)
    pontos = [
        np.array(centro) + gerador.normal(0, ruido, size=len(centro))
        for centro in centros
        for _ in range(por_grupo)
    ]
    return normalizar(np.array(pontos, dtype=np.float32))


def test_vizinhos_exatos_nao_incluem_a_propria_imagem_e_vem_em_ordem() -> None:
    vetores = normalizar(np.array([[1, 0], [0.9, 0.1], [0.4, 0.6], [0, 1]], dtype=np.float32))
    vizinhos = avaliacao.vizinhos_exatos(vetores, k=2)

    assert vizinhos.tolist() == [[1, 2], [0, 2], [3, 1], [2, 1]]


def test_vizinhos_exatos_em_blocos_dao_o_mesmo_resultado() -> None:
    vetores = grupos([[1, 0, 0], [0, 1, 0], [0, 0, 1]], por_grupo=7)
    inteiro = avaliacao.vizinhos_exatos(vetores, k=5)

    assert (avaliacao.vizinhos_exatos(vetores, k=5, bloco=4) == inteiro).all()


def test_precisao_com_grupos_separados_e_total() -> None:
    vetores = grupos([[1, 0, 0], [0, 1, 0], [0, 0, 1]], por_grupo=11)
    categorias = ["a"] * 11 + ["b"] * 11 + ["c"] * 11

    resultado = avaliacao.precisao_dos_vizinhos(vetores, categorias, k=10)

    assert resultado.geral == 1.0
    assert resultado.por_categoria == {"a": 1.0, "b": 1.0, "c": 1.0}
    assert all(not erros for erros in resultado.confusoes.values())


def test_precisao_conta_os_vizinhos_de_outra_categoria() -> None:
    # Dois grupos de 3 pontos; k=4 obriga cada imagem a pegar 2 do mesmo grupo e 2 do outro
    vetores = grupos([[1, 0], [0, 1]], por_grupo=3, ruido=0.01)
    categorias = ["praia"] * 3 + ["montanhas"] * 3

    resultado = avaliacao.precisao_dos_vizinhos(vetores, categorias, k=4)

    assert resultado.geral == pytest.approx(0.5)
    assert resultado.confusoes["praia"] == {"montanhas": 6}
    assert resultado.vizinhos.shape == (6, 4)


def test_precisao_exige_mais_imagens_que_k() -> None:
    with pytest.raises(ValueError, match="não bastam"):
        avaliacao.vizinhos_exatos(grupos([[1, 0]], por_grupo=10), k=10)


def test_precisao_do_texto() -> None:
    vetores = grupos([[1, 0], [0, 1]], por_grupo=3, ruido=0.01)
    categorias = ["a"] * 3 + ["b"] * 3
    textos = normalizar(np.array([[1, 0], [0.2, 1]], dtype=np.float32))

    precisoes, resultados = avaliacao.precisao_do_texto(textos, ["a", "a"], vetores, categorias, 3)

    assert precisoes == [1.0, 0.0]
    assert set(resultados[1].tolist()) == {3, 4, 5}


def test_toda_categoria_da_base_tem_consulta_de_texto() -> None:
    grupos = Path(__file__).resolve().parent.parent / "scripts" / "imagenet" / "grupos.csv"
    with grupos.open(newline="", encoding="utf-8") as arquivo:
        categorias = {linha["grupo"] for linha in csv.DictReader(arquivo)}

    assert categorias - {avaliacao.SEM_CATEGORIA} == set(avaliacao.CONSULTAS)


def test_sem_categoria_conta_como_vizinho_errado_mas_nao_e_avaliada() -> None:
    # 3 "a" juntos e 3 "outros" colados neles: com k=4, cada "a" pega 2 "a" e 2 "outros"
    vetores = grupos([[1, 0], [0.98, 0.2]], por_grupo=3, ruido=0.001)
    categorias = ["a"] * 3 + [avaliacao.SEM_CATEGORIA] * 3

    resultado = avaliacao.precisao_dos_vizinhos(vetores, categorias, k=4)

    assert resultado.geral == pytest.approx(0.5)
    assert resultado.por_categoria == {"a": pytest.approx(0.5)}
    assert set(resultado.confusoes) == {"a"}
    assert resultado.confusoes["a"] == {avaliacao.SEM_CATEGORIA: 6}


def test_base_so_com_sem_categoria() -> None:
    vetores = grupos([[1, 0]], por_grupo=12)
    with pytest.raises(ValueError, match="nenhuma imagem com categoria"):
        avaliacao.precisao_dos_vizinhos(vetores, [avaliacao.SEM_CATEGORIA] * 12)


def test_texto_com_menos_imagens_que_k() -> None:
    vetores = grupos([[1, 0]], por_grupo=3)
    with pytest.raises(ValueError, match="não bastam"):
        avaliacao.precisao_do_texto(vetores[:1], ["a"], vetores, ["a"] * 3, k=10)
