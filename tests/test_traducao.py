"""Tradução da busca por texto com o modelo de verdade (download de ~300 MB na primeira vez)"""

import numpy as np
import pytest

from app import cache, clip_model, traducao
from app.config import config

pytestmark = pytest.mark.clip


def test_traduz_portugues_para_ingles() -> None:
    tradutor = traducao.carregar()
    assert tradutor is not None

    traduzidas = [t.lower() for t in tradutor.traduzir(["pássaro", "móveis", "um gato preto"])]

    assert traduzidas[0].startswith("bird")
    assert "furniture" in traduzidas[1]
    assert "black cat" in traduzidas[2]


def test_consulta_traduzida_cai_perto_do_ingles() -> None:
    clip = clip_model.carregar()
    tradutor = traducao.carregar()

    pt = traducao.codificar_consultas(clip, tradutor, ["pássaro"])[0]
    sem_traducao = clip.codificar_textos(["pássaro"])[0]
    ingles = clip.codificar_textos(["bird"])[0]

    assert float(np.dot(pt, ingles)) > float(np.dot(sem_traducao, ingles))
    assert float(np.dot(pt, ingles)) > 0.95


def test_desligada_nao_carrega_modelo_e_muda_a_chave_do_cache() -> None:
    ligada = cache._chave_texto("gato")
    config().traduzir_busca = False
    try:
        assert traducao.carregar() is None
        assert cache._chave_texto("gato") != ligada
        assert "sem-traducao" in cache._chave_texto("gato")
    finally:
        config().traduzir_busca = True
