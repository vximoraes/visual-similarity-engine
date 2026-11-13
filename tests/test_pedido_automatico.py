import asyncio
import uuid

import aio_pika
import numpy as np
import pytest

from app import repository
from app.clip_model import normalizar
from app.config import config
from app.db import Sessao
from app.jobs import clusterizar

pytestmark = pytest.mark.clip


@pytest.mark.parametrize(
    ("novas", "horas", "esperado"),
    [
        (0, 100.0, False),
        (2, 1.0, False),
        (3, 1.0, True),
        (1, 24.0, True),
        (1, None, False),
        (3, None, True),
    ],
)
def test_deve_pedir(novas: int, horas: float | None, esperado: bool) -> None:
    assert clusterizar.deve_pedir(novas, horas, a_cada=3, intervalo_horas=24) is esperado


async def salvar_embeddings(quantidade: int, semente: int) -> None:
    vetores = normalizar(np.random.default_rng(semente).normal(size=(quantidade, 512)))
    itens = [repository.NovoEmbedding(uuid.uuid4(), uuid.uuid4(), v, None, None) for v in vetores]
    async with Sessao() as sessao, sessao.begin():
        await repository.salvar_embeddings(sessao, itens, "teste")


async def test_passado_o_limite_o_pedido_e_publicado_uma_vez_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config(), "clusterizar_a_cada_imagens", 3)
    monkeypatch.setattr(config(), "k_max", 4)
    await salvar_embeddings(6, semente=1)
    assert await clusterizar.executar(forcar=True) is not None

    conexao = await aio_pika.connect(config().amqp_url)
    async with conexao:
        canal = await conexao.channel()
        fila = await canal.declare_queue(exclusive=True)

        async def publicar() -> None:
            await canal.default_exchange.publish(aio_pika.Message(b"{}"), routing_key=fila.name)

        pedido = clusterizar.PedidoAutomatico(publicar)

        await salvar_embeddings(2, semente=2)
        assert await pedido.verificar() is False
        await salvar_embeddings(2, semente=3)
        assert await pedido.verificar() is True
        await salvar_embeddings(2, semente=4)
        assert await pedido.verificar() is False

        await asyncio.sleep(0.3)
        recebidos = 0
        while await fila.get(no_ack=True, fail=False) is not None:
            recebidos += 1
        assert recebidos == 1
