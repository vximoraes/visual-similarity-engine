"""
Fluxo completo com CLIP de verdade: evento image.processed.post no broker -> worker gera o
embedding -> rotas de similares, busca por texto, clusterização e exclusão.
"""

import asyncio
import json
import uuid
from collections.abc import AsyncGenerator

import aio_pika
import pytest
from httpx import AsyncClient
from sqlalchemy import select

from app import repository
from app.config import config
from app.db import Sessao
from app.jobs import clusterizar
from app.models import EmbeddingImagem
from app.rabbit import EXCHANGE
from app.worker import principal
from tests.conftest import assinar, imagem_jpeg, publicar_thumb

pytestmark = pytest.mark.clip

VERMELHOS = [(220, 20, 30), (200, 30, 40), (235, 10, 20), (210, 40, 35)]
AZUIS = [(20, 40, 220), (30, 50, 200), (10, 30, 235), (40, 60, 210)]


async def publicar(routing_key: str, corpo: dict[str, str]) -> None:
    conexao = await aio_pika.connect(config().amqp_url)
    async with conexao:
        canal = await conexao.channel()
        exchange = await canal.get_exchange(EXCHANGE)
        await exchange.publish(aio_pika.Message(json.dumps(corpo).encode()), routing_key)


async def esperar_embeddings(ids: list[uuid.UUID], presentes: bool, timeout: float = 60) -> None:
    async def contar() -> int:
        async with Sessao() as sessao:
            linhas = await sessao.scalars(
                select(EmbeddingImagem.imagem_id).where(EmbeddingImagem.imagem_id.in_(ids))
            )
            return len(list(linhas))

    alvo = len(ids) if presentes else 0
    for _ in range(int(timeout * 10)):
        if await contar() == alvo:
            return
        await asyncio.sleep(0.1)
    raise AssertionError("worker não processou a tempo")


@pytest.fixture(scope="module")
async def imagens(cliente: AsyncClient) -> AsyncGenerator[dict[str, list[uuid.UUID]]]:
    """Sobe o worker de verdade e publica 8 imagens (4 vermelhas, 4 azuis) pelo broker"""
    tarefa = asyncio.create_task(principal())
    await asyncio.sleep(2)  # tempo de declarar as filas antes de publicar

    dono = uuid.uuid4()
    grupos: dict[str, list[uuid.UUID]] = {"vermelho": [], "azul": []}
    for nome, cores in (("vermelho", VERMELHOS), ("azul", AZUIS)):
        for semente, cor in enumerate(cores):
            imagem_id = uuid.uuid4()
            url = publicar_thumb(dono, imagem_id, imagem_jpeg(cor, ruido=10, semente=semente))
            grupos[nome].append(imagem_id)
            await publicar(
                "image.processed.post",
                {"imageId": str(imagem_id), "ownerId": str(dono), "thumbnailUrl": url},
            )
    await esperar_embeddings(grupos["vermelho"] + grupos["azul"], presentes=True)
    yield grupos
    tarefa.cancel()


async def test_healthz(cliente: AsyncClient) -> None:
    res = await cliente.get("/api/similar/healthz")
    assert res.status_code == 200
    assert res.json() == {"status": "ok", "banco": "ok", "rabbitmq": "ok", "redis": "ok"}


async def test_openapi_com_paths_completos(cliente: AsyncClient) -> None:
    res = await cliente.get("/api/similar/openapi.json")
    assert "/api/similar/images/{image_id}/similares" in res.json()["paths"]


async def test_similares_da_mesma_cor_vem_primeiro(
    cliente: AsyncClient, imagens: dict[str, list[uuid.UUID]]
) -> None:
    alvo = imagens["vermelho"][0]
    res = await cliente.get(f"/api/similar/images/{alvo}/similares", params={"limite": 3})

    assert res.status_code == 200
    ids = [uuid.UUID(i["imageId"]) for i in res.json()["data"]]
    assert alvo not in ids
    assert set(ids) == set(imagens["vermelho"][1:])

    # Segunda chamada vem do cache e devolve o mesmo
    assert (
        await cliente.get(f"/api/similar/images/{alvo}/similares?limite=3")
    ).json() == res.json()


async def test_similares_de_imagem_sem_embedding(cliente: AsyncClient) -> None:
    res = await cliente.get(f"/api/similar/images/{uuid.uuid4()}/similares")
    assert res.status_code == 404
    assert res.json()["data"] is None


async def test_busca_por_texto_exige_login(cliente: AsyncClient) -> None:
    res = await cliente.get("/api/similar/search-text", params={"q": "red"})
    assert res.status_code == 401
    assert res.headers["www-authenticate"] == "Bearer"


async def test_busca_por_texto(cliente: AsyncClient, imagens: dict[str, list[uuid.UUID]]) -> None:
    res = await cliente.get(
        "/api/similar/search-text",
        params={"q": "a solid blue square", "limite": 4},
        headers={"Authorization": f"Bearer {assinar()}"},
    )
    assert res.status_code == 200
    ids = {uuid.UUID(i["imageId"]) for i in res.json()["data"]}
    assert ids == set(imagens["azul"])


async def test_busca_por_texto_em_portugues(
    cliente: AsyncClient, imagens: dict[str, list[uuid.UUID]]
) -> None:
    """Sem a tradução, o CLIP não entende "quadrado azul" (só foi treinado em inglês)"""
    res = await cliente.get(
        "/api/similar/search-text",
        params={"q": "um quadrado azul", "limite": 4},
        headers={"Authorization": f"Bearer {assinar()}"},
    )
    assert res.status_code == 200
    ids = {uuid.UUID(i["imageId"]) for i in res.json()["data"]}
    assert ids == set(imagens["azul"])


async def test_token_de_outra_audiencia(cliente: AsyncClient) -> None:
    res = await cliente.get(
        "/api/similar/search-text",
        params={"q": "red"},
        headers={"Authorization": f"Bearer {assinar(aud='outro')}"},
    )
    assert res.status_code == 401


async def test_clusterizar_separa_as_cores_e_classifica_imagens_novas(
    cliente: AsyncClient, imagens: dict[str, list[uuid.UUID]]
) -> None:
    res = await cliente.post(
        "/api/similar/admin/clusterizacoes", headers={"Authorization": f"Bearer {assinar()}"}
    )
    assert res.status_code == 403

    # Rodado direto (e não pelo pedido ao worker) para o teste saber quando terminou
    config().k_max = 4
    assert await clusterizar.executar(forcar=True) is not None

    async with Sessao() as sessao:
        info = {
            i: (await repository.buscar_embedding(sessao, i))
            for i in imagens["vermelho"] + imagens["azul"]
        }
    vermelhos = {info[i].cluster_id for i in imagens["vermelho"]}  # type: ignore[union-attr]
    azuis = {info[i].cluster_id for i in imagens["azul"]}  # type: ignore[union-attr]
    assert len(vermelhos) == 1 and len(azuis) == 1 and vermelhos != azuis

    clusters = (await cliente.get("/api/similar/clusters")).json()["data"]
    assert sorted(c["tamanho"] for c in clusters) == [4, 4]

    # Imagem nova (vermelha) entra direto no cluster vermelho, sem rodar o k-means de novo
    dono, nova = uuid.uuid4(), uuid.uuid4()
    url = publicar_thumb(dono, nova, imagem_jpeg((225, 25, 25), ruido=10, semente=99))
    await publicar(
        "image.processed.post", {"imageId": str(nova), "ownerId": str(dono), "thumbnailUrl": url}
    )
    await esperar_embeddings([nova], presentes=True)
    res = await cliente.get(f"/api/similar/images/{nova}")
    assert uuid.UUID(res.json()["data"]["clusterId"]) in vermelhos

    pagina = await cliente.get(f"/api/similar/clusters/{next(iter(vermelhos))}/images?limit=2")
    assert len(pagina.json()["data"]["items"]) == 2
    assert pagina.json()["data"]["nextCursor"] is not None


async def test_exclusao_remove_e_nao_volta(
    cliente: AsyncClient, imagens: dict[str, list[uuid.UUID]]
) -> None:
    alvo = imagens["azul"][-1]
    async with Sessao() as sessao:
        dono = (await repository.buscar_embedding(sessao, alvo)).dono_id  # type: ignore[union-attr]
    corpo = {"postId": str(uuid.uuid4()), "imageId": str(alvo), "ownerId": str(dono)}

    await publicar("image.deleted.post", corpo)
    await esperar_embeddings([alvo], presentes=False)

    # Um image.processed.post atrasado (fora de ordem) não recria o embedding
    await publicar("image.processed.post", {"imageId": str(alvo), "ownerId": str(dono)})
    await asyncio.sleep(1.5)
    await esperar_embeddings([alvo], presentes=False, timeout=1)
