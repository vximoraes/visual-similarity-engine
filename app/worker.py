"""
Worker: consome os eventos de imagem do Simila e mantém os embeddings.

- image.processed.post: baixa a thumbnail 224 (pública), gera o embedding com o CLIP em lote,
  classifica no centroide mais próximo da clusterização ativa e grava no pgvector.
- image.deleted.post: apaga o embedding e registra o tombstone.
- vse.comandos.clusterizar: roda o k-means (pedido pela rota admin).

Rodar: `uv run python -m app.worker`
"""

import asyncio
import io
import json
import logging
import re
import uuid
from dataclasses import dataclass
from typing import Any

import aio_pika
import httpx
import numpy as np
from aio_pika.abc import AbstractIncomingMessage
from PIL import Image, UnidentifiedImageError

from app import cache, clip_model, clusterizacao, repository
from app.config import config
from app.db import Sessao
from app.jobs import clusterizar
from app.rabbit import FILA_CLUSTERIZAR, declarar_exchange, declarar_fila, declarar_fila_do_evento

log = logging.getLogger("worker")

TENTATIVAS_DOWNLOAD = 3

UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)


class PayloadInvalido(Exception):
    pass


def ler_ids(corpo: bytes) -> tuple[uuid.UUID, uuid.UUID, dict[str, Any]]:
    """Valida imageId e ownerId. Campos a mais são aceitos (contrato do grupo)"""
    try:
        dados = json.loads(corpo)
    except ValueError as erro:
        raise PayloadInvalido("JSON inválido") from erro
    if not isinstance(dados, dict):
        raise PayloadInvalido("payload não é objeto")
    for campo in ("imageId", "ownerId"):
        if not isinstance(dados.get(campo), str) or not UUID.match(dados[campo]):
            raise PayloadInvalido(f"campo {campo} ausente ou não é UUID")
    return uuid.UUID(dados["imageId"]), uuid.UUID(dados["ownerId"]), dados


def url_da_thumbnail(imagem_id: uuid.UUID, dono_id: uuid.UUID, dados: dict[str, Any]) -> str:
    """O simila-imagens manda thumbnailUrl; sem ela, monta pelo layout <dono>/post/<id>_thumb.jpg"""
    url = dados.get("thumbnailUrl")
    if isinstance(url, str) and url.startswith(("http://", "https://")):
        return url
    return f"{config().media_base_url}/{dono_id}/post/{imagem_id}_thumb.jpg"


@dataclass
class Pendente:
    mensagem: AbstractIncomingMessage
    imagem_id: uuid.UUID
    dono_id: uuid.UUID
    url: str


class Centroides:
    """Centroides da clusterização ativa em memória, recarregados quando a execução ativa muda"""

    def __init__(self) -> None:
        self.execucao_id: uuid.UUID | None = None
        self.ids: list[uuid.UUID] = []
        self.matriz = np.zeros((0, 512), dtype=np.float32)

    async def atualizar(self) -> None:
        async with Sessao() as sessao:
            ativa = await repository.execucao_ativa(sessao)
            if (ativa.id if ativa else None) == self.execucao_id:
                return
            self.execucao_id, self.ids, self.matriz = await repository.centroides_ativos(sessao)
        log.info("Centroides carregados: execução %s, k=%d", self.execucao_id, len(self.ids))

    def classificar(
        self, vetores: clip_model.Vetores
    ) -> list[tuple[uuid.UUID | None, float | None]]:
        if len(self.ids) == 0:
            return [(None, None)] * len(vetores)
        indices, distancias = clusterizacao.atribuir(vetores, self.matriz)
        return [(self.ids[i], float(d)) for i, d in zip(indices, distancias, strict=True)]


class Worker:
    def __init__(self) -> None:
        self.fila: asyncio.Queue[Pendente] = asyncio.Queue()
        self.http = httpx.AsyncClient(timeout=10, follow_redirects=False)
        self.redis = cache.criar_redis()
        self.modelo = clip_model.carregar()
        self.centroides = Centroides()

    async def ao_processar(self, mensagem: AbstractIncomingMessage) -> None:
        try:
            imagem_id, dono_id, dados = ler_ids(mensagem.body)
        except PayloadInvalido as erro:
            log.error("image.processed.post inválido (%s): %s", erro, mensagem.body[:300])
            await mensagem.nack(requeue=False)
            return
        await self.fila.put(
            Pendente(mensagem, imagem_id, dono_id, url_da_thumbnail(imagem_id, dono_id, dados))
        )

    async def get_com_retentativa(self, url: str) -> httpx.Response:
        """Erro de rede passageiro tenta de novo (0,2 s, 0,4 s) antes de ir para a DLQ"""
        for tentativa in range(TENTATIVAS_DOWNLOAD):
            try:
                resposta = await self.http.get(url)
                if resposta.status_code < 500 or tentativa == TENTATIVAS_DOWNLOAD - 1:
                    return resposta
            except httpx.TransportError:
                if tentativa == TENTATIVAS_DOWNLOAD - 1:
                    raise
            await asyncio.sleep(0.2 * 2**tentativa)
        raise AssertionError("inalcançável")

    async def baixar(self, pendente: Pendente) -> Image.Image | None:
        """None quando a imagem não existe mais (excluída no meio do caminho): não adianta tentar"""
        resposta = await self.get_com_retentativa(pendente.url)
        if resposta.status_code == 404:
            return None
        resposta.raise_for_status()
        imagem = Image.open(io.BytesIO(resposta.content))
        imagem.load()
        return imagem

    async def processar_lote(self, lote: list[Pendente]) -> None:
        """
        Uma inferência do CLIP para o lote inteiro (bem mais rápido que uma por imagem). Falha de
        uma imagem não derruba as outras: ela vai sozinha para a DLQ.
        """
        await self.centroides.atualizar()
        async with Sessao() as sessao:
            excluidas = await repository.ids_excluidos(sessao, [p.imagem_id for p in lote])

        baixadas = await asyncio.gather(
            *(self.baixar(p) for p in lote if p.imagem_id not in excluidas), return_exceptions=True
        )
        validos: list[tuple[Pendente, Image.Image]] = []
        for pendente, resultado in zip(
            [p for p in lote if p.imagem_id not in excluidas], baixadas, strict=True
        ):
            if isinstance(resultado, UnidentifiedImageError):
                log.error("Thumbnail de %s não é imagem", pendente.imagem_id)
                await pendente.mensagem.nack(requeue=False)
            elif isinstance(resultado, BaseException):
                log.error("Falha ao baixar %s: %s", pendente.url, resultado)
                await pendente.mensagem.nack(requeue=False)
            elif resultado is None:
                log.warning("Thumbnail de %s não existe mais, ignorada", pendente.imagem_id)
                await pendente.mensagem.ack()
            else:
                validos.append((pendente, resultado))
        for pendente in lote:
            if pendente.imagem_id in excluidas:
                await pendente.mensagem.ack()

        if not validos:
            return
        vetores = await asyncio.to_thread(
            self.modelo.codificar_imagens, [img for _, img in validos]
        )
        classes = self.centroides.classificar(vetores)
        itens = [
            repository.NovoEmbedding(p.imagem_id, p.dono_id, vetores[i], *classes[i])
            for i, (p, _) in enumerate(validos)
        ]
        try:
            async with Sessao() as sessao, sessao.begin():
                await repository.salvar_embeddings(sessao, itens, self.modelo.nome)
        except Exception:
            log.exception("Falha ao gravar o lote")
            for pendente, _ in validos:
                await pendente.mensagem.nack(requeue=False)
            return
        await cache.invalidar_similares(self.redis, [p.imagem_id for p, _ in validos])
        for pendente, _ in validos:
            await pendente.mensagem.ack()
        log.info("Lote de %d embeddings gravado", len(validos))

    async def loop_de_lotes(self) -> None:
        """Junta até LOTE_MAX mensagens ou espera LOTE_ESPERA_MS, o que vier primeiro"""
        while True:
            lote = [await self.fila.get()]
            prazo = asyncio.get_running_loop().time() + config().lote_espera_ms / 1000
            while len(lote) < config().lote_max:
                restante = prazo - asyncio.get_running_loop().time()
                if restante <= 0:
                    break
                try:
                    lote.append(await asyncio.wait_for(self.fila.get(), restante))
                except TimeoutError:
                    break
            try:
                await self.processar_lote(lote)
            except Exception:
                log.exception("Erro inesperado no lote; mensagens vão para a DLQ")
                for pendente in lote:
                    if not pendente.mensagem.processed:
                        await pendente.mensagem.nack(requeue=False)

    async def ao_excluir(self, mensagem: AbstractIncomingMessage) -> None:
        try:
            imagem_id, _, _ = ler_ids(mensagem.body)
            async with Sessao() as sessao, sessao.begin():
                await repository.excluir_embedding(sessao, imagem_id)
            await cache.invalidar_similares(self.redis, [imagem_id])
            await mensagem.ack()
        except Exception:
            log.exception("Falha em image.deleted.post: %s", mensagem.body[:300])
            await mensagem.nack(requeue=False)

    async def ao_pedir_clusterizacao(self, mensagem: AbstractIncomingMessage) -> None:
        # ack antes: o k-means pode demorar e o pedido é idempotente (só roda se houver novidade)
        await mensagem.ack()
        try:
            await clusterizar.executar()
        except Exception:
            log.exception("Falha na clusterização")


async def principal() -> None:
    worker = Worker()
    conexao = await aio_pika.connect_robust(config().amqp_url)

    # Canais separados: o prefetch do lote não deve limitar as exclusões nem os pedidos de k-means
    canal_lotes = await conexao.channel()
    await canal_lotes.set_qos(prefetch_count=config().lote_max * 2)
    exchange = await declarar_exchange(canal_lotes)
    fila_processadas = await declarar_fila_do_evento(canal_lotes, exchange, "image.processed.post")

    canal_outros = await conexao.channel()
    await canal_outros.set_qos(prefetch_count=1)
    exchange_outros = await declarar_exchange(canal_outros)
    fila_excluidas = await declarar_fila_do_evento(
        canal_outros, exchange_outros, "image.deleted.post"
    )
    fila_comandos = await declarar_fila(canal_outros, FILA_CLUSTERIZAR)

    await fila_processadas.consume(worker.ao_processar)
    await fila_excluidas.consume(worker.ao_excluir)
    await fila_comandos.consume(worker.ao_pedir_clusterizacao)
    log.info("Worker iniciado (lote até %d)", config().lote_max)
    await worker.loop_de_lotes()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    asyncio.run(principal())
