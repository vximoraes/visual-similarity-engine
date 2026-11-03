"""
Recalcula as categorias: k-means sobre todos os embeddings, rótulo automático por zero-shot do
CLIP e troca atômica da execução ativa.

Rodar à mão: `uv run python -m app.jobs.clusterizar` (ou pela rota admin, que pede ao worker).
"""

import asyncio
import logging
import uuid

import numpy as np
from sqlalchemy import func, select, text, update

from app import clip_model, clusterizacao, repository
from app.config import config
from app.db import Sessao
from app.models import Cluster, EmbeddingImagem, ExecucaoClusterizacao

log = logging.getLogger(__name__)

# Trava do Postgres: só uma clusterização por vez, mesmo com várias réplicas do worker
TRAVA_CLUSTERIZACAO = 7_310_001


async def precisa_recalcular() -> bool:
    """Evita refazer o k-means quando nada mudou desde a última execução ativa"""
    async with Sessao() as sessao:
        ativa = await repository.execucao_ativa(sessao)
        if ativa is None or ativa.ativado_em is None:
            return True
        total, ultima = (
            await sessao.execute(select(func.count(), func.max(EmbeddingImagem.atualizado_em)))
        ).one()
    return bool(total != ativa.amostras or (ultima and ultima > ativa.ativado_em))


async def executar(forcar: bool = False) -> uuid.UUID | None:
    if not forcar and not await precisa_recalcular():
        log.info("Clusterização ignorada: nenhuma imagem nova desde a última")
        return None

    async with Sessao() as sessao, sessao.begin():
        if not await sessao.scalar(
            text("select pg_try_advisory_xact_lock(:t)"), {"t": TRAVA_CLUSTERIZACAO}
        ):
            log.info("Clusterização ignorada: outra já está rodando")
            return None

        ids, vetores = await repository.todos_embeddings(sessao)
        modelo = clip_model.carregar()
        execucao_id = await repository.criar_execucao(sessao, modelo.nome, len(ids))

        try:
            resultado = await asyncio.to_thread(
                clusterizacao.clusterizar, vetores, config().k_min, config().k_max
            )
        except clusterizacao.AmostrasInsuficientes as erro:
            await sessao.execute(
                update(ExecucaoClusterizacao)
                .where(ExecucaoClusterizacao.id == execucao_id)
                .values(status="falhou", motivo_falha=str(erro))
            )
            log.warning("Clusterização não executada: %s", erro)
            return execucao_id

        termos = [f"a photo of {termo}" for termo in clusterizacao.VOCABULARIO]
        vetores_vocabulario = await asyncio.to_thread(modelo.codificar_textos, termos)
        rotulos = clusterizacao.rotular(resultado.centroides, vetores_vocabulario)

        clusters: list[Cluster] = []
        for indice in range(resultado.k):
            membros = np.flatnonzero(resultado.rotulos == indice)
            representativa = (
                ids[membros[resultado.distancias[membros].argmin()]] if len(membros) else None
            )
            clusters.append(
                Cluster(
                    id=uuid.uuid7(),
                    execucao_id=execucao_id,
                    indice=indice,
                    centroide=resultado.centroides[indice],
                    rotulo=rotulos[indice],
                    tamanho=int(len(membros)),
                    imagem_representativa_id=representativa,
                )
            )

        atribuicoes: list[dict[str, object]] = [
            {
                "imagem_id": imagem_id,
                "cluster_id": clusters[int(resultado.rotulos[i])].id,
                "distancia_centroide": float(resultado.distancias[i]),
            }
            for i, imagem_id in enumerate(ids)
        ]
        await repository.ativar_execucao(
            sessao, execucao_id, resultado.k, resultado.silhueta, clusters, atribuicoes
        )

    log.info(
        "Clusterização %s ativa: k=%d, silhueta=%.3f, %d imagens",
        execucao_id,
        resultado.k,
        resultado.silhueta,
        len(ids),
    )
    return execucao_id


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(executar(forcar=True))
