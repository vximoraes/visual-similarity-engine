import uuid
from dataclasses import dataclass

import numpy as np
from sqlalchemy import delete, func, select, text, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.clip_model import Vetores
from app.models import Cluster, EmbeddingImagem, ExecucaoClusterizacao, ImagemExcluida


@dataclass(frozen=True)
class NovoEmbedding:
    imagem_id: uuid.UUID
    dono_id: uuid.UUID
    vetor: Vetores
    cluster_id: uuid.UUID | None
    distancia: float | None


async def ids_excluidos(sessao: AsyncSession, ids: list[uuid.UUID]) -> set[uuid.UUID]:
    resultado = await sessao.scalars(
        select(ImagemExcluida.imagem_id).where(ImagemExcluida.imagem_id.in_(ids))
    )
    return set(resultado)


async def salvar_embeddings(sessao: AsyncSession, itens: list[NovoEmbedding], modelo: str) -> None:
    """Upsert: a mesma imagem processada de novo só sobrescreve (idempotente)"""
    if not itens:
        return
    comando = insert(EmbeddingImagem).values(
        [
            {
                "imagem_id": i.imagem_id,
                "dono_id": i.dono_id,
                "embedding": i.vetor,
                "modelo": modelo,
                "cluster_id": i.cluster_id,
                "distancia_centroide": i.distancia,
            }
            for i in itens
        ]
    )
    await sessao.execute(
        comando.on_conflict_do_update(
            index_elements=[EmbeddingImagem.imagem_id],
            set_={
                "embedding": comando.excluded.embedding,
                "modelo": comando.excluded.modelo,
                "cluster_id": comando.excluded.cluster_id,
                "distancia_centroide": comando.excluded.distancia_centroide,
                "atualizado_em": func.now(),
            },
        )
    )


async def excluir_embedding(sessao: AsyncSession, imagem_id: uuid.UUID) -> None:
    await sessao.execute(delete(EmbeddingImagem).where(EmbeddingImagem.imagem_id == imagem_id))
    await sessao.execute(
        insert(ImagemExcluida).values(imagem_id=imagem_id).on_conflict_do_nothing()
    )


async def buscar_embedding(sessao: AsyncSession, imagem_id: uuid.UUID) -> EmbeddingImagem | None:
    return await sessao.get(EmbeddingImagem, imagem_id)


async def vizinhos(
    sessao: AsyncSession,
    vetor: Vetores,
    limite: int,
    ef_search: int,
    excluir: uuid.UUID | None = None,
) -> list[tuple[uuid.UUID, float]]:
    """K vizinhos mais próximos pelo HNSW (distância de cosseno). Retorna (id, similaridade)"""
    # SET LOCAL vale só para esta transação; o valor não vem do usuário
    await sessao.execute(text(f"SET LOCAL hnsw.ef_search = {int(ef_search)}"))
    distancia = EmbeddingImagem.embedding.cosine_distance(vetor)
    consulta = select(EmbeddingImagem.imagem_id, (1 - distancia).label("s")).order_by(distancia)
    if excluir is not None:
        consulta = consulta.where(EmbeddingImagem.imagem_id != excluir)
    linhas = await sessao.execute(consulta.limit(limite))
    return [(linha.imagem_id, float(linha.s)) for linha in linhas]


async def execucao_ativa(sessao: AsyncSession) -> ExecucaoClusterizacao | None:
    return await sessao.scalar(
        select(ExecucaoClusterizacao).where(ExecucaoClusterizacao.ativa.is_(True))
    )


async def centroides_ativos(
    sessao: AsyncSession,
) -> tuple[uuid.UUID | None, list[uuid.UUID], Vetores]:
    """(execução ativa, ids dos clusters, matriz de centroides) para classificar imagens novas"""
    execucao = await execucao_ativa(sessao)
    if execucao is None:
        return None, [], np.zeros((0, 512), dtype=np.float32)
    clusters = list(
        await sessao.scalars(
            select(Cluster).where(Cluster.execucao_id == execucao.id).order_by(Cluster.indice)
        )
    )
    return (
        execucao.id,
        [c.id for c in clusters],
        np.array([c.centroide for c in clusters], dtype=np.float32),
    )


async def todos_embeddings(sessao: AsyncSession) -> tuple[list[uuid.UUID], Vetores]:
    linhas = (
        await sessao.execute(select(EmbeddingImagem.imagem_id, EmbeddingImagem.embedding))
    ).all()
    ids = [linha.imagem_id for linha in linhas]
    vetores = np.array([linha.embedding for linha in linhas], dtype=np.float32)
    return ids, vetores.reshape(len(ids), 512)


async def embeddings_por_id(sessao: AsyncSession, ids: list[uuid.UUID]) -> dict[uuid.UUID, Vetores]:
    linhas = await sessao.execute(
        select(EmbeddingImagem.imagem_id, EmbeddingImagem.embedding).where(
            EmbeddingImagem.imagem_id.in_(ids)
        )
    )
    return {linha.imagem_id: np.asarray(linha.embedding, dtype=np.float32) for linha in linhas}


async def listar_clusters(sessao: AsyncSession, execucao_id: uuid.UUID) -> list[Cluster]:
    return list(
        await sessao.scalars(
            select(Cluster)
            .where(Cluster.execucao_id == execucao_id)
            .order_by(Cluster.tamanho.desc())
        )
    )


async def imagens_do_cluster(
    sessao: AsyncSession, cluster_id: uuid.UUID, limite: int, apos: float | None
) -> list[tuple[uuid.UUID, float]]:
    """Mais típicas primeiro (menor distância ao centroide). Cursor = última distância vista"""
    consulta = (
        select(EmbeddingImagem.imagem_id, EmbeddingImagem.distancia_centroide)
        .where(EmbeddingImagem.cluster_id == cluster_id)
        .order_by(EmbeddingImagem.distancia_centroide, EmbeddingImagem.imagem_id)
        .limit(limite)
    )
    if apos is not None:
        consulta = consulta.where(EmbeddingImagem.distancia_centroide > apos)
    linhas = await sessao.execute(consulta)
    return [(linha.imagem_id, float(linha.distancia_centroide or 0)) for linha in linhas]


async def criar_execucao(sessao: AsyncSession, modelo: str, amostras: int) -> uuid.UUID:
    execucao = ExecucaoClusterizacao(
        modelo=modelo, algoritmo="minibatch-kmeans-esferico", amostras=amostras, status="executando"
    )
    sessao.add(execucao)
    await sessao.flush()
    return execucao.id


async def ativar_execucao(
    sessao: AsyncSession,
    execucao_id: uuid.UUID,
    k: int,
    silhueta: float,
    clusters: list[Cluster],
    atribuicoes: list[dict[str, object]],
) -> None:
    """Grava clusters, reclassifica todas as imagens e troca a execução ativa (mesma transação)"""
    sessao.add_all(clusters)
    await sessao.flush()
    await sessao.execute(update(EmbeddingImagem), atribuicoes)
    await sessao.execute(
        update(ExecucaoClusterizacao)
        .where(ExecucaoClusterizacao.ativa.is_(True))
        .values(ativa=False)
    )
    await sessao.execute(
        update(ExecucaoClusterizacao)
        .where(ExecucaoClusterizacao.id == execucao_id)
        .values(status="pronta", ativa=True, k=k, silhueta=silhueta, ativado_em=func.now())
    )
