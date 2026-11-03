import uuid
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app import repository
from app.db import Sessao
from app.routes import Resposta

router = APIRouter(prefix="/api/similar/clusters", tags=["clusters"])


class ClusterResumo(BaseModel):
    id: uuid.UUID
    rotulo: str | None
    tamanho: int
    imagemRepresentativaId: uuid.UUID | None


class PaginaImagens(BaseModel):
    items: list[uuid.UUID]
    nextCursor: float | None


@router.get("", summary="Categorias da clusterização ativa")
async def listar() -> Resposta[list[ClusterResumo]]:
    async with Sessao() as sessao:
        execucao = await repository.execucao_ativa(sessao)
        if execucao is None:
            return Resposta(data=[])
        clusters = await repository.listar_clusters(sessao, execucao.id)
    return Resposta(
        data=[
            ClusterResumo(
                id=c.id,
                rotulo=c.rotulo,
                tamanho=c.tamanho,
                imagemRepresentativaId=c.imagem_representativa_id,
            )
            for c in clusters
        ]
    )


@router.get("/{cluster_id}/images", summary="Imagens de uma categoria (mais típicas primeiro)")
async def imagens(
    cluster_id: uuid.UUID,
    limit: Annotated[int, Query(ge=1, le=50)] = 20,
    cursor: Annotated[float | None, Query(ge=0)] = None,
) -> Resposta[PaginaImagens]:
    """Paginação no padrão do grupo: `{ items, nextCursor }`; nextCursor null na última página"""
    async with Sessao() as sessao:
        ativa = await repository.execucao_ativa(sessao)
        clusters = await repository.listar_clusters(sessao, ativa.id) if ativa else []
        if cluster_id not in {c.id for c in clusters}:
            raise HTTPException(404, "Categoria não encontrada na clusterização ativa")
        linhas = await repository.imagens_do_cluster(sessao, cluster_id, limit, cursor)
    proximo = linhas[-1][1] if len(linhas) == limit else None
    return Resposta(data=PaginaImagens(items=[i for i, _ in linhas], nextCursor=proximo))
