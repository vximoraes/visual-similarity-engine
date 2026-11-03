import asyncio
import uuid
from typing import Annotated

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel

from app import cache, repository
from app.auth import Usuario, usuario_atual
from app.config import config
from app.db import Sessao
from app.routes import Resposta

router = APIRouter(prefix="/api/similar", tags=["similares"])


class ItemSimilar(BaseModel):
    imageId: uuid.UUID
    similaridade: float


class ImagemInfo(BaseModel):
    imageId: uuid.UUID
    clusterId: uuid.UUID | None
    distanciaCentroide: float | None


Limite = Annotated[int, Query(ge=1, le=100)]


@router.get("/images/{image_id}/similares", summary="Imagens semelhantes a uma imagem")
async def similares_da_imagem(
    request: Request, image_id: uuid.UUID, limite: Limite = 30
) -> Resposta[list[ItemSimilar]]:
    """
    Vizinhos mais próximos pelo embedding CLIP (HNSW, cosseno), sem a própria imagem. A lista de
    ids vai para `POST /api/posts/by-images`, que devolve os posts na mesma ordem.
    """
    em_cache = await cache.ler_similares(request.app.state.redis, image_id, limite)
    if em_cache is not None:
        return Resposta(data=[ItemSimilar.model_validate(i) for i in em_cache])

    async with Sessao() as sessao, sessao.begin():
        alvo = await repository.buscar_embedding(sessao, image_id)
        if alvo is None:
            raise HTTPException(404, "Imagem sem embedding (inexistente ou ainda processando)")
        vetor = np.asarray(alvo.embedding, dtype=np.float32)
        itens = await repository.vizinhos(
            sessao, vetor, limite, config().hnsw_ef_search, excluir=image_id
        )

    resposta = [ItemSimilar(imageId=i, similaridade=round(s, 4)) for i, s in itens]
    await cache.gravar_similares(
        request.app.state.redis, image_id, limite, [r.model_dump(mode="json") for r in resposta]
    )
    return Resposta(data=resposta)


@router.get("/search-text", summary="Busca imagens por descrição em texto")
async def buscar_por_texto(
    request: Request,
    usuario: Annotated[Usuario, Depends(usuario_atual)],
    q: Annotated[str, Query(min_length=2, max_length=200)],
    limite: Limite = 30,
) -> Resposta[list[ItemSimilar]]:
    """Codifica o texto com o encoder de texto do CLIP (mesmo espaço das imagens) e busca"""
    vetor = await cache.ler_vetor_texto(request.app.state.redis, q)
    if vetor is None:
        vetor = (await asyncio.to_thread(request.app.state.clip.codificar_textos, [q]))[0]
        await cache.gravar_vetor_texto(request.app.state.redis, q, vetor)

    async with Sessao() as sessao, sessao.begin():
        itens = await repository.vizinhos(sessao, vetor, limite, config().hnsw_ef_search)
    return Resposta(data=[ItemSimilar(imageId=i, similaridade=round(s, 4)) for i, s in itens])


@router.get("/images/{image_id}", summary="Categoria (cluster) de uma imagem")
async def info_da_imagem(image_id: uuid.UUID) -> Resposta[ImagemInfo]:
    async with Sessao() as sessao:
        alvo = await repository.buscar_embedding(sessao, image_id)
    if alvo is None:
        raise HTTPException(404, "Imagem sem embedding (inexistente ou ainda processando)")
    return Resposta(
        data=ImagemInfo(
            imageId=alvo.imagem_id,
            clusterId=alvo.cluster_id,
            distanciaCentroide=alvo.distancia_centroide,
        )
    )
