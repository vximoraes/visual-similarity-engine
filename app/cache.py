"""
Cache no Redis. Opcional por natureza: se o Redis cair, as rotas continuam (só ficam mais lentas).

- similares:<imagem_id> (hash, campo = limite): resultado da busca de vizinhos. Apagado quando a
  imagem é excluída ou o vetor muda; o TTL cobre imagens novas que passariam a entrar na lista.
- texto:<modelo>:<tradutor>:<sha1(texto)>: embedding de uma busca por texto (tradução e encoder
  custam ~110 ms).
"""

import hashlib
import json
import logging
import uuid

import numpy as np
from redis.asyncio import Redis

from app.clip_model import Vetores
from app.config import config

log = logging.getLogger(__name__)


def criar_redis() -> Redis:
    return Redis.from_url(config().redis_url, socket_timeout=0.5, socket_connect_timeout=0.5)


def _chave_similares(imagem_id: uuid.UUID) -> str:
    return f"similares:{imagem_id}"


async def ler_similares(
    redis: Redis, imagem_id: uuid.UUID, limite: int
) -> list[dict[str, object]] | None:
    try:
        valor = await redis.hget(_chave_similares(imagem_id), str(limite))
    except Exception:
        log.warning("Redis indisponível (leitura de similares)")
        return None
    return json.loads(valor) if valor else None


async def gravar_similares(
    redis: Redis, imagem_id: uuid.UUID, limite: int, itens: list[dict[str, object]]
) -> None:
    chave = _chave_similares(imagem_id)
    try:
        async with redis.pipeline(transaction=False) as pipe:
            pipe.hset(chave, str(limite), json.dumps(itens))
            pipe.expire(chave, config().cache_ttl_similares)
            await pipe.execute()
    except Exception:
        log.warning("Redis indisponível (gravação de similares)")


async def invalidar_similares(redis: Redis, ids: list[uuid.UUID]) -> None:
    if not ids:
        return
    try:
        await redis.delete(*[_chave_similares(i) for i in ids])
    except Exception:
        log.warning("Redis indisponível (invalidação)")


def _chave_texto(texto: str) -> str:
    """O vetor depende do CLIP e de a consulta ter sido traduzida ou não (e por qual modelo)"""
    resumo = hashlib.sha1(texto.strip().lower().encode()).hexdigest()
    tradutor = config().traducao_modelo if config().traduzir_busca else "sem-traducao"
    return f"texto:{config().clip_modelo}:{tradutor}:{resumo}"


async def ler_vetor_texto(redis: Redis, texto: str) -> Vetores | None:
    try:
        valor = await redis.get(_chave_texto(texto))
    except Exception:
        return None
    return np.frombuffer(valor, dtype=np.float32) if isinstance(valor, bytes) else None


async def gravar_vetor_texto(redis: Redis, texto: str, vetor: Vetores) -> None:
    try:
        await redis.set(
            _chave_texto(texto), vetor.astype(np.float32).tobytes(), ex=config().cache_ttl_texto
        )
    except Exception:
        log.warning("Redis indisponível (vetor de texto)")


async def ping_redis(redis: Redis) -> bool:
    return bool(await redis.ping())
