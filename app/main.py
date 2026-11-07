import asyncio
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from typing import Any

import aio_pika
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import clip_model, traducao
from app.auth import criar_jwks
from app.cache import criar_redis, ping_redis
from app.config import config
from app.db import engine, ping_banco
from app.rabbit import FILA_CLUSTERIZAR, declarar_exchange, declarar_fila
from app.routes import admin, clusters, similares

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    app.state.jwks = criar_jwks()
    app.state.origens = config().origens_permitidas()
    app.state.redis = criar_redis()
    # O encoder de texto do CLIP fica na API (busca por texto). Carregar no start evita que a
    # primeira busca espere ~2 s
    app.state.clip = await asyncio.to_thread(clip_model.carregar)
    app.state.tradutor = await asyncio.to_thread(traducao.carregar)
    # Conexão robusta: reconecta sozinha; a API sobe mesmo com o broker fora (healthz fica 503)
    app.state.amqp = await aio_pika.connect_robust(config().amqp_url, fail_fast=False)
    canal = await app.state.amqp.channel()
    await declarar_exchange(canal)
    await declarar_fila(canal, FILA_CLUSTERIZAR)
    app.state.canal = canal
    yield
    await app.state.amqp.close()
    await app.state.redis.aclose()
    await engine.dispose()


# O gateway repassa o path inteiro: as rotas já começam com /api/similar (sem root_path).
# Sem interface de docs própria: o Scalar em /api/docs/ lê o openapi.json
app = FastAPI(
    title="Similaridade (visual-similarity-engine)",
    version="0.1.0",
    openapi_url="/api/similar/openapi.json",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)


def erro(
    status: int, errors: list[dict[str, Any]], headers: Mapping[str, str] | None = None
) -> JSONResponse:
    """Resposta de erro no padrão das APIs do Simila: { data: null, errors: [...] }"""
    return JSONResponse({"data": None, "errors": errors}, status_code=status, headers=headers)


@app.exception_handler(StarletteHTTPException)
async def http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    # Mantém os headers (ex.: WWW-Authenticate do 401)
    return erro(exc.status_code, [{"message": str(exc.detail)}], exc.headers)


@app.exception_handler(RequestValidationError)
async def validation_exception(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = [
        {"message": e["msg"], "path": ".".join(str(p) for p in e["loc"])} for e in exc.errors()
    ]
    return erro(422, errors)


@app.exception_handler(Exception)
async def unexpected_exception(request: Request, exc: Exception) -> JSONResponse:
    # A mensagem fica só no log, para não vazar detalhe interno na resposta
    log.exception("Erro não tratado em %s", request.url.path)
    return erro(500, [{"message": "Erro interno do servidor"}])


Proximo = Callable[[Request], Awaitable[Response]]


# Mesma regra de CORS do Auth e do simila-imagens: devolve a origem exata quando está na lista
# (com credenciais), recusa as outras com 403 e deixa passar quem não manda Origin
@app.middleware("http")
async def cors(request: Request, call_next: Proximo) -> Response:
    origin = request.headers.get("origin")
    if origin is None:
        return await call_next(request)
    if origin not in request.app.state.origens:
        return erro(403, [{"message": f"CORS: origem não permitida: {origin}"}])

    cabecalhos = {"Access-Control-Allow-Origin": origin, "Access-Control-Allow-Credentials": "true"}
    if request.method == "OPTIONS" and "access-control-request-method" in request.headers:
        return Response(
            status_code=204,
            headers={
                **cabecalhos,
                "Access-Control-Allow-Methods": "GET,POST,PUT,PATCH,DELETE",
                "Access-Control-Allow-Headers": "Content-Type,Authorization",
            },
        )
    res = await call_next(request)
    res.headers.update(cabecalhos)
    res.headers.add_vary_header("Origin")
    return res


app.add_middleware(GZipMiddleware)


async def _verificar(ping: Callable[[], Awaitable[bool]]) -> bool:
    try:
        return await asyncio.wait_for(ping(), timeout=3)
    except Exception:
        return False


@app.get("/api/similar/healthz", tags=["health"])
async def healthz(request: Request) -> JSONResponse:
    banco, redis = await asyncio.gather(
        _verificar(ping_banco), _verificar(lambda: ping_redis(request.app.state.redis))
    )
    rabbit = not request.app.state.amqp.is_closed
    # Redis é cache: fora do ar deixa o serviço mais lento, não indisponível
    saudavel = banco and rabbit
    estado = {True: "ok", False: "falhou"}
    return JSONResponse(
        {
            "status": "ok" if saudavel else "indisponivel",
            "banco": estado[banco],
            "rabbitmq": estado[rabbit],
            "redis": estado[redis],
        },
        status_code=200 if saudavel else 503,
    )


app.include_router(similares.router)
app.include_router(clusters.router)
app.include_router(admin.router)
