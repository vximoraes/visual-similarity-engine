"""
Integração com dependências reais (Postgres + pgvector, Redis, RabbitMQ), sem mock. O JWKS do
Auth e as thumbnails do storage são servidos por HTTP de verdade em portas locais.
"""

import io
import json
import os
import threading
import time
import uuid
from collections.abc import AsyncGenerator, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import jwt
import numpy as np
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from PIL import Image

PORTA_JWKS = 4598
PORTA_MIDIA = 4597
ISSUER = "http://simila.teste"

os.environ.update(
    {
        "AUTH_JWKS_URL": f"http://127.0.0.1:{PORTA_JWKS}/api/auth/jwks",
        "AUTH_ISSUER": ISSUER,
        "MEDIA_BASE_URL": f"http://127.0.0.1:{PORTA_MIDIA}",
        "LOTE_ESPERA_MS": "50",
    }
)

from httpx import ASGITransport, AsyncClient  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app.db import Sessao  # noqa: E402
from app.main import app  # noqa: E402

CHAVE = Ed25519PrivateKey.generate()
MIDIA: dict[str, bytes] = {}


def _servir(porta: int, responder: Any) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            status, corpo, tipo = responder(self.path)
            self.send_response(status)
            self.send_header("Content-Type", tipo)
            self.end_headers()
            self.wfile.write(corpo)

        def log_message(self, *args: Any) -> None:
            pass

    # Backlog padrão (5) recusa conexões quando o worker baixa o lote inteiro em paralelo
    ThreadingHTTPServer.request_queue_size = 64
    servidor = ThreadingHTTPServer(("127.0.0.1", porta), Handler)
    threading.Thread(target=servidor.serve_forever, daemon=True).start()
    return servidor


@pytest.fixture(scope="session", autouse=True)
def servidores_http() -> Iterator[None]:
    publica = json.loads(jwt.algorithms.OKPAlgorithm.to_jwk(CHAVE.public_key()))
    jwks = json.dumps({"keys": [{**publica, "kid": "teste", "alg": "EdDSA", "use": "sig"}]})
    s1 = _servir(PORTA_JWKS, lambda _p: (200, jwks.encode(), "application/json"))
    s2 = _servir(
        PORTA_MIDIA,
        lambda p: (200, MIDIA[p], "image/jpeg") if p in MIDIA else (404, b"", "text/plain"),
    )
    yield
    s1.shutdown()
    s2.shutdown()


def assinar(sub: str | None = None, role: str = "user", aud: str = "simila") -> str:
    agora = int(time.time())
    payload: dict[str, str | int] = {"sub": sub or str(uuid.uuid4()), "name": "Teste"}
    payload |= {"role": role, "iss": ISSUER, "aud": aud, "iat": agora, "exp": agora + 900}
    return jwt.encode(payload, CHAVE, algorithm="EdDSA", headers={"kid": "teste"})


def imagem_jpeg(cor: tuple[int, int, int], ruido: int = 0, semente: int = 0) -> bytes:
    """Thumbnail 224x224 de cor sólida (com ruído opcional): cores parecidas = vetores parecidos"""
    gerador = np.random.default_rng(semente)
    base = np.full((224, 224, 3), cor, dtype=np.int16)
    if ruido:
        base += gerador.integers(-ruido, ruido + 1, size=base.shape, dtype=np.int16)
    buffer = io.BytesIO()
    Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).save(buffer, format="JPEG")
    return buffer.getvalue()


def publicar_thumb(dono: uuid.UUID, imagem: uuid.UUID, conteudo: bytes) -> str:
    caminho = f"/{dono}/post/{imagem}_thumb.jpg"
    MIDIA[caminho] = conteudo
    return f"http://127.0.0.1:{PORTA_MIDIA}{caminho}"


@pytest.fixture(scope="session", autouse=True)
async def banco_limpo() -> None:
    async with Sessao() as sessao, sessao.begin():
        await sessao.execute(
            text("truncate embeddings_imagem, clusters, execucoes_clusterizacao, imagens_excluidas")
        )


@pytest.fixture(scope="session")
async def cliente() -> AsyncGenerator[AsyncClient]:
    """App real com lifespan (carrega o CLIP, conecta no Redis e no RabbitMQ)"""
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://teste") as c:
            yield c
