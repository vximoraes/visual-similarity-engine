"""
Validação do JWT do Auth pelo JWKS (mesmas regras do simila-auth/docs/validando-jwt.md):
EdDSA fixo, iss = AUTH_ISSUER, aud = "simila", exp e sub obrigatórios.
"""

from typing import Annotated, Literal

import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from app.config import config

bearer = HTTPBearer(auto_error=False)


class Usuario(BaseModel):
    id: str
    nome: str
    papel: Literal["user", "admin"]


def criar_jwks() -> jwt.PyJWKClient:
    # Um por processo: é ele que guarda o cache (5 min). Sem cache_keys: chave retirada do JWKS
    # deixaria de valer só no restart
    return jwt.PyJWKClient(config().auth_jwks_url, lifespan=300, timeout=5)


def _nao_autorizado(mensagem: str) -> HTTPException:
    return HTTPException(401, mensagem, headers={"WWW-Authenticate": "Bearer"})


def usuario_atual(
    request: Request,
    credenciais: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> Usuario:
    """Síncrona de propósito: a busca do JWKS é bloqueante e o FastAPI roda isso num thread pool"""
    if credenciais is None:
        raise _nao_autorizado("Token não informado")
    jwks: jwt.PyJWKClient = request.app.state.jwks
    try:
        chave = jwks.get_signing_key_from_jwt(credenciais.credentials)
    except jwt.PyJWKClientConnectionError as erro:
        # O Auth não respondeu: o token pode ser válido, então não é 401
        raise HTTPException(503, "Serviço de autenticação indisponível") from erro
    except jwt.PyJWTError as erro:
        raise _nao_autorizado("Token inválido ou expirado") from erro
    try:
        payload = jwt.decode(
            credenciais.credentials,
            chave,
            algorithms=["EdDSA"],
            audience="simila",
            issuer=config().auth_issuer,
            options={"require": ["exp", "sub", "iss", "aud"]},
        )
    except jwt.PyJWTError as erro:
        raise _nao_autorizado("Token inválido ou expirado") from erro
    papel = payload.get("role", "user")
    return Usuario(
        id=payload["sub"],
        nome=payload.get("name", ""),
        papel="admin" if papel == "admin" else "user",
    )


def exigir_admin(usuario: Annotated[Usuario, Depends(usuario_atual)]) -> Usuario:
    if usuario.papel != "admin":
        raise HTTPException(403, "Apenas administradores")
    return usuario
