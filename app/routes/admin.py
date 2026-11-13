from typing import Annotated

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from app.auth import Usuario, exigir_admin
from app.rabbit import pedir_clusterizacao as publicar_pedido
from app.routes import Resposta

router = APIRouter(prefix="/api/similar/admin", tags=["admin"])


class PedidoAceito(BaseModel):
    status: str


@router.post("/clusterizacoes", status_code=202, summary="Recalcula as categorias (k-means)")
async def pedir_clusterizacao(
    request: Request, usuario: Annotated[Usuario, Depends(exigir_admin)]
) -> Resposta[PedidoAceito]:
    """
    O k-means roda no worker (CPU pesada fora da API). Pedidos repetidos enquanto um roda são
    descartados pelo worker, que só reprocessa se houver imagens novas desde a última execução.
    """
    await publicar_pedido(request.app.state.canal.default_exchange, usuario.id)
    return Resposta(data=PedidoAceito(status="agendada"))
