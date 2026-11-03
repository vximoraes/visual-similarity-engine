import json
from datetime import UTC, datetime
from typing import Annotated

import aio_pika
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from app.auth import Usuario, exigir_admin
from app.rabbit import FILA_CLUSTERIZAR
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
    corpo = {"pedidoPor": usuario.id, "pedidoEm": datetime.now(UTC).isoformat()}
    await request.app.state.canal.default_exchange.publish(
        aio_pika.Message(
            json.dumps(corpo).encode(),
            content_type="application/json",
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        ),
        routing_key=FILA_CLUSTERIZAR,
    )
    return Resposta(data=PedidoAceito(status="agendada"))
