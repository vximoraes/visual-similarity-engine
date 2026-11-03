"""
Convenção de mensageria do Simila (a mesma do Auth e do simila-imagens):
- exchange `simila` (topic, durável), onde todos publicam com a routing key do evento;
- uma fila durável por evento consumido: `<servico>.<routing-key>`;
- dead-letter exchange `simila.dlx` (direct) e uma DLQ por fila: `<fila>.dlq`.

O prefixo das filas é `vse` (e não `similar`) para não disputar mensagens com o serviço
`similaridade` do professor quando os dois estiverem no mesmo broker: com filas diferentes, cada
um recebe sua cópia do evento.
"""

from aio_pika import ExchangeType
from aio_pika.abc import AbstractChannel, AbstractExchange, AbstractQueue

EXCHANGE = "simila"
DLX = "simila.dlx"
SERVICO = "vse"

# Fila interna (não é evento do grupo): pedidos de reclusterização feitos pela rota admin
FILA_CLUSTERIZAR = f"{SERVICO}.comandos.clusterizar"


def nome_da_fila(routing_key: str) -> str:
    return f"{SERVICO}.{routing_key}"


async def declarar_exchange(canal: AbstractChannel) -> AbstractExchange:
    await canal.declare_exchange(DLX, ExchangeType.DIRECT, durable=True)
    return await canal.declare_exchange(EXCHANGE, ExchangeType.TOPIC, durable=True)


async def declarar_fila(canal: AbstractChannel, nome: str) -> AbstractQueue:
    """Fila durável com DLQ. Argumentos sempre iguais, senão PRECONDITION_FAILED"""
    dlx = await canal.get_exchange(DLX)
    dlq = await canal.declare_queue(f"{nome}.dlq", durable=True)
    await dlq.bind(dlx, routing_key=nome)
    return await canal.declare_queue(
        nome,
        durable=True,
        arguments={"x-dead-letter-exchange": DLX, "x-dead-letter-routing-key": nome},
    )


async def declarar_fila_do_evento(
    canal: AbstractChannel, exchange: AbstractExchange, routing_key: str
) -> AbstractQueue:
    fila = await declarar_fila(canal, nome_da_fila(routing_key))
    await fila.bind(exchange, routing_key=routing_key)
    return fila
