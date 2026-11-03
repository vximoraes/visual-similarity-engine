import asyncio

from alembic import context
from sqlalchemy.engine import Connection

from app.db import criar_engine
from app.models import Base

target_metadata = Base.metadata


def rodar(conexao: Connection) -> None:
    context.configure(connection=conexao, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def online() -> None:
    engine = criar_engine()
    async with engine.connect() as conexao:
        await conexao.run_sync(rodar)
    await engine.dispose()


asyncio.run(online())
