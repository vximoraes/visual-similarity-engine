from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from app.config import config


def criar_engine() -> AsyncEngine:
    # postgresql+psycopg:// usa o psycopg 3 (async)
    url = config().postgres_url.replace("postgresql://", "postgresql+psycopg://", 1)
    return create_async_engine(url, pool_size=5, max_overflow=5, pool_pre_ping=True)


engine = criar_engine()
Sessao = async_sessionmaker(engine, expire_on_commit=False)


async def ping_banco() -> bool:
    async with engine.connect() as conexao:
        await conexao.execute(text("select 1 from embeddings_imagem limit 1"))
    return True
