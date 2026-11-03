import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, Text, func, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

DIMENSOES = 512


class Base(DeclarativeBase):
    pass


class ExecucaoClusterizacao(Base):
    """Cada execução do k-means. Só uma fica ativa: é a que classifica as imagens novas"""

    __tablename__ = "execucoes_clusterizacao"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, server_default=text("uuidv7()"))
    modelo: Mapped[str] = mapped_column(Text)
    algoritmo: Mapped[str] = mapped_column(Text)
    k: Mapped[int | None] = mapped_column(Integer)
    amostras: Mapped[int] = mapped_column(Integer)
    silhueta: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(Text)  # executando | pronta | falhou
    motivo_falha: Mapped[str | None] = mapped_column(Text)
    ativa: Mapped[bool] = mapped_column(Boolean, server_default=text("false"))
    criado_em: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    ativado_em: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Cluster(Base):
    __tablename__ = "clusters"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, server_default=text("uuidv7()"))
    execucao_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("execucoes_clusterizacao.id", ondelete="CASCADE")
    )
    indice: Mapped[int] = mapped_column(Integer)
    centroide: Mapped[list[float]] = mapped_column(Vector(DIMENSOES))
    rotulo: Mapped[str | None] = mapped_column(Text)
    tamanho: Mapped[int] = mapped_column(Integer)
    imagem_representativa_id: Mapped[uuid.UUID | None]


class EmbeddingImagem(Base):
    __tablename__ = "embeddings_imagem"

    imagem_id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    dono_id: Mapped[uuid.UUID]
    # Normalizado (norma L2 = 1): distância de cosseno e produto interno ficam equivalentes
    embedding: Mapped[list[float]] = mapped_column(Vector(DIMENSOES))
    modelo: Mapped[str] = mapped_column(Text)
    cluster_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("clusters.id", ondelete="SET NULL")
    )
    distancia_centroide: Mapped[float | None] = mapped_column(Float)
    criado_em: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    atualizado_em: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class ImagemExcluida(Base):
    """Tombstone: image.processed.post atrasado (depois do image.deleted.post) não recria o vetor"""

    __tablename__ = "imagens_excluidas"

    imagem_id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    excluida_em: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
