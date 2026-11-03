"""inicial: embeddings (pgvector + HNSW), execuções de clusterização, clusters e tombstones

Revision ID: 0001
Revises:
Create Date: 2026-11-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

UUIDV7 = sa.text("uuidv7()")
AGORA = sa.text("now()")


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "execucoes_clusterizacao",
        sa.Column("id", sa.Uuid(), primary_key=True, server_default=UUIDV7),
        sa.Column("modelo", sa.Text(), nullable=False),
        sa.Column("algoritmo", sa.Text(), nullable=False),
        sa.Column("k", sa.Integer()),
        sa.Column("amostras", sa.Integer(), nullable=False),
        sa.Column("silhueta", sa.Float()),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("motivo_falha", sa.Text()),
        sa.Column("ativa", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("criado_em", sa.DateTime(timezone=True), nullable=False, server_default=AGORA),
        sa.Column("ativado_em", sa.DateTime(timezone=True)),
        sa.CheckConstraint(
            "status in ('executando', 'pronta', 'falhou')", name="execucoes_status_valido"
        ),
    )
    # No máximo uma execução ativa
    op.create_index(
        "execucoes_uma_ativa",
        "execucoes_clusterizacao",
        ["ativa"],
        unique=True,
        postgresql_where=sa.text("ativa"),
    )

    op.create_table(
        "clusters",
        sa.Column("id", sa.Uuid(), primary_key=True, server_default=UUIDV7),
        sa.Column(
            "execucao_id",
            sa.Uuid(),
            sa.ForeignKey("execucoes_clusterizacao.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("indice", sa.Integer(), nullable=False),
        sa.Column("centroide", Vector(512), nullable=False),
        sa.Column("rotulo", sa.Text()),
        sa.Column("tamanho", sa.Integer(), nullable=False),
        sa.Column("imagem_representativa_id", sa.Uuid()),
        sa.UniqueConstraint("execucao_id", "indice"),
    )

    op.create_table(
        "embeddings_imagem",
        sa.Column("imagem_id", sa.Uuid(), primary_key=True),
        sa.Column("dono_id", sa.Uuid(), nullable=False),
        sa.Column("embedding", Vector(512), nullable=False),
        sa.Column("modelo", sa.Text(), nullable=False),
        sa.Column("cluster_id", sa.Uuid(), sa.ForeignKey("clusters.id", ondelete="SET NULL")),
        sa.Column("distancia_centroide", sa.Float()),
        sa.Column("criado_em", sa.DateTime(timezone=True), nullable=False, server_default=AGORA),
        sa.Column(
            "atualizado_em", sa.DateTime(timezone=True), nullable=False, server_default=AGORA
        ),
    )
    # Busca aproximada de vizinhos por cosseno. m/ef_construction: padrões recomendados do pgvector
    op.execute(
        "CREATE INDEX embeddings_imagem_hnsw ON embeddings_imagem "
        "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
    )
    op.create_index(
        "embeddings_imagem_cluster",
        "embeddings_imagem",
        ["cluster_id", "distancia_centroide", "imagem_id"],
    )

    op.create_table(
        "imagens_excluidas",
        sa.Column("imagem_id", sa.Uuid(), primary_key=True),
        sa.Column("excluida_em", sa.DateTime(timezone=True), nullable=False, server_default=AGORA),
    )


def downgrade() -> None:
    op.drop_table("imagens_excluidas")
    op.drop_table("embeddings_imagem")
    op.drop_table("clusters")
    op.drop_table("execucoes_clusterizacao")
