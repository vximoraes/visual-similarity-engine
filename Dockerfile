# ===============================================================
# Build: dependências com uv (torch CPU) e os pesos do CLIP baixados
# na imagem, para o container não depender da internet ao subir
# ===============================================================
FROM python:3.14-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12.9 /uv /bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never

# git: o CLIP vem do repositório da OpenAI (não está no PyPI)
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# O clip.load guarda os pesos em ~/.cache/clip: HOME=/app deixa isso dentro de /app
ENV HOME=/app
RUN .venv/bin/python -c "import clip; clip.load('ViT-B/32', device='cpu', jit=False)"

COPY app ./app
COPY alembic ./alembic
COPY alembic.ini ./

# ===============================================================
# Imagem final. A mesma imagem roda a API (padrão) e o worker:
#   docker run ... vse python -m app.worker
# ===============================================================
FROM python:3.14-slim AS runner
WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH" \
    HOME=/app \
    PYTHONUNBUFFERED=1 \
    # O Uvicorn passa a confiar no X-Forwarded-* vindo do gateway
    FORWARDED_ALLOW_IPS="*"

COPY --from=builder /app /app
# Valores padrão do env. O que vier do ambiente tem prioridade (load_dotenv sem override)
COPY .env.example ./.env

RUN useradd --system --uid 10001 app
USER app
EXPOSE 8000

HEALTHCHECK --interval=1m --timeout=5s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/similar/healthz')"

CMD ["fastapi", "run", "app/main.py", "--port", "8000"]
