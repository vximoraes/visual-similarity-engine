from functools import lru_cache
from urllib.parse import urlsplit

from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """Variáveis de ambiente (validadas na primeira leitura). Padrões em .env.example"""

    postgres_url: str
    redis_url: str = "redis://127.0.0.1:6379/0"
    amqp_url: str

    auth_jwks_url: str
    auth_issuer: str
    auth_jwks_cooldown: int = 30
    cors_origins: str = ""

    # Base pública do bucket de mídia do simila-imagens, usada quando o evento não traz a URL
    media_base_url: str

    clip_modelo: str = "ViT-B/32"
    torch_threads: int = 0

    # Worker: lote de inferência (até N mensagens ou T ms, o que vier primeiro)
    lote_max: int = 16
    lote_espera_ms: int = 200

    # Busca: ef_search do HNSW (maior = mais preciso e mais lento) e TTL do cache
    hnsw_ef_search: int = 64
    cache_ttl_similares: int = 300
    cache_ttl_texto: int = 86_400

    # Clusterização: faixa de k testada pela silhueta
    k_min: int = 2
    k_max: int = 30

    def origens_permitidas(self) -> list[str]:
        """A origem do próprio site (o "iss" do JWT é a URL pública) e as extras de CORS_ORIGINS"""
        site = urlsplit(self.auth_issuer)
        extras = [o.strip() for o in self.cors_origins.split(",") if o.strip()]
        return [f"{site.scheme}://{site.netloc}", *extras]


@lru_cache
def config() -> Config:
    return Config()
