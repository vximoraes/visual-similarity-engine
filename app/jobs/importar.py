"""
Importa uma pasta de imagens direto no banco, sem fila e sem storage: serve para carregar o
conjunto de avaliação (`assets/<categoria>/*.jpg`) sem sujar o bucket compartilhado do grupo.

Cada imagem vira a mesma thumbnail 224x224 que o simila-imagens gera, o embedding sai do CLIP em
lote e é gravado com um dono fixo. O `<pasta>/indice.json` guarda `id -> arquivo, categoria`; rodar
de novo reaproveita os ids (upsert, sem duplicar) e remove do banco o que saiu da pasta ou deixou de
abrir como imagem.

Rodar: `make importar PASTA=assets` (ou `uv run python -m app.jobs.importar <pasta>`)
"""

import argparse
import asyncio
import io
import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageCms, ImageOps, UnidentifiedImageError
from redis.asyncio import Redis
from sqlalchemy import delete

from app import cache, clip_model, repository
from app.config import config
from app.db import Sessao
from app.models import EmbeddingImagem
from app.worker import Centroides

log = logging.getLogger(__name__)

# Dono das imagens importadas: não é usuário do Auth, só marca a origem
DONO_IMPORTACAO = uuid.UUID("00000000-0000-7000-8000-00000000a55e")
LADO_THUMBNAIL = 224
QUALIDADE_THUMBNAIL = 85
EXTENSOES = {".jpg", ".jpeg", ".png", ".webp"}
NOME_INDICE = "indice.json"

type Indice = dict[str, dict[str, str]]


@dataclass(frozen=True)
class Entrada:
    imagem_id: uuid.UUID
    arquivo: str
    categoria: str


def para_srgb(imagem: Image.Image) -> Image.Image:
    """Converte pelo perfil ICC embutido (Adobe RGB, Display P3, CMYK), como o sharp faz"""
    perfil = imagem.info.get("icc_profile")
    if not perfil:
        return imagem.convert("RGB")
    try:
        origem = ImageCms.ImageCmsProfile(io.BytesIO(perfil))
        convertida = ImageCms.profileToProfile(
            imagem, origem, ImageCms.createProfile("sRGB"), outputMode="RGB"
        )
    except ImageCms.PyCMSError, OSError:
        # Perfil inválido ou incompatível com o modo da imagem: ignora o perfil, como o sharp
        return imagem.convert("RGB")
    return convertida if convertida is not None else imagem.convert("RGB")


def gerar_thumbnail(conteudo: bytes) -> Image.Image:
    """
    Mesmo pipeline do simila-imagens (`src/lib/processamento.ts`, sharp): orientação do EXIF,
    transparência sobre branco, sRGB pelo perfil ICC, resize cobrindo 224x224 com corte central
    (lanczos) e JPEG qualidade 85. O CLIP recebe o que receberia de uma imagem publicada pelo app.
    """
    imagem = ImageOps.exif_transpose(Image.open(io.BytesIO(conteudo)))
    perfil = imagem.info.get("icc_profile")
    if imagem.mode in ("RGBA", "LA", "PA", "P"):
        imagem = imagem.convert("RGBA")
        fundo = Image.new("RGBA", imagem.size, "white")
        imagem = Image.alpha_composite(fundo, imagem).convert("RGB")
        if perfil:
            imagem.info["icc_profile"] = perfil
    imagem = ImageOps.fit(
        para_srgb(imagem),
        (LADO_THUMBNAIL, LADO_THUMBNAIL),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )
    buffer = io.BytesIO()
    imagem.save(buffer, format="JPEG", quality=QUALIDADE_THUMBNAIL)
    thumbnail = Image.open(buffer)
    thumbnail.load()
    return thumbnail


def listar_imagens(pasta: Path) -> list[tuple[str, str]]:
    """(arquivo relativo à pasta, categoria = nome da subpasta), em ordem"""
    return [
        (caminho.relative_to(pasta).as_posix(), caminho.parent.name)
        for caminho in sorted(pasta.glob("*/*"))
        if caminho.suffix.lower() in EXTENSOES and caminho.is_file()
    ]


def ler_indice(pasta: Path) -> Indice:
    caminho = pasta / NOME_INDICE
    if not caminho.exists():
        return {}
    dados: Indice = json.loads(caminho.read_text(encoding="utf-8"))
    return dados


def gravar_indice(pasta: Path, indice: Indice) -> None:
    """Grava num arquivo temporário e renomeia: uma interrupção não deixa o índice pela metade"""
    temporario = pasta / f"{NOME_INDICE}.tmp"
    temporario.write_text(
        json.dumps(indice, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporario.replace(pasta / NOME_INDICE)


def planejar(
    imagens: list[tuple[str, str]], indice: Indice
) -> tuple[list[Entrada], list[uuid.UUID]]:
    """Reaproveita o id de quem já está no índice; devolve (entradas, ids que saíram da pasta)"""
    id_por_arquivo = {dados["arquivo"]: uuid.UUID(i) for i, dados in indice.items()}
    entradas = [
        Entrada(id_por_arquivo.get(arquivo) or uuid.uuid7(), arquivo, categoria)
        for arquivo, categoria in imagens
    ]
    presentes = {arquivo for arquivo, _ in imagens}
    removidos = [i for arquivo, i in id_por_arquivo.items() if arquivo not in presentes]
    return entradas, removidos


@dataclass
class Lote:
    validas: list[Entrada]
    thumbnails: list[Image.Image]
    invalidas: list[Entrada]


def preparar_lote(pasta: Path, entradas: list[Entrada]) -> Lote:
    lote = Lote([], [], [])
    for entrada in entradas:
        try:
            lote.thumbnails.append(gerar_thumbnail((pasta / entrada.arquivo).read_bytes()))
            lote.validas.append(entrada)
        except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as erro:
            log.warning("Ignorada %s: %s", entrada.arquivo, erro)
            lote.invalidas.append(entrada)
    return lote


async def remover(redis: Redis, pasta: Path, indice: Indice, ids: list[uuid.UUID]) -> None:
    """Tira do banco e do índice (sem tombstone: não é exclusão de post, só saiu do conjunto)"""
    async with Sessao() as sessao, sessao.begin():
        await sessao.execute(delete(EmbeddingImagem).where(EmbeddingImagem.imagem_id.in_(ids)))
    await cache.invalidar_similares(redis, ids)
    for imagem_id in ids:
        indice.pop(str(imagem_id), None)
    gravar_indice(pasta, indice)


async def executar(pasta: Path, dono: uuid.UUID = DONO_IMPORTACAO) -> Indice:
    imagens = listar_imagens(pasta)
    if not imagens:
        raise SystemExit(f"Nenhuma imagem em {pasta}/<categoria>/")
    indice = ler_indice(pasta)
    entradas, removidos = planejar(imagens, indice)

    modelo = await asyncio.to_thread(clip_model.carregar)
    centroides = Centroides()
    await centroides.atualizar()
    redis = cache.criar_redis()
    try:
        if removidos:
            await remover(redis, pasta, indice, removidos)
            log.info("%d imagens saíram da pasta e foram removidas do banco", len(removidos))

        tamanho = config().lote_max
        for inicio in range(0, len(entradas), tamanho):
            lote = await asyncio.to_thread(
                preparar_lote, pasta, entradas[inicio : inicio + tamanho]
            )
            # Já importada antes e agora não abre: sai, para o índice não apontar para lixo
            ja_importadas = [e.imagem_id for e in lote.invalidas if str(e.imagem_id) in indice]
            if ja_importadas:
                await remover(redis, pasta, indice, ja_importadas)
            validas = lote.validas
            if not validas:
                continue
            vetores = await asyncio.to_thread(modelo.codificar_imagens, lote.thumbnails)
            classes = centroides.classificar(vetores)
            itens = [
                repository.NovoEmbedding(e.imagem_id, dono, vetores[i], *classes[i])
                for i, e in enumerate(validas)
            ]
            async with Sessao() as sessao, sessao.begin():
                await repository.salvar_embeddings(sessao, itens, modelo.nome)
            await cache.invalidar_similares(redis, [e.imagem_id for e in validas])

            # Índice gravado a cada lote: se parar no meio, rodar de novo não duplica imagens
            for entrada in validas:
                indice[str(entrada.imagem_id)] = {
                    "arquivo": entrada.arquivo,
                    "categoria": entrada.categoria,
                }
            gravar_indice(pasta, indice)
            log.info("%d de %d imagens importadas", inicio + len(validas), len(entradas))
    finally:
        await redis.aclose()

    categorias = len({dados["categoria"] for dados in indice.values()})
    log.info("Importação concluída: %d imagens em %d categorias", len(indice), categorias)
    return indice


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Importa <pasta>/<categoria>/*.jpg no banco")
    parser.add_argument("pasta", type=Path)
    asyncio.run(executar(parser.parse_args().pasta))
