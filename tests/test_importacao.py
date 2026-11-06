"""
Importação direta de uma pasta (sem fila e sem storage) e avaliação, com o CLIP de verdade e
imagens geradas na hora.
"""

import io
import json
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path

import numpy as np
import pytest
from httpx import AsyncClient
from PIL import Image, ImageCms
from sqlalchemy import delete, func, select

from app.db import Sessao
from app.jobs import avaliar, importar
from app.models import EmbeddingImagem
from tests.conftest import imagem_jpeg

pytestmark = pytest.mark.clip

VERMELHO, AZUL = (220, 20, 30), (20, 40, 220)


def salvar(caminho: Path, conteudo: bytes) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_bytes(conteudo)


def cor_media(imagem: Image.Image, caixa: tuple[int, int, int, int]) -> np.ndarray:
    return np.asarray(imagem.crop(caixa), dtype=np.float32).reshape(-1, 3).mean(axis=0)


@pytest.fixture
async def dono() -> AsyncGenerator[uuid.UUID]:
    """Dono próprio do teste; apaga o que importou para não afetar os outros testes"""
    dono = uuid.uuid4()
    yield dono
    async with Sessao() as sessao, sessao.begin():
        await sessao.execute(delete(EmbeddingImagem).where(EmbeddingImagem.dono_id == dono))


async def contar_do_dono(dono: uuid.UUID) -> int:
    async with Sessao() as sessao:
        total = await sessao.scalar(select(func.count()).where(EmbeddingImagem.dono_id == dono))
    return int(total or 0)


def test_thumbnail_corta_no_centro_sem_distorcer() -> None:
    # 400x300: o resize para 224 de altura deixa ~37 px de sobra de cada lado, que são cortados
    original = np.full((300, 400, 3), VERMELHO, dtype=np.uint8)
    original[:, :30] = original[:, -30:] = (0, 255, 0)
    buffer = io.BytesIO()
    Image.fromarray(original).save(buffer, format="PNG")

    thumbnail = importar.gerar_thumbnail(buffer.getvalue())

    assert thumbnail.size == (224, 224) and thumbnail.mode == "RGB"
    assert thumbnail.format == "JPEG"
    assert cor_media(thumbnail, (0, 0, 4, 224))[1] < 60


def test_thumbnail_aplica_orientacao_do_exif_e_fundo_branco() -> None:
    # Gravada com o vermelho em cima e orientação 6 (girar 90° horário): exibida, o vermelho fica à
    # direita
    pixels = np.zeros((224, 224, 3), dtype=np.uint8)
    pixels[:112], pixels[112:] = VERMELHO, AZUL
    exif = Image.Exif()
    exif[0x0112] = 6
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="JPEG", exif=exif.tobytes())

    girada = importar.gerar_thumbnail(buffer.getvalue())

    assert cor_media(girada, (200, 100, 224, 124))[0] > 180
    assert cor_media(girada, (0, 100, 24, 124))[2] > 180

    transparente = Image.new("RGBA", (300, 300), (0, 0, 0, 0))
    buffer = io.BytesIO()
    transparente.save(buffer, format="PNG")
    assert cor_media(importar.gerar_thumbnail(buffer.getvalue()), (0, 0, 224, 224)).min() > 250


async def test_importa_pasta_e_semelhantes_devolvem_vizinhos(
    cliente: AsyncClient, tmp_path: Path, dono: uuid.UUID
) -> None:
    salvar(tmp_path / "vermelhos" / "a.jpg", imagem_jpeg(VERMELHO, ruido=10, semente=1))
    salvar(tmp_path / "vermelhos" / "b.jpg", imagem_jpeg((200, 30, 40), ruido=10, semente=2))
    salvar(tmp_path / "azuis" / "c.jpg", imagem_jpeg(AZUL, ruido=10, semente=3))
    salvar(tmp_path / "azuis" / "nao-e-imagem.jpg", b"texto qualquer")
    (tmp_path / "solta-na-raiz.jpg").write_bytes(imagem_jpeg(AZUL))

    indice = await importar.executar(tmp_path, dono)

    assert sorted((d["categoria"], d["arquivo"]) for d in indice.values()) == [
        ("azuis", "azuis/c.jpg"),
        ("vermelhos", "vermelhos/a.jpg"),
        ("vermelhos", "vermelhos/b.jpg"),
    ]
    assert json.loads((tmp_path / "indice.json").read_text(encoding="utf-8")) == indice
    assert await contar_do_dono(dono) == 3

    ids = {d["arquivo"]: uuid.UUID(i) for i, d in indice.items()}
    res = await cliente.get(f"/api/similar/images/{ids['vermelhos/a.jpg']}/similares")
    assert res.status_code == 200
    vizinhos = [uuid.UUID(i["imageId"]) for i in res.json()["data"]]
    assert vizinhos.index(ids["vermelhos/b.jpg"]) < vizinhos.index(ids["azuis/c.jpg"])

    # Rodar de novo reaproveita os ids; arquivo que saiu da pasta sai do banco e do índice
    (tmp_path / "azuis" / "c.jpg").unlink()
    de_novo = await importar.executar(tmp_path, dono)

    assert set(de_novo) == {str(ids["vermelhos/a.jpg"]), str(ids["vermelhos/b.jpg"])}
    assert await contar_do_dono(dono) == 2


async def test_avaliacao_de_grupos_separados(tmp_path: Path, dono: uuid.UUID) -> None:
    """11 imagens por categoria: com k=10 a precisão máxima é 100%"""
    for semente in range(11):
        salvar(tmp_path / "gatos" / f"{semente}.jpg", imagem_jpeg(VERMELHO, 25, semente))
        salvar(tmp_path / "cachorros" / f"{semente}.jpg", imagem_jpeg(AZUL, 25, semente))
    await importar.executar(tmp_path, dono)

    resultado, texto = await avaliar.executar(tmp_path)

    assert resultado.geral == 1.0
    assert set(resultado.por_categoria) == {"gatos", "cachorros"}
    assert texto is not None
    assert sorted(texto.consultas) == ["cachorro", "cat", "dog", "gato"]
    relatorio = (tmp_path / "relatorio.html").read_text(encoding="utf-8")
    assert 'src="gatos/0.jpg"' in relatorio


def test_thumbnail_de_cmyk_sai_em_rgb() -> None:
    buffer = io.BytesIO()
    Image.new("CMYK", (300, 300), (0, 255, 255, 0)).save(buffer, format="JPEG")

    thumbnail = importar.gerar_thumbnail(buffer.getvalue())

    assert thumbnail.mode == "RGB"
    assert cor_media(thumbnail, (0, 0, 224, 224))[0] > 180


async def test_imagem_que_deixou_de_abrir_sai_do_banco_e_do_indice(
    tmp_path: Path, dono: uuid.UUID
) -> None:
    salvar(tmp_path / "vermelhos" / "a.jpg", imagem_jpeg(VERMELHO))
    salvar(tmp_path / "vermelhos" / "b.jpg", imagem_jpeg(VERMELHO, ruido=10))
    primeiro = await importar.executar(tmp_path, dono)

    (tmp_path / "vermelhos" / "b.jpg").write_bytes(b"corrompido")
    segundo = await importar.executar(tmp_path, dono)

    assert [d["arquivo"] for d in segundo.values()] == ["vermelhos/a.jpg"]
    assert set(segundo) < set(primeiro)
    assert await contar_do_dono(dono) == 1


async def test_avaliacao_sem_consulta_de_texto_mede_so_os_semelhantes(
    tmp_path: Path, dono: uuid.UUID
) -> None:
    for semente in range(11):
        salvar(tmp_path / "vermelhos" / f"{semente}.jpg", imagem_jpeg(VERMELHO, 25, semente))
        salvar(tmp_path / "azuis" / f"{semente}.jpg", imagem_jpeg(AZUL, 25, semente))
    await importar.executar(tmp_path, dono)

    resultado, texto = await avaliar.executar(tmp_path)

    assert texto is None
    assert resultado.geral == 1.0
    assert "Busca por texto" not in (tmp_path / "relatorio.html").read_text(encoding="utf-8")


async def test_avaliacao_com_poucas_imagens_explica_o_motivo(
    tmp_path: Path, dono: uuid.UUID
) -> None:
    salvar(tmp_path / "gatos" / "a.jpg", imagem_jpeg(VERMELHO))
    salvar(tmp_path / "gatos" / "b.jpg", imagem_jpeg(AZUL))
    await importar.executar(tmp_path, dono)

    with pytest.raises(SystemExit, match="não bastam"):
        await avaliar.executar(tmp_path)


def test_thumbnail_converte_pelo_perfil_icc_embutido() -> None:
    perfil = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    buffer = io.BytesIO()
    Image.new("RGB", (300, 300), VERMELHO).save(buffer, format="JPEG", icc_profile=perfil)

    thumbnail = importar.gerar_thumbnail(buffer.getvalue())

    assert thumbnail.mode == "RGB"
    assert np.allclose(cor_media(thumbnail, (0, 0, 224, 224)), VERMELHO, atol=6)


def test_thumbnail_ignora_perfil_icc_invalido() -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (300, 300), AZUL).save(buffer, format="JPEG", icc_profile=b"lixo" * 40)

    thumbnail = importar.gerar_thumbnail(buffer.getvalue())

    assert np.allclose(cor_media(thumbnail, (0, 0, 224, 224)), AZUL, atol=6)
