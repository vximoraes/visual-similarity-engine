import threading
from dataclasses import dataclass
from typing import Any

import clip
import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

from app.config import config

Vetores = NDArray[np.float32]


@dataclass(frozen=True)
class ModeloClip:
    nome: str
    modelo: Any
    preprocessar: Any

    def codificar_imagens(self, imagens: list[Image.Image]) -> Vetores:
        """Um lote de imagens vira uma matriz (n, 512) normalizada (norma L2 = 1)"""
        lote = torch.stack([self.preprocessar(img.convert("RGB")) for img in imagens])
        with torch.inference_mode():
            vetores = self.modelo.encode_image(lote).float()
        return normalizar(vetores.numpy())

    def codificar_textos(self, textos: list[str]) -> Vetores:
        """Mesmo espaço vetorial das imagens: dá para comparar texto com imagem"""
        tokens = clip.tokenize(textos, truncate=True)
        with torch.inference_mode():
            vetores = self.modelo.encode_text(tokens).float()
        return normalizar(vetores.numpy())


def normalizar(vetores: NDArray[Any]) -> Vetores:
    normas = np.linalg.norm(vetores, axis=1, keepdims=True)
    resultado: Vetores = (vetores / np.clip(normas, 1e-12, None)).astype(np.float32)
    return resultado


_modelo: ModeloClip | None = None
_trava = threading.Lock()


def carregar() -> ModeloClip:
    """Carrega uma vez por processo (~340 MB em memória). Os pesos ficam em ~/.cache/clip"""
    global _modelo
    with _trava:
        if _modelo is None:
            if config().torch_threads > 0:
                torch.set_num_threads(config().torch_threads)
            modelo, preprocessar = clip.load(config().clip_modelo, device="cpu", jit=False)
            modelo.eval()
            _modelo = ModeloClip(config().clip_modelo, modelo, preprocessar)
    return _modelo
