"""
Tradução da busca por texto para inglês antes do encoder do CLIP, que só foi treinado com legendas
em inglês: na base do professor, a precisão@10 em português sobe de 46,4% para 80,7%
(`docs/avaliacao.md`). Modelo MarianMT local, sem serviço externo, carregado uma vez por processo.

Desligável por `TRADUZIR_BUSCA=false`: consulta já em inglês às vezes estraga ("clothes" vira
"eyelashes"), e o grupo pode preferir o CLIP puro.
"""

import threading
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import MarianMTModel, MarianTokenizer

from app.clip_model import ModeloClip, Vetores
from app.config import config


@dataclass(frozen=True)
class Tradutor:
    nome: str
    modelo: Any
    tokenizador: Any
    # O tokenizador não é seguro entre threads; a tradução de uma consulta leva ~85 ms
    trava: threading.Lock = field(default_factory=threading.Lock, compare=False)

    def traduzir(self, textos: list[str]) -> list[str]:
        with self.trava, torch.inference_mode():
            lote = self.tokenizador(textos, return_tensors="pt", padding=True, truncation=True)
            # Limite: o max_length (512) do modelo; a consulta tem até 200 caracteres
            saida = self.modelo.generate(**lote)
            return [str(self.tokenizador.decode(t, skip_special_tokens=True)) for t in saida]


_tradutor: Tradutor | None = None
_trava = threading.Lock()


def carregar() -> Tradutor | None:
    """None com a tradução desligada. Os pesos (~300 MB) ficam no cache do Hugging Face"""
    global _tradutor
    if not config().traduzir_busca:
        return None
    with _trava:
        if _tradutor is None:
            nome = config().traducao_modelo
            # Any: os tipos do transformers não cobrem eval() e generate()
            modelo: Any = MarianMTModel.from_pretrained(nome)
            modelo.eval()
            _tradutor = Tradutor(nome, modelo, MarianTokenizer.from_pretrained(nome))
    return _tradutor


def codificar_consultas(clip: ModeloClip, tradutor: Tradutor | None, textos: list[str]) -> Vetores:
    """Vetor de cada busca por texto: traduz para inglês (se ligado) e codifica com o CLIP"""
    if tradutor is not None:
        textos = tradutor.traduzir(textos)
    return clip.codificar_textos(textos)
