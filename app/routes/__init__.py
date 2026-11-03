from pydantic import BaseModel


class Resposta[T](BaseModel):
    """Resposta de sucesso no padrão das APIs do Simila: { data, errors: null }"""

    data: T
    errors: None = None
