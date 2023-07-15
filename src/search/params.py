from pydantic import BaseModel


class Hyperparams(BaseModel):
    vocab_size: int
