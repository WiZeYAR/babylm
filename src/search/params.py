from pydantic import BaseModel


class Hyperparams(BaseModel):
    vocab_size: int
    max_position_embedding: int
    num_attention_heads: int
    num_hidden_layers: int