import torch
import torch.nn as nn

from .tokenizer import ByteTokenizer


class ByteEmbedding(nn.Module):
    """把 ByteTokenizer 輸出的 token id 序列查表轉成向量。

    輸入 shape (batch, seq_len) 的 LongTensor -> 輸出 (batch, seq_len, embed_dim)。
    padding_idx 對齊 ByteTokenizer.PAD_ID，補齊位置固定為 0 向量且不參與梯度更新。
    """

    def __init__(
        self,
        embed_dim: int = 128,
        dropout: float = 0.1,
        vocab_size: int = ByteTokenizer.VOCAB_SIZE,
        padding_idx: int = ByteTokenizer.PAD_ID,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)

    @property
    def embed_dim(self) -> int:
        return self.embedding.embedding_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.embedding(ids))
