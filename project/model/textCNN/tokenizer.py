import torch


class ByteTokenizer:
    """把 raw bytes（flow payload）轉成固定長度的 byte-level token id 序列。

    每個位元組值 (0-255) 直接當作 token id，不做任何字典查找；額外保留一個
    PAD id 用來補齊長度，embedding 層可搭配 padding_idx=PAD_ID 讓補齊部分
    不參與梯度更新。
    """

    PAD_ID = 256
    VOCAB_SIZE = 257

    def __init__(self, max_len: int = 1500):
        if max_len <= 0:
            raise ValueError("max_len 必須為正整數")
        self.max_len = max_len

    def encode(self, data: bytes) -> list[int]:
        """單一 flow payload -> 長度固定為 max_len 的 token id list（截斷或補 PAD）。"""
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("data must be bytes")
        ids = list(data[: self.max_len])
        ids += [self.PAD_ID] * (self.max_len - len(ids))
        return ids

    def encode_batch(self, batch) -> torch.Tensor:
        """多個 flow payload -> shape (batch, max_len) 的 LongTensor。"""
        return torch.tensor([self.encode(data) for data in batch], dtype=torch.long)

    def __call__(self, data):
        if isinstance(data, (bytes, bytearray)):
            return torch.tensor(self.encode(data), dtype=torch.long)
        return self.encode_batch(data)

    def decode(self, ids) -> bytes:
        """token id 序列還原成 bytes，去掉 PAD（僅供除錯用）。"""
        return bytes(int(i) for i in ids if int(i) != self.PAD_ID)
