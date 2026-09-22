import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    """Kim (2014) TextCNN：单一 embedding 来源，多种 kernel size 的 1D 卷积
    + max-over-time pooling，串接后接一层线性分类头。

    输入是已经过 embedding 层的张量 (batch, seq_len, embed_dim)，卷积用
    valid padding（不补零），seq_len 需要 >= 最大的 kernel_size。
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        kernel_sizes=(3, 4, 5),
        num_filters: int = 100,
        dropout: float = 0.5,
        max_norm: float | None = 3.0,
    ):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, num_filters, kernel_size=k) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)
        self.max_norm = max_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, embed_dim) -> logits (batch, num_classes)。"""
        x = x.transpose(1, 2)  # Conv1d 要 channel-first: (batch, embed_dim, seq_len)
        pooled = [F.relu(conv(x)).max(dim=2).values for conv in self.convs]
        feat = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(feat))

    @torch.no_grad()
    def apply_max_norm(self):
        """每次 optimizer.step() 后呼叫：把 fc 每个输出类别对应的权重向量
        限制在 L2 norm <= max_norm 以内（Kim 2014 原论文的正则化做法）。"""
        if self.max_norm is None:
            return
        norms = self.fc.weight.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
        scale = (norms.clamp(max=self.max_norm) / norms)
        self.fc.weight.mul_(scale)


class _ConvGroup(nn.Module):
    """MGNC-CNN 里单一 embedding group 的卷积部分：多种 kernel size 各自
    max-over-time pooling 后串接，组与组之间互不混合卷积。"""

    def __init__(self, embed_dim: int, kernel_sizes=(3, 4, 5), num_filters: int = 100):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, num_filters, kernel_size=k) for k in kernel_sizes]
        )

    @property
    def out_dim(self) -> int:
        return sum(conv.out_channels for conv in self.convs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        pooled = [F.relu(conv(x)).max(dim=2).values for conv in self.convs]
        return torch.cat(pooled, dim=1)


class MGNCCNN(nn.Module):
    """MGNC-CNN (Zhang, Roller & Wallace, 2016)。

    与单纯把多个 embedding 沿特征维度 concat 后丢进同一个 TextCNN（"multichannel"
    CNN）不同，MGNC-CNN 让每组 embedding 各自独立做卷积＋pooling（组间互不干扰），
    只在池化后的特征向量层面拼接；最后分类层的权重再依来源分组分别做 L2
    max-norm 限制，让品质较差的 embedding 来源可以被赋予更严格（更小）的 norm
    上限，等同于对不同来源做不同强度的正则化。

    使用方式：每组各自的 embedding（例如 ByteEmbedding 的输出，或不同来源的
    embedding）先算好，forward 时以 list 传入，顺序需与建构子的 embed_dims 一致。
    """

    def __init__(
        self,
        embed_dims: list[int],
        num_classes: int,
        kernel_sizes=(3, 4, 5),
        num_filters: int = 100,
        dropout: float = 0.5,
        max_norms: list[float] | None = None,
    ):
        super().__init__()
        if not embed_dims:
            raise ValueError("embed_dims 不可为空")

        self.groups = nn.ModuleList(
            [_ConvGroup(dim, kernel_sizes, num_filters) for dim in embed_dims]
        )
        self.group_sizes = [g.out_dim for g in self.groups]
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(sum(self.group_sizes), num_classes)

        if max_norms is None:
            max_norms = [3.0] * len(embed_dims)
        if len(max_norms) != len(embed_dims):
            raise ValueError("max_norms 长度必须与 embed_dims 一致")
        self.max_norms = list(max_norms)

        # 各组池化特征在拼接后向量里的 [start, end) 位置，apply_max_norm 时要用
        offsets = [0]
        for size in self.group_sizes:
            offsets.append(offsets[-1] + size)
        self._offsets = offsets

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        """xs: 长度须等于 group 数，各自形状 (batch, seq_len, embed_dim_i)。"""
        if len(xs) != len(self.groups):
            raise ValueError(f"预期 {len(self.groups)} 组输入，收到 {len(xs)} 组")
        feats = [group(x) for group, x in zip(self.groups, xs)]
        feat = torch.cat(feats, dim=1)
        return self.fc(self.dropout(feat))

    @torch.no_grad()
    def apply_max_norm(self):
        """每次 optimizer.step() 后呼叫：依组别对 fc.weight 的对应区块做
        L2 max-norm 裁剪（每组各自的 max_norms[i] 上限）。"""
        weight = self.fc.weight  # (num_classes, sum(group_sizes))
        for i, s in enumerate(self.max_norms):
            start, end = self._offsets[i], self._offsets[i + 1]
            block = weight[:, start:end]
            norms = block.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
            scale = (norms.clamp(max=s) / norms)
            block.mul_(scale)
