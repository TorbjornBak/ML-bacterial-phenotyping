import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_positions(length: int, dim: int, device=None) -> torch.Tensor:
    """Fixed sine/cosine position table of shape [length, dim], built for any length."""
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim)
    )
    table = torch.zeros(length, dim, device=device)
    table[:, 0::2] = torch.sin(position * div_term)
    table[:, 1::2] = torch.cos(position * div_term[: dim // 2])
    return table


class TransformerEncoderBlock(nn.Module):
    """Pre-norm self-attention + feed-forward block.

    Attention is computed by ``F.scaled_dot_product_attention`` with a boolean
    key mask, which lets PyTorch use its fused flash/memory-efficient kernels.
    Memory then grows linearly with sequence length instead of materialising a
    [B, heads, T, T] score matrix, which is what makes genome-length token
    sequences trainable on a laptop CPU.
    """

    def __init__(self, dim: int, nhead: int, ff_dim: int, dropout: float):
        super().__init__()
        if dim % nhead:
            raise ValueError(f"emb_dim ({dim}) must be divisible by nhead ({nhead})")
        self.nhead = nhead
        self.attention_norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.attention_out = nn.Linear(dim, dim)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor | None) -> torch.Tensor:
        # x: [B, T, D]; key_mask: [B, 1, 1, T] bool (True = attend) or None
        batch, length, dim = x.shape
        q, k, v = (
            self.qkv(self.attention_norm(x))
            .view(batch, length, 3, self.nhead, dim // self.nhead)
            .permute(2, 0, 3, 1, 4)  # 3 x [B, H, T, D/H]
        )
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask)
        attended = attended.transpose(1, 2).reshape(batch, length, dim)
        x = x + self.dropout(self.attention_out(attended))
        return x + self.dropout(self.ff(self.ff_norm(x)))


# ----- Transformer model: embedding -> PE -> encoder blocks -> masked mean pool -> classifier -----
class TransformerKmerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 64,
        nhead: int = 4,
        ff_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 2,
        pad_id: int = 0,
        dropout: float = 0.1,
        use_mask: bool = True,
        positional_encoding: bool = True,
    ):
        super().__init__()
        self.use_mask = use_mask
        self.pad_id = pad_id
        self.positional_encoding = positional_encoding

        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.emb_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            TransformerEncoderBlock(emb_dim, nhead, ff_dim, dropout) for _ in range(num_layers)
        )
        self.norm = nn.LayerNorm(emb_dim)
        self.head_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(emb_dim, num_classes)

    def _valid_positions(self, token_ids, lengths, mask) -> torch.Tensor | None:
        """[B, T] bool, True for real tokens; None when nothing is padded."""
        if not self.use_mask:
            return None
        if mask is not None:
            valid = mask > 0
        elif lengths is not None:
            positions = torch.arange(token_ids.size(1), device=token_ids.device)
            valid = positions.unsqueeze(0) < lengths.to(token_ids.device).unsqueeze(1)
        elif self.pad_id is not None:
            valid = token_ids != self.pad_id
        else:
            return None
        # A row with no valid key would make attention NaN; keep its first slot.
        valid = valid.clone()
        valid[:, 0] |= ~valid.any(dim=1)
        return None if bool(valid.all()) else valid

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor = None, mask: torch.Tensor | None = None):
        # token_ids: [B, T] Long; lengths: [B]; mask: [B, T] bool/0-1 (True = real token)
        valid = self._valid_positions(token_ids, lengths, mask)
        x = self.emb(token_ids)  # [B, T, D]
        if self.positional_encoding:
            x = x + sinusoidal_positions(x.size(1), x.size(2), device=x.device).unsqueeze(0)
        x = self.emb_dropout(x)

        key_mask = None if valid is None else valid[:, None, None, :]
        for block in self.blocks:
            x = block(x, key_mask)
        x = self.norm(x)

        # Masked mean pooling over time
        if valid is not None:
            w = valid.unsqueeze(-1).to(x.dtype)  # [B, T, 1]
            feat = (x * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)  # [B, D]
        else:
            feat = x.mean(dim=1)  # [B, D]

        return self.fc(self.head_dropout(feat))  # [B, num_classes]
