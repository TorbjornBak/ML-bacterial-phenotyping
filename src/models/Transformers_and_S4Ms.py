import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 15000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)            # [L, D]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [L,1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd
        self.register_buffer("pe", pe)                # not a parameter

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        # x: [B, T, D]
        T = x.size(1)
        x = x + self.pe[:T].unsqueeze(0)              # [1,T,D] + [B,T,D]
        return self.dropout(x)

# ----- Transformer model: embedding -> PE -> TransformerEncoder -> masked pool -> classifier -----
class TransformerKmerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 128,
        nhead: int = 8,
        ff_dim: int = 512,
        num_layers: int = 4,
        num_classes: int = 2,
        pad_id: int = 0,
        dropout: float = 0.1,
        use_mask: bool = True,
    ):
        super().__init__()
        self.use_mask = use_mask
        self.pad_id = pad_id

        self.head_dropout = nn.Dropout(dropout)
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.pos = PositionalEncoding(emb_dim, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,  # inputs as [B, T, D]
            norm_first=True,   # LayerNorm before attention (more stable)
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor = None, mask: torch.Tensor | None = None):
        # token_ids: [B, T] Long; mask: [B, T] Bool (True=valid) or 0/1
        x = self.emb(token_ids)                 # [B, T, D]
        x = self.pos(x)                         # [B, T, D]

        # Transformer expects src_key_padding_mask with True for PAD positions
        # MPS doesn't support nested tensors; avoid passing mask for long sequences
        key_padding_mask = None
        if self.pad_id is not None:
            # Derive from pad_id if mask not provided
            key_padding_mask = (token_ids == self.pad_id)  # [B, T]
            
            # For MPS compatibility: only use mask if not all-False
            if key_padding_mask.any():
                # Convert bool mask to float for better MPS support in some versions
                key_padding_mask = key_padding_mask.float()
            else:
                key_padding_mask = None

        # Use masked attention with explicit attention mask instead of padding mask if issues persist
        out = self.enc(x, src_key_padding_mask=key_padding_mask)  # [B, T, D]

        # Masked mean pooling over time
        if mask is not None:
            w = (mask > 0).float().unsqueeze(-1)   # [B, T, 1]
            denom = w.sum(dim=1).clamp_min(1.0)    # [B, 1]
            feat = (out * w).sum(dim=1) / denom    # [B, D]
        else:
            feat = out.mean(dim=1)                 # [B, D]

        logits = self.fc(self.head_dropout(feat))  # [B, num_classes]
        return logits
