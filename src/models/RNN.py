import torch
from torch import nn
import torch.nn.functional as F

# ----- RNN model: embedding -> BiGRU -> temporal BatchNorm -> global pool -> classifier -----
class RNNKmerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim=16,
        rnn_hidden=128,
        num_layers=1,
        bidirectional=True,
        num_classes=2,
        pad_id=0,
        dropout=0.1
    ):
        super().__init__()
       
        self.pad_id = pad_id
        self.bidirectional = bidirectional
        self.head_dropout = nn.Dropout(dropout)

        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=rnn_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        
        feat_dim = rnn_hidden * (2 if bidirectional else 1)
        # BatchNorm over pooled features
        self.bn = nn.BatchNorm1d(feat_dim)

        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, token_ids: torch.Tensor):
        # token_ids: [B, T] Long
        x = self.emb(token_ids)  # [B, T, D]
        out, _ = self.gru(x.contiguous())  # [B, T, H*dir]
        
        feat = out.mean(dim=1)  # [B, H*dir]
        feat = self.bn(feat)    # BatchNorm on features
        
        logits = self.fc(self.head_dropout(feat))  # [B, num_classes]
        return logits


# ----- RNN model: embedding -> BiGRU -> temporal BatchNorm -> global pool -> classifier -----
class RNN_MLP_KmerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim=16,
        rnn_hidden=128,
        num_layers=1,
        bidirectional=True,
        num_classes=2,
        pad_id=0,
        dropout=0.1,
        emb_dropout=0.1,
        pooling="attn",
        norm="layer",
    ):
        super().__init__()
       
        self.pad_id = pad_id
        self.bidirectional = bidirectional

        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.pooling = pooling

        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=rnn_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        
        feat_dim = rnn_hidden * (2 if bidirectional else 1)
       

       # Optional attention pooling
        if pooling == "attn":
            self.attn = nn.Linear(feat_dim, 1)
        else:
            self.attn = None

        # Normalization
        if norm == "layer":
            self.norm = nn.LayerNorm(feat_dim)
        elif norm == "batch":
            self.norm = nn.BatchNorm1d(feat_dim)
        else:
            self.norm = None

        self.head_dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, rnn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_hidden, num_classes),
        )

    def _masked_mean(self, x, mask):
        # x: [B, T, C], mask: [B, T] (bool)
        mask = mask.unsqueeze(-1)  # [B, T, 1]
        x = x * mask  # zero out pads
        denom = mask.sum(dim=1).clamp_min(1)  # [B, 1]
        return x.sum(dim=1) / denom

    def _attn_pool(self, x, mask):
        # x: [B, T, C], mask: [B, T] (bool)
        scores = self.attn(x).squeeze(-1)  # [B, T]
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=1).unsqueeze(-1)  # [B, T, 1]
        return (x * weights).sum(dim=1)  # [B, C]

    def forward(self, token_ids: torch.Tensor):
        # token_ids: [B, T] Long
        mask = (token_ids != self.pad_id)  # [B, T] for padded batches


        x = self.emb(token_ids)  # [B, T, D]
        x = self.emb_dropout(x)
        out, hn = self.gru(x)      
        #out, _ = self.gru(x.contiguous())  # [B, T, H*dir]
        
        # Pooling
        if self.pooling == "mean":
            feat = self._masked_mean(out, mask)
        elif self.pooling == "last":
            if self.bidirectional:
                # last layer forward/backward states
                h_last = hn[-2:]             # [2, B, H]
                feat = torch.cat([h_last[0], h_last[1]], dim=-1)  # [B, 2H]
            else:
                feat = hn[-1]                # [B, H]
        else:  # "attn"
            feat = self._attn_pool(out, mask)

        # Normalization
        if self.norm is not None:
            if isinstance(self.norm, nn.BatchNorm1d):
                feat = self.norm(feat)       # [B, C]
            else:
                feat = self.norm(feat)       # [B, C]

        logits = self.head(self.head_dropout(feat))  # [B, num_classes]
        return logits

# ----- Perceiver-style latent cross-attention resampler -----
class PerceiverResampler(nn.Module):
    def __init__(self, model_dim: int, num_latents: int = 16, num_heads: int = 4, dropout: float = 0.1, num_layers: int = 2):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, model_dim))
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "ln_q": nn.LayerNorm(model_dim),
                "ln_kv": nn.LayerNorm(model_dim),
                "attn": nn.MultiheadAttention(
                    embed_dim=model_dim, num_heads=num_heads, dropout=dropout, batch_first=True
                ),
                "drop": nn.Dropout(dropout),
                "ffn": nn.Sequential(
                    nn.LayerNorm(model_dim),
                    nn.Linear(model_dim, model_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(model_dim * 4, model_dim),
                    nn.Dropout(dropout),
                )
            }) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None):
        # x: [B, T, C], key_padding_mask: [B, T] with True where pads should be ignored
        B, _, C = x.shape
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)  # [B, L, C]
        for blk in self.blocks:
            q = blk["ln_q"](latents)
            kv = blk["ln_kv"](x)
            attn_out, _ = blk["attn"](q, kv, kv, key_padding_mask=key_padding_mask)  # [B, L, C]
            latents = latents + blk["drop"](attn_out)
            latents = latents + blk["ffn"](latents)
        return latents  # [B, L, C]

# ----- Griffin-like classifier: BiGRU encoder + latent resampler + MLP head -----
class GriffinLikeClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 32,
        rnn_hidden: int = 128,
        num_layers: int = 1,
        bidirectional: bool = True,
        num_latents: int = 16,
        num_heads: int = 4,
        resampler_layers: int = 2,
        num_classes: int = 2,
        pad_id: int = 0,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        proj_dim: int = 128,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.bidirectional = bidirectional

        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.emb_drop = nn.Dropout(emb_dropout)

        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=rnn_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        feat_dim = rnn_hidden * (2 if bidirectional else 1)
        self.proj = nn.Identity()
        model_dim = feat_dim  # keep equal; change proj if you want a different model_dim

        self.resampler = PerceiverResampler(
            model_dim=model_dim,
            num_latents=num_latents,
            num_heads=num_heads,
            dropout=dropout,
            num_layers=resampler_layers,
        )

        self.norm = nn.LayerNorm(model_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(model_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, num_classes),
        )

    def forward(self, token_ids: torch.Tensor):
        # token_ids: [B, T] Long
        mask = (token_ids != self.pad_id)                # [B, T] True for real tokens
        key_padding_mask = ~mask                         # MultiheadAttention ignores True positions

        x = self.emb(token_ids)                          # [B, T, D]
        x = self.emb_drop(x)
        seq, _ = self.gru(x)                             # [B, T, H*dir]
        seq = self.proj(seq)                             # [B, T, C]

        latents = self.resampler(seq, key_padding_mask=key_padding_mask)  # [B, L, C]
        feat = self.norm(latents.mean(dim=1))            # [B, C] (mean over latents)

        logits = self.head(feat)                         # [B, num_classes]
        return logits