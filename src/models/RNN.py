import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

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
        # Accept: 
        # - List[LongTensor[T_i]]
        # - Tensor[B, T]
        # - Tensor[T] (single sequence)
        if isinstance(token_ids, (list, tuple)):
            device = self.emb.weight.device
            lengths = torch.as_tensor([t.size(0) for t in token_ids], device=device)
            token_ids = pad_sequence(token_ids, batch_first=True, padding_value=self.pad_id)  # [B, T_max]
        else:
            if token_ids.dim() == 1:
                token_ids = token_ids.unsqueeze(0)  # [1, T]
            lengths = (token_ids != self.pad_id).sum(dim=1)

        mask = (token_ids != self.pad_id)  # [B, T_max]

        x = self.emb(token_ids)  # [B, T_max, D]
        x = self.emb_dropout(x)

        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, hn = self.gru(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)  # [B, T_max, H*dir]

        # Pooling
        if self.pooling == "mean":
            feat = self._masked_mean(out, mask)
        elif self.pooling == "last":
            if self.bidirectional:
                h_last = hn[-2:]  # [2, B, H]
                feat = torch.cat([h_last[0], h_last[1]], dim=-1)  # [B, 2H]
            else:
                feat = hn[-1]  # [B, H]
        else:  # "attn"
            feat = self._attn_pool(out, mask)

        # Normalization
        if self.norm is not None:
            feat = self.norm(feat)

        logits = self.head(self.head_dropout(feat))  # [B, num_classes]
        return logits





# One-hot GRU that consumes per-token vectors directly (no embedding/projection).
class OneHot_RNN_MLP_KmerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,        # V (e.g., 4 for A,C,T,G, or K*A if flattened tokens)
        rnn_hidden: int = 128,
        num_layers: int = 1,
        bidirectional: bool = True,
        num_classes: int = 2,
        dropout: float = 0.1,
        input_dropout: float = 0.1,
        pooling: str = "mean",  # "mean" | "attn" | "last"
        norm: str = "layer",
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.pooling = pooling

        self.input_dropout = nn.Dropout(input_dropout)
        self.gru = nn.GRU(
            input_size=vocab_size,
            hidden_size=rnn_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        feat_dim = rnn_hidden * (2 if bidirectional else 1)
        self.attn = nn.Linear(feat_dim, 1) if pooling == "attn" else None

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
        # x: [B,T,C], mask: [B,T]
        mask = mask.unsqueeze(-1)
        x = x * mask
        denom = mask.sum(dim=1).clamp_min(1)
        return x.sum(dim=1) / denom

    def _attn_pool(self, x, mask):
        # x: [B,T,C], mask: [B,T]
        scores = self.attn(x).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=1).unsqueeze(-1)
        return (x * weights).sum(dim=1)

    def forward(self, x_padded: torch.Tensor, lengths: torch.Tensor, mask: torch.Tensor = None):
        """
        x_padded: [B, T, V] float (already padded with zeros)
        lengths : [B] long
        mask    : optional [B, T] bool; if None, built from lengths
        """
        assert x_padded.dim() == 3, f"Expected [B,T,V], got {x_padded.shape}"
        # Make sure dtype/contiguity are cuDNN-friendly
        x = x_padded.to(dtype=torch.float32)
        if not x.is_contiguous():
            x = x.contiguous()

        B, T, V = x.shape
        assert V == self.gru.input_size, f"V={V} must equal vocab_size={self.gru.input_size}"

        if mask is None:
            mask = torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)  # [B,T]

        x = self.input_dropout(x)
        # Also guard contiguity after dropout
        if not x.is_contiguous():
            x = x.contiguous()

        out, _ = self.gru(x)  # [B, T, H*dir]

        if self.pooling == "mean":
            feat = self._masked_mean(out, mask)
        elif self.pooling == "attn":
            feat = self._attn_pool(out, mask)
        else:  # "last" (use lengths indexing)
            last_idx = (lengths - 1).clamp_min(0)
            feat = out[torch.arange(B, device=out.device), last_idx]

        if self.norm is not None:
            feat = self.norm(feat)

        logits = self.head(self.head_dropout(feat))
        return logits