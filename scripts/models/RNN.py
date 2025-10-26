import torch
from torch import nn

# ----- RNN model: embedding -> BiGRU -> masked global pool -> classifier -----
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
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, token_ids: torch.Tensor):
        # token_ids: [B, T] Long; mask: [B, T] Bool or 0/1
        
        x = self.emb(token_ids)  # [B, T, D]                            

        out, _ = self.gru(x.contiguous())                    
        
        # Global average over valid timesteps when lengths unknown
        feat = out.mean(dim=1)  # [B, H*dir]

        logits = self.fc(self.head_dropout(feat))  # [B, num_classes]
        return logits
