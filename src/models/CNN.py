import torch
from torch import nn

# ----- CNN model: embedding -> Conv1d blocks -> global pool -> classifier -----

class CNNKmerClassifier(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 emb_dim=128, 
                 conv_dim = 128, 
                 kernel_size = 7, 
                 num_classes=2, 
                 pad_id=0, 
                 dropout = 0.2):
        super().__init__()
        self.kernel_size = kernel_size
        # approximate 'same' padding per conv layer
        self.pad = kernel_size // 2
        
        self.head_dropout = nn.Dropout(dropout)
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        # Reduce downsampling to avoid zero-length tensors. Use only two stride-2 stages and no max-pooling.
        self.conv = nn.Sequential(
            nn.Conv1d(emb_dim, 128, kernel_size=kernel_size, padding=self.pad, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout1d(dropout),

            nn.Conv1d(128, conv_dim, kernel_size=kernel_size, padding=self.pad, stride=2),
            nn.ReLU(inplace=True),
            nn.Dropout1d(dropout),

        )
        self.pool = nn.AdaptiveMaxPool1d(1)  # → [B, C, 1] (Maxpool?)
        self.fc = nn.Linear(conv_dim, num_classes)

    def forward(self, token_ids):
        # token_ids: [B, T] Long
        x = self.emb(token_ids)          # [B, T, D]
        x = x.transpose(1, 2)            # [B, D, T] for Conv1d

        z = self.conv(x)                 # [B, C, T']

        
        feat = self.pool(z).squeeze(-1)     # [B, C]
        
        logits = self.fc(self.head_dropout(feat))              # [B, num_classes]
        return logits


# ----- CNN model: embedding -> Conv1d blocks -> global pool -> classifier -----

class CNNKmerClassifierLarge(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 emb_dim=128, 
                 conv_dim = 128, 
                 nn_dim = 64,
                 kernel_size = 7, 
                 num_classes=2, 
                 pad_id=0, 
                 dropout = 0.2):
        super().__init__()
        # self.kernel_size = kernel_size
        # approximate 'same' padding per conv layer
        self.pad = kernel_size // 2
        
        self.head_dropout = nn.Dropout(dropout)
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        
        g = min(32, conv_dim)
        while g > 1 and (conv_dim % g) != 0:
            g //= 2 
        self.conv = nn.Sequential(
            nn.Conv1d(emb_dim, conv_dim, kernel_size=kernel_size + (kernel_size // 2), padding=self.pad, stride=2),
            #nn.BatchNorm1d(conv_dim),
            nn.GroupNorm(g, conv_dim),
            nn.ReLU(inplace=True),
            nn.Dropout1d(dropout),

            nn.Conv1d(conv_dim, conv_dim, kernel_size=kernel_size, padding=self.pad, stride=2),
            nn.GroupNorm(g, conv_dim),
            nn.ReLU(inplace=True),
            nn.Dropout1d(dropout),

            
            nn.Conv1d(conv_dim, conv_dim, kernel_size=kernel_size * 2, padding=self.pad, stride=2),
            nn.GroupNorm(g, conv_dim),
            nn.ReLU(inplace=True),
            nn.Dropout1d(dropout),

        )
        self.pool = nn.AdaptiveAvgPool1d(1)  # → [B, C, 1] (Maxpool?)
        #self.fc1 = nn.Linear(conv_dim, num_classes)
        self.head = nn.Sequential(
            nn.GroupNorm(g, conv_dim),
            nn.Linear(conv_dim, nn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(nn_dim, nn_dim//2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(nn_dim//2, num_classes),
            )

        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, token_ids):
        # token_ids: [B, T] Long
        x = self.emb(token_ids)          # [B, T, D]
        x = x.transpose(1, 2)            # [B, D, T] for Conv1d

        z = self.conv(x)                 # [B, C, T']

        feat = self.pool(z).squeeze(-1)     # [B, C]
        
        logits = self.head(self.head_dropout(feat))              # [B, num_classes]
        return logits


