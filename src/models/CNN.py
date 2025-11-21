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

	def forward(self, token_ids, lengths=None, mask=None):
		# token_ids: [B, T] Long
		x = self.emb(token_ids)          # [B, T, D]
		x = x.transpose(1, 2)            # [B, D, T] for Conv1d

		z = self.conv(x)                 # [B, C, T']

		
		feat = self.pool(z).squeeze(-1)     # [B, C]
		
		logits = self.fc(self.head_dropout(feat))              # [B, num_classes]
		return logits
	
class CNNKmerClassifier_v2(nn.Module):
	def __init__(self, vocab_size, emb_dim=96, conv_dim=96, kernel_size=7, num_classes=2, pad_id=0, dropout=0.2):
		super().__init__()
		pad = kernel_size // 2
		self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
		self.bias = True
		self.conv = nn.Sequential(

			nn.Conv1d(emb_dim, conv_dim, kernel_size=kernel_size, padding=pad, stride=1, bias=self.bias),
			nn.BatchNorm1d(conv_dim),
			nn.ReLU(inplace=True),
			nn.Dropout1d(dropout),

			nn.Conv1d(conv_dim, conv_dim, kernel_size=kernel_size//2, padding=pad, stride=2, bias=self.bias),
			nn.BatchNorm1d(conv_dim),
			nn.ReLU(inplace=True),
			nn.Dropout1d(dropout),

			nn.Conv1d(conv_dim, conv_dim, kernel_size=kernel_size, padding=pad, stride=3, bias=self.bias),
			nn.BatchNorm1d(conv_dim),
			nn.ReLU(inplace=True),
			nn.Dropout1d(dropout),
		)
		#self.pool_max = nn.AdaptiveMaxPool1d(1)
		self.pool_avg = nn.AdaptiveAvgPool1d(1)
		self.head = nn.Sequential(
			nn.Linear(conv_dim, conv_dim),
			nn.ReLU(inplace=True),
			nn.Dropout(dropout),
			nn.Linear(conv_dim, num_classes),
		)

	def forward(self, token_ids, lengths=None, mask=None):
		x = self.emb(token_ids).transpose(1, 2)
		z = self.conv(x)
		feat = self.pool_avg(z).squeeze(-1)
		#feat = torch.cat([self.pool_max(z), self.pool_avg(z)], dim=1).squeeze(-1)
		return self.head(feat)


class CNN_ONEHOT_SMALL(nn.Module):
	def __init__(self, input_size, conv_dim=96, kernel_size=7, num_classes=2, dropout=0.2):
		super().__init__()
		pad_main = kernel_size // 2                 # 7 → 3
		k_small = kernel_size // 2                  # 3
		pad_small = k_small // 2                    # 1 → keeps length
		self.bias = True
		self.conv = nn.Sequential(
			nn.Conv1d(input_size, conv_dim, kernel_size=kernel_size, padding=pad_main, stride=1, bias=self.bias),
			nn.BatchNorm1d(conv_dim),
			nn.ReLU(inplace=True),
			nn.Dropout1d(dropout),

			nn.Conv1d(conv_dim, conv_dim, kernel_size=k_small, padding=pad_small, stride=1, bias=self.bias),
			nn.BatchNorm1d(conv_dim),
			nn.ReLU(inplace=True),
			nn.Dropout1d(dropout),
		)
		self.head = nn.Sequential(
			nn.Linear(conv_dim, conv_dim),
			nn.ReLU(inplace=True),
			nn.Dropout(dropout),
			nn.Linear(conv_dim, num_classes),
		)

	def forward(self, x_padded: torch.Tensor, lengths: torch.Tensor = None, mask: torch.Tensor = None):
		"""
		x_padded: [B, T, V] (from collate) or [B, V, T]
		lengths:  [B]
		mask:     [B, T]
		"""
		# Ensure channel-first for Conv1d
		# Ensure channel-first for Conv1d
		if x_padded.dim() == 3 and x_padded.size(1) != self.conv[0].in_channels and x_padded.size(2) == self.conv[0].in_channels:
			# Incoming shape [B, T, V] -> transpose to [B, V, T]
			x_padded = x_padded.transpose(1, 2).contiguous()

		z = self.conv(x_padded)  # [B, C, T]

		if mask is None:
			if lengths is not None:
				T = z.size(-1)
				mask = (torch.arange(T, device=z.device).unsqueeze(0) < lengths.unsqueeze(1)).to(z.dtype)
			else:
				mask = torch.ones(z.size(0), z.size(-1), device=z.device, dtype=z.dtype)
		else:
			mask = mask[:, :z.size(-1)].to(z.dtype)

		valid_counts = mask.sum(dim=-1).clamp_min(1).unsqueeze(1)
		masked_sum = (z * mask.unsqueeze(1)).sum(dim=-1)
		feat = masked_sum / valid_counts

		return self.head(feat)
	


class CNN_ONEHOT_MEDIUM(nn.Module):
	def __init__(self, input_size, conv_dim=128, kernel_size=15, num_classes=2, dropout=0.2):
		super().__init__()
		pad_1 = kernel_size // 2                 # 7 → 3
		k_2 = kernel_size // 2                  # 3
		pad_2 = k_2 // 2          
		k_3 = kernel_size // 4
		pad_3 = k_3 // 2          # 1 → keeps length
		self.bias = True
		self.conv = nn.Sequential(
			nn.Conv1d(input_size, conv_dim, kernel_size=kernel_size, padding=pad_1, stride=1, bias=self.bias),
			nn.BatchNorm1d(conv_dim),
			nn.ReLU(inplace=True),
			nn.Dropout1d(dropout),

			nn.Conv1d(conv_dim, conv_dim, kernel_size=k_2, padding=pad_2, stride=1, bias=self.bias),
			nn.BatchNorm1d(conv_dim),
			nn.ReLU(inplace=True),
			nn.Dropout1d(dropout),

			nn.Conv1d(conv_dim, conv_dim, kernel_size=k_2, padding=pad_2, stride=1, bias=self.bias),
			nn.BatchNorm1d(conv_dim),
			nn.ReLU(inplace=True),
			nn.Dropout1d(dropout),

			nn.Conv1d(conv_dim, conv_dim, kernel_size=k_3, padding=pad_3, stride=1, bias=self.bias),
			nn.BatchNorm1d(conv_dim),
			nn.ReLU(inplace=True),
			nn.Dropout1d(dropout),
		)
		self.head = nn.Sequential(
			nn.Linear(conv_dim, conv_dim//2),
			nn.ReLU(inplace=True),
			nn.Dropout(dropout),
			nn.Linear(conv_dim//2, conv_dim//2),
			nn.ReLU(inplace=True),
			nn.Dropout(dropout),
			nn.Linear(conv_dim//2, num_classes),
		)

	def forward(self, x_padded: torch.Tensor, lengths: torch.Tensor = None, mask: torch.Tensor = None):
		"""
		x_padded: [B, T, V] (from collate) or [B, V, T]
		lengths:  [B]
		mask:     [B, T]
		"""
		# Ensure channel-first for Conv1d
		# Ensure channel-first for Conv1d
		if x_padded.dim() == 3 and x_padded.size(1) != self.conv[0].in_channels and x_padded.size(2) == self.conv[0].in_channels:
			# Incoming shape [B, T, V] -> transpose to [B, V, T]
			x_padded = x_padded.transpose(1, 2).contiguous()

		z = self.conv(x_padded)  # [B, C, T]

		if mask is None:
			if lengths is not None:
				T = z.size(-1)
				mask = (torch.arange(T, device=z.device).unsqueeze(0) < lengths.unsqueeze(1)).to(z.dtype)
			else:
				mask = torch.ones(z.size(0), z.size(-1), device=z.device, dtype=z.dtype)
		else:
			mask = mask[:, :z.size(-1)].to(z.dtype)

		valid_counts = mask.sum(dim=-1).clamp_min(1).unsqueeze(1)
		masked_sum = (z * mask.unsqueeze(1)).sum(dim=-1)
		feat = masked_sum / valid_counts

		return self.head(feat)
	
class CNN_ONEHOT_LARGE(nn.Module):
	def __init__(self, input_size, conv_dim=128, kernel_size=15, num_classes=2, dropout=0.2):
		super().__init__()
		pad_1 = kernel_size // 2                 # 7 → 3
		k_2 = kernel_size // 2                  # 3
		pad_2 = k_2 // 2          
		k_3 = kernel_size // 4
		pad_3 = k_3 // 2          # 1 → keeps length
		self.bias = True
		self.conv = nn.Sequential(
			
			nn.Conv1d(input_size, conv_dim, kernel_size=50, padding=50//2, stride=2, bias=self.bias),
			nn.BatchNorm1d(conv_dim),
			nn.ReLU(inplace=True),
			nn.Dropout1d(dropout),

			

			nn.Conv1d(conv_dim, conv_dim, kernel_size=kernel_size, padding=pad_1, stride=1, bias=self.bias),
			nn.BatchNorm1d(conv_dim),
			nn.ReLU(inplace=True),
			nn.Dropout1d(dropout),

			nn.Conv1d(conv_dim, conv_dim, kernel_size=k_2, padding=pad_2, stride=1, bias=self.bias),
			nn.BatchNorm1d(conv_dim),
			nn.ReLU(inplace=True),
			nn.Dropout1d(dropout),

			
			nn.Conv1d(conv_dim, conv_dim, kernel_size=k_3, padding=pad_3, stride=1, bias=self.bias),
			nn.BatchNorm1d(conv_dim),
			nn.ReLU(inplace=True),
			nn.Dropout1d(dropout),

			nn.Conv1d(conv_dim, conv_dim, kernel_size=k_3, padding=pad_3, stride=1, bias=self.bias),
			nn.BatchNorm1d(conv_dim),
			nn.ReLU(inplace=True),
			nn.Dropout1d(dropout),
		)
		self.head = nn.Sequential(
			nn.Linear(conv_dim, conv_dim//2),
			nn.ReLU(inplace=True),
			nn.Dropout(dropout),
			nn.Linear(conv_dim//2, conv_dim//2),
			nn.ReLU(inplace=True),
			nn.Dropout(dropout),
			nn.Linear(conv_dim//2, num_classes),
		)

	def forward(self, x_padded: torch.Tensor, lengths: torch.Tensor = None, mask: torch.Tensor = None):
		"""
		x_padded: [B, T, V] (from collate) or [B, V, T]
		lengths:  [B]
		mask:     [B, T]
		"""
		# Ensure channel-first for Conv1d
		# Ensure channel-first for Conv1d
		if x_padded.dim() == 3 and x_padded.size(1) != self.conv[0].in_channels and x_padded.size(2) == self.conv[0].in_channels:
			# Incoming shape [B, T, V] -> transpose to [B, V, T]
			x_padded = x_padded.transpose(1, 2).contiguous()

		z = self.conv(x_padded)  # [B, C, T]

		if mask is None:
			if lengths is not None:
				T = z.size(-1)
				mask = (torch.arange(T, device=z.device).unsqueeze(0) < lengths.unsqueeze(1)).to(z.dtype)
			else:
				mask = torch.ones(z.size(0), z.size(-1), device=z.device, dtype=z.dtype)
		else:
			mask = mask[:, :z.size(-1)].to(z.dtype)

		valid_counts = mask.sum(dim=-1).clamp_min(1).unsqueeze(1)
		masked_sum = (z * mask.unsqueeze(1)).sum(dim=-1)
		feat = masked_sum / valid_counts

		return self.head(feat)


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

	def forward(self, token_ids, lengths=None, mask=None):
		# token_ids: [B, T] Long
		x = self.emb(token_ids)          # [B, T, D]
		x = x.transpose(1, 2)            # [B, D, T] for Conv1d

		z = self.conv(x)                 # [B, C, T']

		feat = self.pool(z).squeeze(-1)     # [B, C]
		
		logits = self.head(self.head_dropout(feat))              # [B, num_classes]
		return logits



class CNNKmerClassifier_w_embeddings(nn.Module):
	def __init__(self, 
				 emb_dim=128, 
				 conv_dim=128, 
				 kernel_size=7, 
				 num_classes=2, 
				 dropout=0.2):
		super().__init__()
		self.kernel_size = kernel_size
		self.pad = kernel_size // 2
		
		self.head_dropout = nn.Dropout(dropout)
		# Reduce downsampling to avoid zero-length tensors. Use only two stride-2 stages and no max-pooling.
		self.conv = nn.Sequential(
			nn.Conv1d(emb_dim, conv_dim, kernel_size=kernel_size, padding=self.pad, stride=1),
			nn.ReLU(inplace=True),
			nn.Dropout1d(dropout),

			nn.Conv1d(conv_dim, conv_dim, kernel_size=kernel_size, padding=self.pad, stride=2),
			nn.ReLU(inplace=True),
			nn.Dropout1d(dropout),

		)
		self.pool = nn.AdaptiveMaxPool1d(1)  # → [B, C, 1] (Maxpool?)
		self.fc = nn.Linear(conv_dim, num_classes)

	def forward(self, embeddings, lengths=None, mask=None):
		# token_ids: [B, T] Long
		x = embeddings.transpose(1, 2)   # [B, D, T]
		z = self.conv(x)                 # [B, C, T']
		feat = self.pool(z).squeeze(-1)     # [B, C]
		logits = self.fc(self.head_dropout(feat))            # [B, num_classes]
		return logits


class CNNKmerClassifierLarge_w_embeddings(nn.Module):
	def __init__(self, 
				 emb_dim=128, 
				 conv_dim=128, 
				 kernel_size=7, 
				 num_classes=2, 
				 dropout=0.2):
		super().__init__()
		self.kernel_size = kernel_size
		self.pad = kernel_size // 2
		
		self.head_dropout = nn.Dropout(dropout)
		# Reduce downsampling to avoid zero-length tensors. Use only two stride-2 stages and no max-pooling.
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

		self.pool = nn.AdaptiveMaxPool1d(1)  # → [B, C, 1] (Maxpool?)
		self.fc = nn.Linear(conv_dim, num_classes)

	def forward(self, embeddings):
		# token_ids: [B, T] Long
		x = embeddings.transpose(1, 2)   # [B, D, T]
		z = self.conv(x)                 # [B, C, T']
		feat = self.pool(z).squeeze(-1)     # [B, C]
		logits = self.fc(self.head_dropout(feat))            # [B, num_classes]
		return logits
