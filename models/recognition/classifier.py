import torch
import torch.nn as nn


class SignClassifier(nn.Module):
    """Per-frame keypoint embedding -> temporal encoder (GRU or Transformer) -> pooled clip logits."""

    def __init__(self, in_features, num_classes, arch="gru", hidden=128, num_layers=2, dropout=0.3, max_frames=64):
        super().__init__()
        self.arch = arch
        self.embed = nn.Sequential(nn.Linear(in_features, hidden * 2), nn.LayerNorm(hidden * 2), nn.ReLU())

        if arch == "gru":
            self.encoder = nn.GRU(
                hidden * 2, hidden, num_layers=num_layers, batch_first=True, bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        elif arch == "transformer":
            self.position = nn.Parameter(torch.zeros(1, max_frames, hidden * 2))
            layer = nn.TransformerEncoderLayer(
                hidden * 2, nhead=4, dim_feedforward=hidden * 4, dropout=dropout, batch_first=True
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        else:
            raise ValueError(f"Unknown arch: {arch}")

        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden * 4, num_classes))

    def forward(self, x):
        h = self.embed(x)
        if self.arch == "gru":
            h, _ = self.encoder(h)
        else:
            h = self.encoder(h + self.position[:, : h.shape[1]])
        pooled = torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=-1)
        return self.head(pooled)
