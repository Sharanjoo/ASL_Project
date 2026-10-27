import torch
import torch.nn as nn


class TemporalRefiner(nn.Module):
    """Window of per-frame keypoints (t-r..t+r) -> refined keypoints at t.

    Coordinates are predicted as a residual on the centre frame's input; the score
    channel is re-estimated so downstream stages see how much to trust each point.
    """

    def __init__(self, num_keypoints=73, window=5, arch="transformer", hidden=256, num_layers=2, dropout=0.1):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.window = window
        self.arch = arch
        in_features = num_keypoints * 3

        self.embed = nn.Sequential(nn.Linear(in_features, hidden), nn.LayerNorm(hidden), nn.ReLU())
        if arch == "transformer":
            self.position = nn.Parameter(torch.zeros(1, window, hidden))
            layer = nn.TransformerEncoderLayer(hidden, nhead=4, dim_feedforward=hidden * 2, dropout=dropout, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
            out_features = hidden
        elif arch == "gru":
            self.encoder = nn.GRU(hidden, hidden // 2, num_layers=num_layers, batch_first=True, bidirectional=True)
            out_features = hidden
        else:
            raise ValueError(f"Unknown arch: {arch}")

        self.head = nn.Linear(out_features, num_keypoints * 3)

    def forward(self, windows):
        """windows: (N, window, K, 3) -> (N, K, 3)."""
        n = windows.shape[0]
        h = self.embed(windows.reshape(n, self.window, -1))
        if self.arch == "transformer":
            h = self.encoder(h + self.position)
        else:
            h, _ = self.encoder(h)
        out = self.head(h[:, self.window // 2]).reshape(n, self.num_keypoints, 3)

        centre = windows[:, self.window // 2]
        coords = centre[..., :2] + out[..., :2]
        score = torch.sigmoid(out[..., 2:3])
        return torch.cat([coords, score], dim=-1)


def refinement_loss(pred, target, score_threshold=0.2, score_weight=0.1, untrusted_weight=0.1):
    # Low-confidence clean points still pull weakly toward the clean estimate; with zero
    # weight their outputs drift freely and wipe out what the classifier relies on.
    trusted = (target[..., 2] >= score_threshold).float()
    point_weight = trusted + untrusted_weight * (1.0 - trusted)
    coord_error = (pred[..., :2] - target[..., :2]).abs().sum(dim=-1)
    coord_loss = (coord_error * point_weight).sum() / point_weight.sum().clamp(min=1.0)
    score_loss = (pred[..., 2] - target[..., 2]).abs().mean()
    return coord_loss + score_weight * score_loss
