import numpy as np
import torch
import torch.nn as nn
import torchvision

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class FrameGRUClassifier(nn.Module):
    """Raw-frame baseline: ResNet-18 on every frame -> bidirectional GRU -> pooled clip logits.

    Same temporal head as models/recognition/classifier.py, so the only difference from the
    keypoint pipeline is what each frame is turned into: appearance features, not keypoints.
    """

    def __init__(self, num_classes, hidden=128, num_layers=2, dropout=0.3, pretrained=True, freeze_backbone=False):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = torchvision.models.resnet18(weights=weights)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.embed = nn.Sequential(nn.Linear(512, hidden * 2), nn.LayerNorm(hidden * 2), nn.ReLU())
        self.encoder = nn.GRU(
            hidden * 2, hidden, num_layers=num_layers, batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden * 4, num_classes))

    def forward(self, x):
        """x: (B, T, 3, H, W) ImageNet-normalised."""
        b, t = x.shape[:2]
        h = self.backbone(x.flatten(0, 1)).view(b, t, -1)
        h, _ = self.encoder(self.embed(h))
        pooled = torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=-1)
        return self.head(pooled)


def frames_to_tensor(frames):
    """(T, H, W, 3) uint8 RGB -> (T, 3, H, W) float32, ImageNet-normalised."""
    x = (frames.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(0, 3, 1, 2).copy())
