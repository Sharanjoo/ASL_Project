import torch.nn as nn
import torchvision


class KeypointHeatmapNet(nn.Module):
    """ResNet backbone + 3 deconv layers -> one heatmap per keypoint at stride 4 (SimpleBaseline)."""

    def __init__(self, num_keypoints, backbone="resnet18", pretrained=True, deconv_channels=256):
        super().__init__()
        resnet = getattr(torchvision.models, backbone)(weights="DEFAULT" if pretrained else None)
        in_channels = resnet.fc.in_features
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        layers = []
        for _ in range(3):
            layers += [
                nn.ConvTranspose2d(in_channels, deconv_channels, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(deconv_channels),
                nn.ReLU(inplace=True),
            ]
            in_channels = deconv_channels
        self.deconv = nn.Sequential(*layers)
        self.head = nn.Conv2d(deconv_channels, num_keypoints, kernel_size=1)

    def forward(self, x):
        return self.head(self.deconv(self.backbone(x)))


def weighted_mse(pred, target, weight):
    return (((pred - target) ** 2) * weight[:, :, None, None]).mean()
