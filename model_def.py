"""
model_def.py
============
Definisi semua arsitektur model yang digunakan dalam training.
Disalin dari klasifikasi_gpu.py agar dapat di-import oleh main.py.
"""

import torch
import torch.nn as nn
from torchvision import transforms


# ===========================================================
# AKTIVASI KUSTOM
# ===========================================================

class HardSwish(nn.Module):
    def forward(self, x):
        return x * torch.nn.functional.relu6(x + 3) / 6


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# ===========================================================
# SQUEEZE-AND-EXCITATION BLOCK
# ===========================================================

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# ===========================================================
# BOTTLENECK BLOCK (untuk MobileNetSmall)
# ===========================================================

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, t, use_se=False, activation="RE"):
        super().__init__()
        self.stride = stride
        self.use_se = use_se
        expanded = int(in_channels * t)

        self.expansion = (
            nn.Sequential(
                nn.Conv2d(in_channels, expanded, 1, bias=False),
                nn.BatchNorm2d(expanded),
                nn.ReLU6(inplace=True) if activation == "RE" else HardSwish(),
            )
            if t != 1
            else None
        )

        self.depthwise = nn.Sequential(
            nn.Conv2d(
                expanded, expanded, 3,
                stride=stride, padding=1, groups=expanded, bias=False
            ),
            nn.BatchNorm2d(expanded),
            nn.ReLU6(inplace=True) if activation == "RE" else HardSwish(),
        )

        self.se = SEBlock(expanded) if use_se else None

        self.projection = nn.Sequential(
            nn.Conv2d(expanded, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        residual = x
        if self.expansion:
            x = self.expansion(x)
        x = self.depthwise(x)
        if self.se:
            x = self.se(x)
        x = self.projection(x)
        if self.stride == 1 and x.shape[1] == residual.shape[1]:
            x += residual
        return x


# ===========================================================
# MOBILENET SMALL (kustom)
# ===========================================================

class MobileNetSmall(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            HardSwish(),
        )
        self.bottlenecks = nn.Sequential(
            Bottleneck(16, 16,  stride=2, t=1,   use_se=True,  activation="RE"),
            Bottleneck(16, 16,  stride=2, t=4,                 activation="RE"),
            Bottleneck(16, 24,  stride=1, t=3,                 activation="RE"),
            Bottleneck(24, 24,  stride=2, t=3,   use_se=True,  activation="RE"),
            Bottleneck(24, 40,  stride=1, t=3,   use_se=True,  activation="HS"),
            Bottleneck(40, 40,  stride=1, t=3,   use_se=True,  activation="HS"),
            Bottleneck(40, 40,  stride=1, t=6,   use_se=True,  activation="HS"),
            Bottleneck(40, 48,  stride=1, t=2.5, use_se=True,  activation="HS"),
            Bottleneck(48, 48,  stride=1, t=2.3, use_se=True,  activation="HS"),
            Bottleneck(48, 96,  stride=1, t=2.3, use_se=True,  activation="HS"),
            Bottleneck(96, 96,  stride=1, t=6,   use_se=True,  activation="HS"),
            Bottleneck(96, 96,  stride=1, t=6,   use_se=True,  activation="HS"),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 576, 1, bias=False),
            nn.BatchNorm2d(576),
            HardSwish(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(576, 1024, 1, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bottlenecks(x)
        x = self.avgpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ===========================================================
# MBCONV BLOCK (untuk EfficientNetB0)
# ===========================================================

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 expand_ratio, se_ratio, drop_connect_rate=0.0):
        super().__init__()
        self.stride = stride
        self.drop_connect_rate = drop_connect_rate
        expanded = in_channels * expand_ratio

        self.expand_conv = (
            nn.Sequential(
                nn.Conv2d(in_channels, expanded, 1, bias=False),
                nn.BatchNorm2d(expanded),
                Swish(),
            )
            if expand_ratio != 1
            else None
        )

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                expanded, expanded, kernel_size,
                stride=stride, padding=kernel_size // 2,
                groups=expanded, bias=False
            ),
            nn.BatchNorm2d(expanded),
            Swish(),
        )

        self.se = SEBlock(expanded, reduction=int(1 / se_ratio)) if se_ratio > 0 else None

        self.project_conv = nn.Sequential(
            nn.Conv2d(expanded, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def drop_connect(self, x, rate):
        keep = 1.0 - rate
        mask = keep + torch.rand(
            [x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device
        )
        return x / keep * torch.floor(mask)

    def forward(self, x):
        identity = x
        if self.expand_conv:
            x = self.expand_conv(x)
        x = self.depthwise_conv(x)
        if self.se:
            x = self.se(x)
        x = self.project_conv(x)
        if self.stride == 1 and x.shape == identity.shape:
            if self.drop_connect_rate > 0:
                x = self.drop_connect(x, self.drop_connect_rate)
            x += identity
        return x


# ===========================================================
# EFFICIENTNET B0 (kustom)
# ===========================================================

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            Swish(),
        )
        self.blocks = nn.Sequential(
            MBConvBlock(32,   16,  3, 1, 1, 0.25),
            MBConvBlock(16,   24,  3, 2, 6, 0.25),
            MBConvBlock(24,   40,  5, 2, 6, 0.25),
            MBConvBlock(40,   80,  3, 2, 6, 0.25),
            MBConvBlock(80,   80,  5, 1, 6, 0.25),
            MBConvBlock(80,  112,  5, 2, 6, 0.25),
            MBConvBlock(112, 112,  5, 1, 6, 0.25),
            MBConvBlock(112, 112,  5, 1, 6, 0.25),
            MBConvBlock(112, 192,  5, 1, 6, 0.25),
            MBConvBlock(192, 192,  5, 1, 6, 0.25),
            MBConvBlock(192, 192,  5, 1, 6, 0.25),
            MBConvBlock(192, 192,  5, 1, 6, 0.25),
            MBConvBlock(192, 320,  3, 2, 6, 0.25),
            MBConvBlock(320, 1280, 1, 1, 6, 0.25),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1280, num_classes, 1, bias=True),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return torch.flatten(x, 1)


# ===========================================================
# ALEXNET KUSTOM
# ===========================================================

class AlexNetCustom(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Dropout(0.5),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 28 * 28, 500),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(500, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ===========================================================
# PREPROCESSING (SAMA SEPERTI SAAT TRAINING)
# ===========================================================

PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
