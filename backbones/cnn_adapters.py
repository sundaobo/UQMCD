import os

import torch
import torch.nn as nn


DEFAULT_OUT_CHANNELS = (96, 192, 384, 768)


def _load_pretrained_weights(model, ckpt_path, strict=False):
    if ckpt_path is None:
        return
    if not os.path.isfile(ckpt_path):
        print(f"Pretrained checkpoint not found: {ckpt_path}")
        return

    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model", "model_state_dict"):
                if key in checkpoint:
                    checkpoint = checkpoint[key]
                    break
        if not isinstance(checkpoint, dict):
            raise TypeError("checkpoint is not a state dict")

        state_dict = {}
        for key, value in checkpoint.items():
            clean_key = key
            for prefix in ("module.", "backbone.", "encoder."):
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix):]
            if clean_key.startswith(("fc.", "head.", "classifier.")):
                continue
            state_dict[clean_key] = value

        incompatible = model.load_state_dict(state_dict, strict=strict)
        print(f"Loaded pretrained weights from {ckpt_path}")
        print(incompatible)
    except Exception as exc:
        print(f"Failed to load pretrained weights from {ckpt_path}: {exc}")
        print("Initializing backbone randomly.")


class FeatureProjection(nn.Module):
    def __init__(self, in_channels, out_channels=DEFAULT_OUT_CHANNELS):
        super().__init__()
        self.out_channels = tuple(out_channels)
        self.proj = nn.ModuleList()
        for in_ch, out_ch in zip(in_channels, self.out_channels):
            if in_ch == out_ch:
                self.proj.append(nn.Identity())
            else:
                self.proj.append(
                    nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(inplace=True),
                    )
                )

    def forward(self, features):
        return [proj(feat) for proj, feat in zip(self.proj, features)]


class SiameseResNetAdapter(nn.Module):

    _STAGE_CHANNELS = {
        "resnet18": (64, 128, 256, 512),
        "resnet34": (64, 128, 256, 512),
        "resnet50": (256, 512, 1024, 2048),
        "resnet101": (256, 512, 1024, 2048),
        "resnet152": (256, 512, 1024, 2048),
    }

    def __init__(
        self,
        arch="resnet50",
        out_channels=DEFAULT_OUT_CHANNELS,
        pretrained_path=None,
    ):
        super().__init__()
        try:
            from torchvision import models
        except ImportError as exc:
            raise ImportError("torchvision is required for ResNet backbones") from exc

        arch = arch.lower()
        if arch not in self._STAGE_CHANNELS or not hasattr(models, arch):
            available = ", ".join(sorted(self._STAGE_CHANNELS))
            raise ValueError(f"Unsupported ResNet backbone '{arch}'. Available: {available}")

        self.arch = arch
        self.out_channels = tuple(out_channels)
        self.backbone = getattr(models, arch)(weights=None)
        _load_pretrained_weights(self.backbone, pretrained_path, strict=False)
        self.proj = FeatureProjection(self._STAGE_CHANNELS[arch], self.out_channels)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        l1 = self.backbone.layer1(x)
        l2 = self.backbone.layer2(l1)
        l3 = self.backbone.layer3(l2)
        l4 = self.backbone.layer4(l3)
        return self.proj([l1, l2, l3, l4])


class SiameseConvNeXtAdapter(nn.Module):

    _STAGE_CHANNELS = {
        "convnext_tiny": (96, 192, 384, 768),
        "convnext_small": (96, 192, 384, 768),
        "convnext_base": (128, 256, 512, 1024),
        "convnext_large": (192, 384, 768, 1536),
    }

    def __init__(
        self,
        arch="convnext_tiny",
        out_channels=DEFAULT_OUT_CHANNELS,
        pretrained_path=None,
    ):
        super().__init__()
        try:
            from torchvision import models
        except ImportError as exc:
            raise ImportError("torchvision is required for ConvNeXt backbones") from exc

        arch = arch.lower()
        if arch not in self._STAGE_CHANNELS or not hasattr(models, arch):
            available = ", ".join(sorted(self._STAGE_CHANNELS))
            raise ValueError(f"Unsupported ConvNeXt backbone '{arch}'. Available: {available}")

        self.arch = arch
        self.out_channels = tuple(out_channels)
        self.backbone = getattr(models, arch)(weights=None)
        _load_pretrained_weights(self.backbone, pretrained_path, strict=False)
        self.proj = FeatureProjection(self._STAGE_CHANNELS[arch], self.out_channels)

    def forward(self, x):
        features = []
        for idx, layer in enumerate(self.backbone.features):
            x = layer(x)
            if idx in (1, 3, 5, 7):
                features.append(x)
        return self.proj(features)
