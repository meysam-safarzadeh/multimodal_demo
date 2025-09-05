import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Literal
from torchvision.models import ResNet18_Weights, ResNet50_Weights, EfficientNet_B0_Weights, ViT_B_16_Weights


class ImageEncoderClassification(nn.Module):

    SUPPORTED_MODELS = Literal['resnet18', 'resnet50', 'efficientnet_b0', 'vit_b_16']

    def __init__(
        self,
        backbone: SUPPORTED_MODELS = 'resnet18',
        pretrained: bool = True,
        out_dim: int = 256,
        in_channels: int = 3,
        freeze_backbone: bool = False
    ):
        """
        Args:
            backbone: Model backbone to use.
            pretrained: Load pretrained weights.
            out_dim: Size of output feature vector.
            in_channels: Number of input channels (e.g., 3 for RGB, 1 for grayscale/thermal).
            freeze_backbone: If True, backbone weights are frozen.
        """
        super().__init__()

        self.backbone_name = backbone
        self.pretrained = pretrained
        self.out_dim = out_dim
        self.in_channels = in_channels

        # Build backbone model
        if backbone == 'resnet18':
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            self.base = models.resnet18(weights=weights)
        elif backbone == 'resnet50':
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            self.base = models.resnet50(weights=weights)
        elif backbone == 'efficientnet_b0':
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.base = models.efficientnet_b0(weights=weights)
        elif backbone == 'vit_b_16':
            weights = ViT_B_16_Weights.DEFAULT if pretrained else None
            self.base = models.vit_b_16(weights=weights)
        else:
            raise ValueError(f"Backbone '{backbone}' not supported.")

        # Adapt input channels if needed (for non-RGB images)
        if in_channels != 3:
            # Change first conv layer
            if hasattr(self.base, 'conv1'):
                conv1 = self.base.conv1
                self.base.conv1 = nn.Conv2d(
                    in_channels, conv1.out_channels,
                    kernel_size=conv1.kernel_size,
                    stride=conv1.stride,
                    padding=conv1.padding,
                    bias=conv1.bias is not None
                )
            # For ViT and EfficientNet, you may need to adapt the patch embedding (extra work if needed)

        # Remove classification head, keep feature extractor
        if 'resnet' in backbone:
            self.feature_dim = self.base.fc.in_features
            self.base = nn.Sequential(*list(self.base.children())[:-1])  # Removes FC layer
        elif 'efficientnet' in backbone:
            self.feature_dim = self.base.classifier[1].in_features
            self.base.classifier = nn.Identity()
        elif 'vit' in backbone:
            self.feature_dim = self.base.heads.head.in_features
            self.base.heads = nn.Identity()

        # Optional: Freeze backbone
        if freeze_backbone:
            for param in self.base.parameters():
                param.requires_grad = False

        # Final projection to out_dim (if needed)
        if out_dim != self.feature_dim:
            self.projector = nn.Linear(self.feature_dim, out_dim)
        else:
            self.projector = nn.Identity()

        # Normalize output if you wish (optional)
        # self.norm = nn.BatchNorm1d(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] input images
        Returns:
            Feature tensor of shape [B, out_dim]
        """
        feats = self.base(x)
        if feats.ndim == 4:  # For ResNet: [B, F, 1, 1]
            feats = feats.view(feats.size(0), -1)
        feats = self.projector(feats)
        # feats = self.norm(feats)  # Optional normalization
        return feats

