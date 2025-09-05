import torch
import torch.nn as nn
from typing import List, Optional


class ClassifierHead(nn.Module):
    """
    Flexible, deep classifier head for multimodal feature fusion.
    Supports arbitrary MLP depth/width, dropout, batchnorm, and activations.
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        hidden_dims: Optional[List[int]] = None,
        activation: str = "relu",
        dropout: float = 0.0,
        use_batchnorm: bool = False
    ):
        """
        Args:
            in_dim: Input feature dimension (from fusion).
            num_classes: Number of output classes.
            hidden_dims: List of hidden layer sizes (e.g., [512, 256]).
            activation: Activation function ('relu', 'gelu', 'leakyrelu', etc.).
            dropout: Dropout probability (0 = off).
            use_batchnorm: If True, applies BatchNorm1d after each layer.
        """
        super().__init__()
        hidden_dims = hidden_dims or [512, 256]
        act = self._get_activation(activation)

        layers = []
        prev_dim = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h

        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

    def _get_activation(self, act_name: str):
        act_name = act_name.lower()
        if act_name == "relu":
            return nn.ReLU()
        elif act_name == "gelu":
            return nn.GELU()
        elif act_name == "leakyrelu":
            return nn.LeakyReLU()
        elif act_name == "elu":
            return nn.ELU()
        else:
            raise ValueError(f"Unsupported activation: {act_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_dim] fused feature tensor
        Returns:
            [B, num_classes] logits
        """
        return self.classifier(x)
