import torch
import torch.nn as nn


class TabularEncoder(nn.Module):
    """
    Dummy tabular encoder: passes input through unchanged.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        # No layers at all

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
