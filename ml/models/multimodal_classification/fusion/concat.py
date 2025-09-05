import torch
import torch.nn as nn
from typing import List, Optional


class ConcatVectors(nn.Module):
    """
    Concatenation-based fusion module for multimodal features.
    Accepts a variable number of modality feature tensors, concatenates along last dimension,
    and optionally projects to a specified output dimension.
    """

    def __init__(self, input_dims: List[int], out_dim: Optional[int] = None, use_norm: bool = False):
        """
        Args:
            input_dims: List of feature dimensions from each modality (e.g., [128, 256, 128])
            out_dim: If set, projects concatenated features to this dimension
            use_norm: If True, applies BatchNorm1d to the fused vector
        """
        super().__init__()
        self.input_dims = input_dims
        self.concat_dim = sum(input_dims)
        self.out_dim = out_dim if out_dim is not None else self.concat_dim

        # Optional projection layer (learnable fusion)
        if self.out_dim != self.concat_dim:
            self.projector = nn.Linear(self.concat_dim, self.out_dim)
        else:
            self.projector = nn.Identity()

        self.use_norm = use_norm
        if self.use_norm:
            self.norm = nn.BatchNorm1d(self.out_dim)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of [B, feat_dim] tensors (same batch size, arbitrary dims)
        Returns:
            [B, out_dim] fused feature tensor
        """
        # Ensure all batch sizes match
        batch_sizes = [f.size(0) for f in features]
        if len(set(batch_sizes)) != 1:
            raise ValueError(f"All input features must have the same batch size. Got: {batch_sizes}")

        # Optionally check feature dims match expected (can comment if unnecessary)
        for i, (f, expected_dim) in enumerate(zip(features, self.input_dims)):
            if f.size(1) != expected_dim:
                raise ValueError(f"Feature {i} has dim {f.size(1)}, expected {expected_dim}")

        # Concatenate along last dimension
        x = torch.cat(features, dim=-1)
        x = self.projector(x)
        if self.use_norm:
            x = self.norm(x)
        return x


class ConcatFeatureMaps(nn.Module):
    """
    Concatenates a list of feature maps along the channel (C) dimension.
    All input feature maps must have the same H and W.
    """
    def __init__(self):
        super().__init__()

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of [B, C_i, H, W] tensors
        Returns:
            fused: [B, sum(C_i), H, W]
        """
        if not features:
            raise ValueError("No features provided for fusion.")
        # Check spatial sizes
        spatial = features[0].shape[2:]
        for f in features:
            if f.shape[2:] != spatial:
                raise ValueError("All feature maps must have the same H, W dimensions.")
        # Concatenate along channel axis
        fused = torch.cat(features, dim=1)
        return fused
