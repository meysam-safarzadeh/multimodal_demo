import torch
import torch.nn as nn
from typing import List, Union, Optional
from sentence_transformers import SentenceTransformer


class TextEncoder(nn.Module):
    """
    TextEncoder module that uses a pretrained SentenceTransformer
    to encode raw text inputs into dense vector embeddings.
    Designed to be used as a feature extractor in multimodal pipelines.
    Can be frozen so it does not update during training.
    """
    def __init__(
        self,
        backbone: str = "all-MiniLM-L6-v2",
        out_dim: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        freeze: bool = True
    ):
        super().__init__()
        self.model = SentenceTransformer(backbone, device=device)
        # Determine embedding dimension
        sample_output = self.model.encode(["sample"])
        self.embedding_dim = sample_output.shape[-1]
        # Optional projection layer
        if out_dim and out_dim != self.embedding_dim:
            self.proj = nn.Linear(self.embedding_dim, out_dim)
            self.out_dim = out_dim
        else:
            self.proj = None
            self.out_dim = self.embedding_dim
        if freeze:
            self.freeze_encoder()

    def freeze_encoder(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()  # Optional: set to eval mode for consistent inference

    def forward(self, texts: List[str]) -> torch.Tensor:
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        embeddings = torch.from_numpy(embeddings).to(next(self.parameters()).device)
        if self.proj is not None:
            embeddings = self.proj(embeddings)
        return embeddings
