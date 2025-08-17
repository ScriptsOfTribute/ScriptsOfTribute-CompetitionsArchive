import torch
import torch.nn as nn

class PatronOwnerEmbedding(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.owner_proj = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, one_hot: torch.Tensor) -> torch.Tensor:
        squeeze_time = False
        if one_hot.dim() == 3:
            one_hot = one_hot.unsqueeze(1)
            squeeze_time = True

        B, T, P, _ = one_hot.shape
        x = self.owner_proj(one_hot)
        x = x.mean(dim=2)

        if squeeze_time:
            x = x.squeeze(1)
        return x
