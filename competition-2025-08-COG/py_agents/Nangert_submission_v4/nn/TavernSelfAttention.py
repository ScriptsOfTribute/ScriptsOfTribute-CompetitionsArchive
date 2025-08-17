import torch
import torch.nn as nn

class TavernSelfAttention(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.q = nn.Linear(dim, hidden_dim)
        self.k = nn.Linear(dim, hidden_dim)
        self.v = nn.Linear(dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
            q = self.q(x)
            k = self.k(x)
            v = self.v(x)
            attn = torch.softmax(q @ k.transpose(-2, -1) / (k.size(-1) ** 0.5), dim=-1)
            ctx = (attn @ v).mean(dim=1)
            return self.out(ctx)