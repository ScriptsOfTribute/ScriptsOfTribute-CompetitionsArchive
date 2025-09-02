import math
import torch
import torch.nn as nn

class CrossAttentionScorer(nn.Module):
    def __init__(self, context_dim: int, move_dim: int, attn_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        assert attn_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(context_dim, attn_dim)
        self.k_proj = nn.Linear(move_dim, attn_dim)

    def forward(self, context: torch.Tensor, move_emb: torch.Tensor) -> torch.Tensor:
        B, N, Dm = move_emb.shape

        Q = self.q_proj(context)
        K = self.k_proj(move_emb)

        Q = Q.view(B, self.num_heads, 1, self.head_dim)
        K = K.view(B, N, self.num_heads, self.head_dim)
        K = K.transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)).squeeze(-2)  # [B, H, N]
        scores = scores * self.scale

        logits = scores.mean(dim=1)
        return logits