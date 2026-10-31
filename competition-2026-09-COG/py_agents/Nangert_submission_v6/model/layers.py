import torch
import torch.nn as nn
import torch.nn.functional as F
_LENGTH_BUCKETS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096)

def bucket_length(n: int) -> int:
    for b in _LENGTH_BUCKETS:
        if b >= n:
            return b
    return n

class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float=1e-08) -> None:
        super().__init__()
        self.eps = eps
        self.scale = dim ** 0.5
        self.gain = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        return x * (self.scale / rms) * self.gain

class ResidualBlock(nn.Module):

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = RMSNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = F.gelu(self.linear1(h))
        h = self.linear2(h)
        return x + h

class MultiheadAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        assert d_model % num_heads == 0, 'd_model must be divisible by num_heads'
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, key_padding_mask: torch.Tensor | None=None) -> torch.Tensor:
        if Q.dim() == 2:
            n_q = Q.shape[0]
            n_k = K.shape[0]
            q = self.W_q(Q).view(n_q, self.num_heads, self.d_k).transpose(0, 1)
            k = self.W_k(K).view(n_k, self.num_heads, self.d_k).transpose(0, 1)
            v = self.W_v(V).view(n_k, self.num_heads, self.d_k).transpose(0, 1)
            scale = self.d_k ** 0.5
            attn = torch.matmul(q, k.transpose(-2, -1)) / scale
            attn = F.softmax(attn, dim=-1)
            out = torch.matmul(attn, v)
            out = out.transpose(0, 1).contiguous().view(n_q, self.d_model)
            return self.W_o(out)
        B, n_q, _ = Q.shape
        n_k = K.shape[1]
        q = self.W_q(Q).view(B, n_q, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(K).view(B, n_k, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(V).view(B, n_k, self.num_heads, self.d_k).transpose(1, 2)
        scale = self.d_k ** 0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        if key_padding_mask is not None:
            neg = torch.finfo(attn.dtype).min / 2
            attn = attn.masked_fill(~key_padding_mask[:, None, None, :], neg)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, n_q, self.d_model)
        return self.W_o(out)

class ISAB(nn.Module):

    def __init__(self, d_model: int, num_inducing: int, num_heads: int) -> None:
        super().__init__()
        self.inducing_points = nn.Parameter(torch.randn(num_inducing, d_model) * 0.02)
        self.norm1 = RMSNorm(d_model)
        self.attn1 = MultiheadAttention(d_model, num_heads)
        self.norm2 = RMSNorm(d_model)
        self.attn2 = MultiheadAttention(d_model, num_heads)

    def forward(self, X: torch.Tensor, key_padding_mask: torch.Tensor | None=None) -> torch.Tensor:
        if X.dim() == 2:
            I = self.inducing_points
            X_normed = self.norm1(X)
            H = I + self.attn1(I, X_normed, X_normed)
            H_normed = self.norm2(H)
            return X + self.attn2(X, H_normed, H_normed)
        B = X.shape[0]
        I = self.inducing_points.unsqueeze(0).expand(B, -1, -1)
        X_normed = self.norm1(X)
        H = I + self.attn1(I, X_normed, X_normed, key_padding_mask=key_padding_mask)
        H_normed = self.norm2(H)
        return X + self.attn2(X, H_normed, H_normed)

class PMA(nn.Module):

    def __init__(self, d_model: int, k: int, num_heads: int) -> None:
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(k, d_model) * 0.02)
        self.norm = RMSNorm(d_model)
        self.attn = MultiheadAttention(d_model, num_heads)
        self.output_dim = k * d_model

    def forward(self, X: torch.Tensor, key_padding_mask: torch.Tensor | None=None) -> torch.Tensor:
        if X.dim() == 2:
            X_normed = self.norm(X)
            S = self.seeds
            out = S + self.attn(S, X_normed, X_normed)
            return out.flatten()
        B = X.shape[0]
        X_normed = self.norm(X)
        S = self.seeds.unsqueeze(0).expand(B, -1, -1)
        out = S + self.attn(S, X_normed, X_normed, key_padding_mask=key_padding_mask)
        return out.flatten(1)

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model: int, num_heads: int, ffn_dim: int | None=None) -> None:
        super().__init__()
        ffn_dim = ffn_dim or 2 * d_model
        self.attn_norm = RMSNorm(d_model)
        self.attn = MultiheadAttention(d_model, num_heads)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, d_model))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X_normed = self.attn_norm(X)
        X = X + self.attn(X_normed, X_normed, X_normed)
        X = X + self.ffn(self.ffn_norm(X))
        return X

class ZoneTransformer(nn.Module):

    def __init__(self, d_model: int, num_layers: int, num_heads: int, num_zones: int, ffn_dim: int | None=None) -> None:
        super().__init__()
        self.zone_type_embed = nn.Embedding(num_zones, d_model)
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, num_heads, ffn_dim) for _ in range(num_layers)])
        self.output_norm = RMSNorm(d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        n = tokens.shape[-2]
        zone_ids = torch.arange(n, device=tokens.device)
        X = tokens + self.zone_type_embed(zone_ids)
        for layer in self.layers:
            X = layer(X)
        return self.output_norm(X)
