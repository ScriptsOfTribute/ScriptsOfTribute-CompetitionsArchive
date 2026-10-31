from typing import Any, Dict, List, Tuple
import torch
import torch.nn as nn
from torch.distributions import Categorical
from .layers import RMSNorm
from .move_encoder import MoveEncoder
from .state_encoder import StateEncoder

class MoveStateScorer(nn.Module):

    def __init__(self, state_dim: int, move_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.state_proj = nn.Linear(state_dim, move_dim)
        self.score_mlp = nn.Sequential(nn.Linear(move_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, state_emb: torch.Tensor, move_embs: torch.Tensor) -> torch.Tensor:
        if move_embs.dim() == 2:
            n = move_embs.shape[0]
            state_ctx = self.state_proj(state_emb)
            state_ctx = state_ctx.unsqueeze(0).expand(n, -1)
            x = torch.cat([move_embs, state_ctx], dim=-1)
            return self.score_mlp(x).squeeze(-1)
        m = move_embs.shape[1]
        state_ctx = self.state_proj(state_emb)
        state_ctx = state_ctx.unsqueeze(1).expand(-1, m, -1)
        x = torch.cat([move_embs, state_ctx], dim=-1)
        return self.score_mlp(x).squeeze(-1)

class PolicyNetwork(nn.Module):
    is_recurrent = False
    architecture = 'stateless'

    def __init__(self, config) -> None:
        super().__init__()
        self.state_encoder = StateEncoder(config)
        self.move_encoder = MoveEncoder(config, self.state_encoder.get_card_embedding())
        self.scorer = MoveStateScorer(config.state_embed_dim, config.move_embed_dim, config.move_embed_dim)
        self.temperature = 1.0
        self.value_head = nn.Sequential(RMSNorm(config.state_embed_dim), nn.Linear(config.state_embed_dim, 64), nn.GELU(), nn.Linear(64, 1), nn.Tanh())

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, tensor_dict: Dict[str, torch.Tensor], move_features_list: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
        state_emb = self.state_encoder(tensor_dict)
        move_embs = self.move_encoder.encode_moves(move_features_list, str(self.device))
        scores = self.scorer(state_emb, move_embs)
        return (scores / self.temperature, state_emb)

    def predict_value(self, tensor_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        state_emb = self.state_encoder(tensor_dict)
        return self.value_head(state_emb).squeeze(-1)

    def select_action(self, tensor_dict: Dict[str, torch.Tensor], move_features_list: List[Dict[str, Any]], deterministic: bool=False) -> Tuple[int, torch.Tensor, torch.Tensor]:
        logits, _ = self.forward(tensor_dict, move_features_list)
        dist = Categorical(logits=logits)
        if deterministic:
            action = logits.argmax()
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return (action.item(), log_prob, entropy)

    def evaluate_action(self, tensor_dict: Dict[str, torch.Tensor], move_features_list: List[Dict[str, Any]], action_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, _ = self.forward(tensor_dict, move_features_list)
        dist = Categorical(logits=logits)
        action_t = torch.tensor(action_idx, device=logits.device)
        return (dist.log_prob(action_t), dist.entropy())

    def evaluate_action_and_value(self, tensor_dict: Dict[str, torch.Tensor], move_features_list: List[Dict[str, Any]], action_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state_emb = self.state_encoder(tensor_dict)
        move_embs = self.move_encoder.encode_moves(move_features_list, str(self.device))
        logits = self.scorer(state_emb, move_embs) / self.temperature
        dist = Categorical(logits=logits)
        action_t = torch.tensor(action_idx, device=logits.device)
        value = self.value_head(state_emb).squeeze(-1)
        return (dist.log_prob(action_t), dist.entropy(), value)

    def evaluate_actions_batch(self, state_batch: Dict[str, torch.Tensor], move_batch: Dict[str, torch.Tensor], actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state_emb = self.state_encoder.forward_collated(state_batch)
        move_embs = self.move_encoder.forward_collated(move_batch)
        logits = self.scorer(state_emb, move_embs) / self.temperature
        neg = torch.finfo(logits.dtype).min / 2
        logits = logits.masked_fill(~move_batch['move_mask'], neg)
        dist = Categorical(logits=logits)
        values = self.value_head(state_emb).squeeze(-1)
        return (dist.log_prob(actions), dist.entropy(), values)

    def predict_values_batch(self, state_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        state_emb = self.state_encoder.forward_collated(state_batch)
        return self.value_head(state_emb).squeeze(-1)
