from typing import Any, Dict, List, Sequence, Tuple, Union
import torch
import torch.nn as nn
from torch.distributions import Categorical
from .layers import RMSNorm, bucket_length
from .move_encoder import MoveEncoder
from .policy_network import MoveStateScorer
from .state_encoder import StateEncoder, collate_tensor_dicts, slice_state_batch
Hidden = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]

class RecurrentPolicyNetwork(nn.Module):
    is_recurrent = True
    uses_previous_action = False
    architecture = 'recurrent'

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.state_encoder = StateEncoder(config)
        self.move_encoder = MoveEncoder(config, self.state_encoder.get_card_embedding())
        self.scorer = MoveStateScorer(config.state_embed_dim, config.move_embed_dim, config.move_embed_dim)
        self.temperature = 1.0
        cell = config.recurrent_cell.lower()
        if cell == 'gru':
            cell_cls = nn.GRU
        elif cell == 'lstm':
            cell_cls = nn.LSTM
        else:
            raise ValueError(f'Unknown recurrent_cell: {config.recurrent_cell!r}')
        self._cell_type = cell
        self.temporal = cell_cls(input_size=config.state_embed_dim, hidden_size=config.temporal_hidden_dim, num_layers=config.temporal_layers)
        fused_dim = config.state_embed_dim + config.temporal_hidden_dim
        self.temporal_fusion = nn.Sequential(RMSNorm(fused_dim), nn.Linear(fused_dim, config.state_embed_dim), nn.GELU(), nn.Linear(config.state_embed_dim, config.state_embed_dim))
        self.value_head = nn.Sequential(RMSNorm(config.state_embed_dim), nn.Linear(config.state_embed_dim, 64), nn.GELU(), nn.Linear(64, 1), nn.Tanh())

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @staticmethod
    def collate_states(tensor_dicts, device='cpu'):
        return collate_tensor_dicts(tensor_dicts, device)

    @staticmethod
    def slice_states(batch, idx, device):
        return slice_state_batch(batch, idx, device)

    def collate_previous_actions(self, actions, device='cpu'):
        return None

    def slice_previous_actions(self, batch, idx, device):
        return None

    def encode_recurrent_inputs(self, state_batch: Dict[str, torch.Tensor], previous_action_batch=None) -> Tuple[torch.Tensor, torch.Tensor]:
        state_emb = self.state_encoder.forward_collated(state_batch)
        return (state_emb, state_emb)

    def initial_hidden(self, batch_size: int=1) -> Hidden:
        h = torch.zeros(self.temporal.num_layers, batch_size, self.temporal.hidden_size, device=self.device)
        if self._cell_type == 'lstm':
            return (h, h.clone())
        return h

    @staticmethod
    def hidden_select(hidden: Hidden, idx: Sequence[int]) -> Hidden:
        if isinstance(hidden, tuple):
            return (hidden[0][:, idx], hidden[1][:, idx])
        return hidden[:, idx]

    @staticmethod
    def hidden_cat(hiddens: List[Hidden]) -> Hidden:
        if isinstance(hiddens[0], tuple):
            return (torch.cat([h[0] for h in hiddens], dim=1), torch.cat([h[1] for h in hiddens], dim=1))
        return torch.cat(hiddens, dim=1)

    @staticmethod
    def hidden_to(hidden: Hidden, device: Union[str, torch.device]) -> Hidden:
        if isinstance(hidden, tuple):
            return (hidden[0].to(device), hidden[1].to(device))
        return hidden.to(device)

    def step_sequence(self, emb_seq: torch.Tensor, hidden: Hidden) -> Tuple[torch.Tensor, Hidden]:
        return self.temporal(emb_seq, hidden)

    def fuse(self, state_emb: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return state_emb + self.temporal_fusion(torch.cat([state_emb, h], dim=-1))

    def forward(self, tensor_dict: Dict[str, torch.Tensor], move_features_list: List[Dict[str, Any]], hidden: Hidden) -> Tuple[torch.Tensor, torch.Tensor, Hidden]:
        state_emb = self.state_encoder(tensor_dict)
        out, new_hidden = self.temporal(state_emb.view(1, 1, -1), hidden)
        ctx = self.fuse(state_emb, out[0, 0])
        move_embs = self.move_encoder.encode_moves(move_features_list, str(self.device))
        scores = self.scorer(ctx, move_embs)
        return (scores / self.temperature, ctx, new_hidden)

    def select_action(self, tensor_dict: Dict[str, torch.Tensor], move_features_list: List[Dict[str, Any]], hidden: Hidden, deterministic: bool=False) -> Tuple[int, torch.Tensor, torch.Tensor, Hidden]:
        logits, _, new_hidden = self.forward(tensor_dict, move_features_list, hidden)
        dist = Categorical(logits=logits)
        if deterministic:
            action = logits.argmax()
        else:
            action = dist.sample()
        return (action.item(), dist.log_prob(action), dist.entropy(), new_hidden)

    def run_segments(self, state_embs: torch.Tensor, seg_lengths: Sequence[int], init_hidden: Hidden) -> torch.Tensor:
        n_seg = len(seg_lengths)
        n_pad = bucket_length(n_seg)
        lengths = torch.tensor(seg_lengths, device=state_embs.device)
        l_pad = bucket_length(int(lengths.max().item()))
        s_idx = torch.repeat_interleave(torch.arange(n_seg, device=state_embs.device), lengths)
        offsets = torch.cumsum(lengths, dim=0) - lengths
        t_idx = torch.arange(state_embs.shape[0], device=state_embs.device) - offsets[s_idx]
        buf = torch.zeros(l_pad, n_pad, state_embs.shape[1], device=state_embs.device, dtype=state_embs.dtype)
        buf[t_idx, s_idx] = state_embs
        out, _ = self.temporal(buf, self._pad_hidden(init_hidden, n_pad))
        return out[t_idx, s_idx]

    def _pad_hidden(self, hidden: Hidden, n_pad: int) -> Hidden:

        def pad(h: torch.Tensor) -> torch.Tensor:
            if h.shape[1] == n_pad:
                return h
            zeros = torch.zeros(h.shape[0], n_pad - h.shape[1], h.shape[2], device=h.device, dtype=h.dtype)
            return torch.cat([h, zeros], dim=1)
        if isinstance(hidden, tuple):
            return (pad(hidden[0]), pad(hidden[1]))
        return pad(hidden)

    def score_context_batch(self, ctx: torch.Tensor, move_batch: Dict[str, torch.Tensor], actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        move_embs = self.move_encoder.forward_collated(move_batch)
        logits = self.scorer(ctx, move_embs) / self.temperature
        neg = torch.finfo(logits.dtype).min / 2
        logits = logits.masked_fill(~move_batch['move_mask'], neg)
        dist = Categorical(logits=logits)
        values = self.value_head(ctx).squeeze(-1)
        return (dist.log_prob(actions), dist.entropy(), values)
