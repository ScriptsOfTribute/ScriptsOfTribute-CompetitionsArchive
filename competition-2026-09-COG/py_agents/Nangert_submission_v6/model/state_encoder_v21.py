from typing import Dict, List, Union
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from .layers import RMSNorm, bucket_length
from .state_encoder import StateEncoder, _pad_zone, collate_tensor_dicts, slice_state_batch
from ..utils.game_state_to_vector import PUBLIC_TOKEN_BUCKETS
_PUBLIC_CARD_ZONES = ('draw', 'opp_hand_draw')
_PUBLIC_SEQUENCES = ('public_event_ids', 'upcoming_effect_ids', 'next_turn_effect_ids')

def _pad_token_sequences(values: List[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([value.shape[0] for value in values])
    ids = pad_sequence(values, batch_first=True)
    width = bucket_length(max(1, ids.shape[1]))
    if ids.shape[1] < width:
        ids = torch.cat([ids, torch.zeros(len(values), width - ids.shape[1], dtype=torch.long)], dim=1)
    mask = torch.arange(width)[None, :] < lengths[:, None]
    return (ids, mask)

def collate_tensor_dicts_v21(tensor_dicts: List[Dict[str, torch.Tensor]], device: Union[str, torch.device]='cpu') -> Dict[str, torch.Tensor]:
    batch = collate_tensor_dicts(tensor_dicts, 'cpu')

    def cpu(tensor: torch.Tensor) -> torch.Tensor:
        return tensor if tensor.device.type == 'cpu' else tensor.cpu()
    for zone in _PUBLIC_CARD_ZONES:
        ids, feats, mask = _pad_zone([cpu(td[f'{zone}_ids']) for td in tensor_dicts], [cpu(td[f'{zone}_feats']) for td in tensor_dicts])
        batch[f'{zone}_ids'] = ids
        batch[f'{zone}_feats'] = feats
        batch[f'{zone}_mask'] = mask
    for key in _PUBLIC_SEQUENCES:
        ids, mask = _pad_token_sequences([cpu(td[key]) for td in tensor_dicts])
        batch[key] = ids
        batch[f'{key}_mask'] = mask
    batch['choice_context_id'] = torch.stack([cpu(td['choice_context_id']) for td in tensor_dicts])
    batch['choice_followup_id'] = torch.stack([cpu(td['choice_followup_id']) for td in tensor_dicts])
    batch['public_scalars'] = torch.stack([cpu(td['public_scalars']) for td in tensor_dicts])
    return {key: value.to(device) for key, value in batch.items()}

def slice_state_batch_v21(batch: Dict[str, torch.Tensor], idx: torch.Tensor, device: Union[str, torch.device]) -> Dict[str, torch.Tensor]:
    out = slice_state_batch(batch, idx, device)
    for zone in _PUBLIC_CARD_ZONES:
        mask = batch[f'{zone}_mask'][idx]
        width = min(bucket_length(max(1, int(mask.sum(dim=1).max().item()))), mask.shape[1])
        out[f'{zone}_ids'] = batch[f'{zone}_ids'][idx, :width].to(device)
        out[f'{zone}_feats'] = batch[f'{zone}_feats'][idx, :width].to(device)
        out[f'{zone}_mask'] = mask[:, :width].to(device)
    for key in _PUBLIC_SEQUENCES:
        mask = batch[f'{key}_mask'][idx]
        width = min(bucket_length(max(1, int(mask.sum(dim=1).max().item()))), mask.shape[1])
        out[key] = batch[key][idx, :width].to(device)
        out[f'{key}_mask'] = mask[:, :width].to(device)
    for key in ('choice_context_id', 'choice_followup_id', 'public_scalars'):
        out[key] = batch[key][idx].to(device)
    return out

class StateEncoderV21(StateEncoder):
    _event_embed_dim = 32
    _event_hidden_dim = 64

    def __init__(self, config) -> None:
        super().__init__(config)
        hidden_dim = config.hidden_dim
        state_dim = config.state_embed_dim
        self.public_token_embedding = nn.Embedding(PUBLIC_TOKEN_BUCKETS, self._event_embed_dim, padding_idx=0)
        self.public_event_gru = nn.GRU(self._event_embed_dim, self._event_hidden_dim, batch_first=True)
        self.public_scalar_mlp = nn.Sequential(nn.Linear(5, 32), nn.GELU(), nn.Linear(32, 32))
        public_dim = 2 * hidden_dim + 3 * self._event_hidden_dim + 2 * self._event_embed_dim + 32
        self.public_residual = nn.Sequential(RMSNorm(public_dim), nn.Linear(public_dim, state_dim), nn.GELU(), nn.Linear(state_dim, state_dim))
        nn.init.normal_(self.public_residual[-1].weight, std=0.001)
        nn.init.zeros_(self.public_residual[-1].bias)

    def _event_single(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.numel() == 0:
            return torch.zeros(self._event_hidden_dim, device=ids.device, dtype=self.card_embedding.weight.dtype)
        _, hidden = self.public_event_gru(self.public_token_embedding(ids).unsqueeze(0))
        return hidden[-1, 0]

    def _event_batch(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.public_event_gru(self.public_token_embedding(ids))
        lengths = mask.sum(dim=1)
        last = (lengths - 1).clamp_min(0)
        rows = torch.arange(ids.shape[0], device=ids.device)
        gathered = outputs[rows, last]
        return gathered * (lengths > 0).unsqueeze(1).to(gathered.dtype)

    def _public_single(self, td: Dict[str, torch.Tensor]) -> torch.Tensor:
        card_zones = [self.card_zone_encoder(td[f'{zone}_ids'], td[f'{zone}_feats']) for zone in _PUBLIC_CARD_ZONES]
        events = [self._event_single(td[key]) for key in _PUBLIC_SEQUENCES]
        choices = [self.public_token_embedding(td['choice_context_id']), self.public_token_embedding(td['choice_followup_id'])]
        scalars = self.public_scalar_mlp(td['public_scalars'])
        public = torch.cat([*card_zones, *events, *choices, scalars])
        return self.public_residual(public)

    def _public_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        card_zones = [self.card_zone_encoder.forward_batch(batch[f'{zone}_ids'], batch[f'{zone}_feats'], batch[f'{zone}_mask']) for zone in _PUBLIC_CARD_ZONES]
        events = [self._event_batch(batch[key], batch[f'{key}_mask']) for key in _PUBLIC_SEQUENCES]
        choices = [self.public_token_embedding(batch['choice_context_id']), self.public_token_embedding(batch['choice_followup_id'])]
        scalars = self.public_scalar_mlp(batch['public_scalars'])
        return self.public_residual(torch.cat([*card_zones, *events, *choices, scalars], dim=-1))

    def forward(self, td: Dict[str, torch.Tensor]) -> torch.Tensor:
        return super().forward(td) + self._public_single(td)

    def forward_collated(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return super().forward_collated(batch) + self._public_batch(batch)
