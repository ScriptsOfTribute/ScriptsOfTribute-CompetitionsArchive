from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from .layers import ISAB, PMA, ResidualBlock, RMSNorm, ZoneTransformer, bucket_length
from ..utils.encode_effects_string import EFFECT_DIM
from ..utils.game_state_to_vector import AGENT_FEAT_DIM, CARD_FEAT_DIM
_FIXED_KEYS = ('current_scalars', 'enemy_scalars', 'game_scalars', 'patron_states', 'combo_counts', 'choice_scalars')
_VARIABLE_ZONES = ('hand', 'played', 'cooldown', 'known', 'tavern', 'opp_played', 'opp_known', 'agents', 'opp_agents', 'choice_card')

def _pad_zone(ids_list: List[torch.Tensor], feats_list: List[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([x.shape[0] for x in ids_list], dtype=torch.long)
    ids = pad_sequence(ids_list, batch_first=True)
    feats = pad_sequence(feats_list, batch_first=True)
    b = len(ids_list)
    n_max = bucket_length(max(1, ids.shape[1]))
    if ids.shape[1] < n_max:
        pad_n = n_max - ids.shape[1]
        ids = torch.cat([ids, torch.zeros(b, pad_n, dtype=torch.long)], dim=1)
        feats = torch.cat([feats.reshape(b, -1, feats_list[0].shape[-1]), torch.zeros(b, pad_n, feats_list[0].shape[-1])], dim=1)
    mask = torch.arange(n_max)[None, :] < lengths[:, None]
    return (ids, feats, mask)

def collate_tensor_dicts(tds: List[Dict[str, torch.Tensor]], device: Union[str, torch.device]='cpu') -> Dict[str, torch.Tensor]:

    def _cpu(t: torch.Tensor) -> torch.Tensor:
        return t if t.device.type == 'cpu' else t.cpu()
    batch: Dict[str, torch.Tensor] = {}
    for key in _FIXED_KEYS:
        batch[key] = torch.stack([_cpu(td[key]) for td in tds])
    for zone in _VARIABLE_ZONES:
        ids, feats, mask = _pad_zone([_cpu(td[f'{zone}_ids']) for td in tds], [_cpu(td[f'{zone}_feats']) for td in tds])
        batch[f'{zone}_ids'] = ids
        batch[f'{zone}_feats'] = feats
        batch[f'{zone}_mask'] = mask
    vec_list = [_cpu(td['choice_effect_vecs']) for td in tds]
    lengths = torch.tensor([v.shape[0] for v in vec_list], dtype=torch.long)
    vecs = pad_sequence(vec_list, batch_first=True)
    e_max = bucket_length(max(1, vecs.shape[1]))
    if vecs.shape[1] < e_max:
        vecs = torch.cat([vecs.reshape(len(tds), -1, EFFECT_DIM), torch.zeros(len(tds), e_max - vecs.shape[1], EFFECT_DIM)], dim=1)
    batch['choice_effect_vecs'] = vecs
    batch['choice_effect_mask'] = torch.arange(e_max)[None, :] < lengths[:, None]
    return {k: v.to(device) for k, v in batch.items()}

def slice_state_batch(batch: Dict[str, torch.Tensor], idx: torch.Tensor, device: Union[str, torch.device]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key in _FIXED_KEYS:
        out[key] = batch[key][idx].to(device)
    for zone in _VARIABLE_ZONES:
        mask = batch[f'{zone}_mask'][idx]
        n = min(bucket_length(max(1, int(mask.sum(dim=1).max().item()))), mask.shape[1])
        out[f'{zone}_ids'] = batch[f'{zone}_ids'][idx, :n].to(device)
        out[f'{zone}_feats'] = batch[f'{zone}_feats'][idx, :n].to(device)
        out[f'{zone}_mask'] = mask[:, :n].to(device)
    eff_mask = batch['choice_effect_mask'][idx]
    e = min(bucket_length(max(1, int(eff_mask.sum(dim=1).max().item()))), eff_mask.shape[1])
    out['choice_effect_vecs'] = batch['choice_effect_vecs'][idx, :e].to(device)
    out['choice_effect_mask'] = eff_mask[:, :e].to(device)
    return out

def _encode_set_batch(encoder: Union['CardSetEncoder', 'AgentSetEncoder'], ids: torch.Tensor, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    card_emb = encoder.card_embedding(ids)
    feat_out = F.relu(encoder.feat_mlp(feats))
    combined = encoder.norm(F.gelu(encoder.combined_mlp(torch.cat([card_emb, feat_out], dim=-1))))
    attended = encoder.isab(combined, key_padding_mask=mask)
    pooled = encoder.pma(attended, key_padding_mask=mask)
    nonempty = mask.any(dim=1, keepdim=True).to(pooled.dtype)
    return pooled * nonempty

class CardSetEncoder(nn.Module):

    def __init__(self, card_embed_dim: int, feat_dim: int, hidden_dim: int, num_cards: int, card_embedding: Optional[nn.Embedding]=None, num_inducing: int=4, num_heads: int=4) -> None:
        super().__init__()
        self.card_embedding = card_embedding or nn.Embedding(num_cards, card_embed_dim)
        self.feat_mlp = nn.Linear(feat_dim, 16)
        self.combined_mlp = nn.Linear(card_embed_dim + 16, hidden_dim)
        self.norm = RMSNorm(hidden_dim)
        self.isab = ISAB(hidden_dim, num_inducing, num_heads)
        self.pma = PMA(hidden_dim, k=1, num_heads=num_heads)
        self.output_dim = hidden_dim
        self._hidden_dim = hidden_dim

    def forward(self, ids: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        if ids.shape[0] == 0:
            return torch.zeros(self.output_dim, device=feats.device)
        card_emb = self.card_embedding(ids)
        feat_out = F.relu(self.feat_mlp(feats))
        combined = self.norm(F.gelu(self.combined_mlp(torch.cat([card_emb, feat_out], dim=-1))))
        attended = self.isab(combined)
        return self.pma(attended)

    def forward_batch(self, ids: torch.Tensor, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return _encode_set_batch(self, ids, feats, mask)

class AgentSetEncoder(nn.Module):

    def __init__(self, card_embed_dim: int, feat_dim: int, hidden_dim: int, num_cards: int, card_embedding: Optional[nn.Embedding]=None, num_inducing: int=4, num_heads: int=4) -> None:
        super().__init__()
        self.card_embedding = card_embedding or nn.Embedding(num_cards, card_embed_dim)
        self.feat_mlp = nn.Linear(feat_dim, 16)
        self.combined_mlp = nn.Linear(card_embed_dim + 16, hidden_dim)
        self.norm = RMSNorm(hidden_dim)
        self.isab = ISAB(hidden_dim, num_inducing, num_heads)
        self.pma = PMA(hidden_dim, k=1, num_heads=num_heads)
        self.output_dim = hidden_dim
        self._hidden_dim = hidden_dim

    def forward(self, ids: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        if ids.shape[0] == 0:
            return torch.zeros(self.output_dim, device=feats.device)
        card_emb = self.card_embedding(ids)
        feat_out = F.relu(self.feat_mlp(feats))
        combined = self.norm(F.gelu(self.combined_mlp(torch.cat([card_emb, feat_out], dim=-1))))
        attended = self.isab(combined)
        return self.pma(attended)

    def forward_batch(self, ids: torch.Tensor, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return _encode_set_batch(self, ids, feats, mask)

class StateEncoder(nn.Module):
    _CARD_ZONE_KEYS = ('hand', 'played', 'cooldown', 'known', 'tavern', 'opp_played', 'opp_known')

    def __init__(self, config) -> None:
        super().__init__()
        card_embed_dim = config.card_embed_dim
        hidden_dim = config.hidden_dim
        num_cards = config.num_cards
        self.card_embedding = nn.Embedding(num_cards, card_embed_dim)
        self.scalar_mlp = nn.Linear(28, 64)
        self.patron_mlp = nn.Linear(30, 32)
        self.combo_mlp = nn.Linear(10, 16)
        self.choice_scalar_mlp = nn.Linear(4, 8)
        num_inducing = getattr(config, 'num_inducing_points', 4)
        num_heads = getattr(config, 'num_attention_heads', 4)
        self.card_zone_encoder = CardSetEncoder(card_embed_dim, CARD_FEAT_DIM, hidden_dim, num_cards, self.card_embedding, num_inducing, num_heads)
        self.agent_zone_encoder = AgentSetEncoder(card_embed_dim, AGENT_FEAT_DIM, hidden_dim, num_cards, self.card_embedding, num_inducing, num_heads)
        self.choice_card_encoder = CardSetEncoder(card_embed_dim, CARD_FEAT_DIM, hidden_dim, num_cards, self.card_embedding, num_inducing, num_heads)
        self.choice_effect_mlp = nn.Linear(EFFECT_DIM, 16)
        self.scalar_proj = nn.Linear(64, hidden_dim)
        self.patron_proj = nn.Linear(32, hidden_dim)
        self.combo_choice_proj = nn.Linear(24, hidden_dim)
        self.choice_effect_proj = nn.Linear(16, hidden_dim)
        num_zone_tokens = 13
        zone_layers = getattr(config, 'zone_transformer_layers', 2)
        zone_heads = getattr(config, 'zone_transformer_heads', 4)
        zone_ffn = getattr(config, 'zone_transformer_ffn_dim', 128)
        self.zone_transformer = ZoneTransformer(d_model=hidden_dim, num_layers=zone_layers, num_heads=zone_heads, num_zones=num_zone_tokens, ffn_dim=zone_ffn)
        fusion_input_dim = num_zone_tokens * hidden_dim
        self.fusion_proj = nn.Linear(fusion_input_dim, 384)
        self.fusion_block = ResidualBlock(384)
        self.fusion_norm = RMSNorm(384)
        self.fusion_out = nn.Linear(384, config.state_embed_dim)

    def forward(self, td: Dict[str, torch.Tensor]) -> torch.Tensor:
        scalars = torch.cat([td['current_scalars'], td['enemy_scalars'], td['game_scalars']])
        scalar_out = F.relu(self.scalar_mlp(scalars))
        patron_out = F.relu(self.patron_mlp(td['patron_states'].flatten()))
        combo_out = F.relu(self.combo_mlp(td['combo_counts']))
        choice_out = F.relu(self.choice_scalar_mlp(td['choice_scalars']))
        card_zone_outs = [self.card_zone_encoder(td[f'{k}_ids'], td[f'{k}_feats']) for k in self._CARD_ZONE_KEYS]
        agent_outs = [self.agent_zone_encoder(td['agents_ids'], td['agents_feats']), self.agent_zone_encoder(td['opp_agents_ids'], td['opp_agents_feats'])]
        choice_card_out = self.choice_card_encoder(td['choice_card_ids'], td['choice_card_feats'])
        choice_effect_out = self._encode_choice_effects(td['choice_effect_vecs'])
        scalar_token = F.gelu(self.scalar_proj(scalar_out))
        patron_token = F.gelu(self.patron_proj(patron_out))
        combo_choice_token = F.gelu(self.combo_choice_proj(torch.cat([combo_out, choice_out])))
        choice_effect_token = F.gelu(self.choice_effect_proj(choice_effect_out))
        choice_token = choice_card_out + choice_effect_token
        zone_tokens = torch.stack([*card_zone_outs, *agent_outs, choice_token, scalar_token, patron_token, combo_choice_token])
        enriched = self.zone_transformer(zone_tokens)
        fused = enriched.flatten()
        h = F.gelu(self.fusion_proj(fused))
        h = self.fusion_block(h)
        return self.fusion_out(self.fusion_norm(h))

    def forward_collated(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        scalars = torch.cat([batch['current_scalars'], batch['enemy_scalars'], batch['game_scalars']], dim=-1)
        scalar_out = F.relu(self.scalar_mlp(scalars))
        patron_out = F.relu(self.patron_mlp(batch['patron_states'].flatten(1)))
        combo_out = F.relu(self.combo_mlp(batch['combo_counts']))
        choice_out = F.relu(self.choice_scalar_mlp(batch['choice_scalars']))
        card_zone_outs = [self.card_zone_encoder.forward_batch(batch[f'{k}_ids'], batch[f'{k}_feats'], batch[f'{k}_mask']) for k in self._CARD_ZONE_KEYS]
        agent_outs = [self.agent_zone_encoder.forward_batch(batch['agents_ids'], batch['agents_feats'], batch['agents_mask']), self.agent_zone_encoder.forward_batch(batch['opp_agents_ids'], batch['opp_agents_feats'], batch['opp_agents_mask'])]
        choice_card_out = self.choice_card_encoder.forward_batch(batch['choice_card_ids'], batch['choice_card_feats'], batch['choice_card_mask'])
        eff_mask = batch['choice_effect_mask']
        eff_vecs = batch['choice_effect_vecs']
        mean_vec = (eff_vecs * eff_mask.unsqueeze(-1).to(eff_vecs.dtype)).sum(1) / eff_mask.sum(1).clamp(min=1).unsqueeze(-1)
        choice_effect_out = F.relu(self.choice_effect_mlp(mean_vec)) * eff_mask.any(dim=1, keepdim=True).to(eff_vecs.dtype)
        scalar_token = F.gelu(self.scalar_proj(scalar_out))
        patron_token = F.gelu(self.patron_proj(patron_out))
        combo_choice_token = F.gelu(self.combo_choice_proj(torch.cat([combo_out, choice_out], dim=-1)))
        choice_effect_token = F.gelu(self.choice_effect_proj(choice_effect_out))
        choice_token = choice_card_out + choice_effect_token
        zone_tokens = torch.stack([*card_zone_outs, *agent_outs, choice_token, scalar_token, patron_token, combo_choice_token], dim=1)
        enriched = self.zone_transformer(zone_tokens)
        fused = enriched.flatten(1)
        h = F.gelu(self.fusion_proj(fused))
        h = self.fusion_block(h)
        return self.fusion_out(self.fusion_norm(h))

    def forward_batch(self, tds: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        device = self.card_embedding.weight.device
        return self.forward_collated(collate_tensor_dicts(tds, device))

    def get_card_embedding(self) -> nn.Embedding:
        return self.card_embedding

    def _encode_choice_effects(self, vecs: torch.Tensor) -> torch.Tensor:
        if vecs.shape[0] == 0:
            return torch.zeros(16, device=vecs.device)
        return F.relu(self.choice_effect_mlp(vecs.mean(dim=0)))
