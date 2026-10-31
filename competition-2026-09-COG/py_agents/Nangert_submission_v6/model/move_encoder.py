from typing import Any, Dict, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import RMSNorm, bucket_length
from ..utils.encode_effects_string import EFFECT_DIM, encode_effects_combined

def collate_move_features(move_features_lists: List[List[Dict[str, Any]]], device: Union[str, torch.device]='cpu') -> Dict[str, torch.Tensor]:
    b = len(move_features_lists)
    m_max = max(1, max((len(fl) for fl in move_features_lists), default=1))
    c_max = 1
    e_max = 1
    for fl in move_features_lists:
        for f in fl:
            ids = f['choice_card_ids']
            if ids:
                c_max = max(c_max, len(ids))
            vecs = f['choice_effect_vecs']
            if vecs is not None and vecs.shape[0] > 0:
                e_max = max(e_max, vecs.shape[0])
    m_max = bucket_length(m_max)
    c_max = bucket_length(c_max)
    e_max = bucket_length(e_max)
    move_type = torch.zeros(b, m_max, dtype=torch.long)
    move_mask = torch.zeros(b, m_max, dtype=torch.bool)
    card_present = torch.zeros(b, m_max, dtype=torch.bool)
    card_id = torch.zeros(b, m_max, dtype=torch.long)
    card_scalars = torch.zeros(b, m_max, 5)
    card_effects = torch.zeros(b, m_max, EFFECT_DIM)
    patron_present = torch.zeros(b, m_max, dtype=torch.bool)
    patron_id = torch.zeros(b, m_max, dtype=torch.long)
    patron_favor = torch.zeros(b, m_max, 1)
    attack_present = torch.zeros(b, m_max, dtype=torch.bool)
    attack_hp = torch.zeros(b, m_max, 1)
    buy_present = torch.zeros(b, m_max, dtype=torch.bool)
    buy_combo = torch.zeros(b, m_max, 1)
    choice_ids = torch.zeros(b, m_max, c_max, dtype=torch.long)
    choice_mask = torch.zeros(b, m_max, c_max, dtype=torch.bool)
    choice_eff_vecs = torch.zeros(b, m_max, e_max, EFFECT_DIM)
    choice_eff_mask = torch.zeros(b, m_max, e_max, dtype=torch.bool)
    for bi, fl in enumerate(move_features_lists):
        for mi, f in enumerate(fl):
            move_type[bi, mi] = f['move_type']
            move_mask[bi, mi] = True
            if f['card_id'] is not None:
                card_present[bi, mi] = True
                card_id[bi, mi] = f['card_id']
                card_scalars[bi, mi] = torch.tensor([float(f['card_cost']), float(f['card_hp']), float(f['card_taunt']), float(f['card_type']), float(f['card_deck'])])
                card_effects[bi, mi] = encode_effects_combined(f['card_effects'] or [])
            if f['patron_id'] is not None:
                patron_present[bi, mi] = True
                patron_id[bi, mi] = f['patron_id']
                patron_favor[bi, mi, 0] = float(f['patron_favor'])
            if f['target_agent_hp'] is not None:
                attack_present[bi, mi] = True
                attack_hp[bi, mi, 0] = float(f['target_agent_hp'])
            if f['combo_count_after_buy'] is not None:
                buy_present[bi, mi] = True
                buy_combo[bi, mi, 0] = float(f['combo_count_after_buy'])
            ids = f['choice_card_ids']
            if ids:
                n = len(ids)
                choice_ids[bi, mi, :n] = torch.tensor(ids, dtype=torch.long)
                choice_mask[bi, mi, :n] = True
            vecs = f['choice_effect_vecs']
            if vecs is not None and vecs.shape[0] > 0:
                n = vecs.shape[0]
                choice_eff_vecs[bi, mi, :n] = vecs if vecs.device.type == 'cpu' else vecs.cpu()
                choice_eff_mask[bi, mi, :n] = True
    batch = {'move_type': move_type, 'move_mask': move_mask, 'card_present': card_present, 'card_id': card_id, 'card_scalars': card_scalars, 'card_effects': card_effects, 'patron_present': patron_present, 'patron_id': patron_id, 'patron_favor': patron_favor, 'attack_present': attack_present, 'attack_hp': attack_hp, 'buy_present': buy_present, 'buy_combo': buy_combo, 'choice_ids': choice_ids, 'choice_mask': choice_mask, 'choice_eff_vecs': choice_eff_vecs, 'choice_eff_mask': choice_eff_mask}
    return {k: v.to(device) for k, v in batch.items()}

def slice_move_batch(batch: Dict[str, torch.Tensor], idx: torch.Tensor, device: Union[str, torch.device]) -> Dict[str, torch.Tensor]:
    move_mask = batch['move_mask'][idx]
    m = min(bucket_length(max(1, int(move_mask.sum(dim=1).max().item()))), move_mask.shape[1])
    out: Dict[str, torch.Tensor] = {}
    for key, val in batch.items():
        out[key] = val[idx, :m]
    c = min(bucket_length(max(1, int(out['choice_mask'].sum(dim=2).max().item()))), out['choice_mask'].shape[2])
    out['choice_ids'] = out['choice_ids'][:, :, :c]
    out['choice_mask'] = out['choice_mask'][:, :, :c]
    e = min(bucket_length(max(1, int(out['choice_eff_mask'].sum(dim=2).max().item()))), out['choice_eff_mask'].shape[2])
    out['choice_eff_vecs'] = out['choice_eff_vecs'][:, :, :e]
    out['choice_eff_mask'] = out['choice_eff_mask'][:, :, :e]
    return {k: v.to(device) for k, v in out.items()}

class MoveEncoder(nn.Module):

    def __init__(self, config, card_embedding: nn.Embedding) -> None:
        super().__init__()
        card_embed_dim = config.card_embed_dim
        self.card_embedding = card_embedding
        self.move_type_embed = nn.Embedding(config.num_move_types, 16)
        self.card_scalar_mlp = nn.Linear(5, 16)
        self.card_effects_mlp = nn.Linear(EFFECT_DIM, 16)
        self.patron_embed = nn.Embedding(config.num_patrons, 16)
        self.patron_favor_mlp = nn.Linear(1, 4)
        self.attack_context_mlp = nn.Linear(1, 4)
        self.buy_context_mlp = nn.Linear(1, 4)
        self.choice_effect_mlp = nn.Linear(EFFECT_DIM, 16)
        self.input_norm = RMSNorm(156)
        self.fusion1 = nn.Linear(156, config.move_embed_dim)
        self.hidden_norm = RMSNorm(config.move_embed_dim)
        self.fusion2 = nn.Linear(config.move_embed_dim, config.move_embed_dim)
        self._card_embed_dim = card_embed_dim

    def forward(self, feats: Dict[str, Any], device: str) -> torch.Tensor:
        move_type_t = torch.tensor([feats['move_type']], dtype=torch.long, device=device)
        move_type_out = self.move_type_embed(move_type_t).squeeze(0)
        if feats['card_id'] is not None:
            card_id_t = torch.tensor([feats['card_id']], dtype=torch.long, device=device)
            card_emb = self.card_embedding(card_id_t).squeeze(0)
            card_scalars = torch.tensor([float(feats['card_cost']), float(feats['card_hp']), float(feats['card_taunt']), float(feats['card_type']), float(feats['card_deck'])], dtype=torch.float32, device=device)
            card_scalar_out = F.relu(self.card_scalar_mlp(card_scalars))
            effects = feats['card_effects'] or []
            card_eff_out = F.relu(self.card_effects_mlp(encode_effects_combined(effects).to(device)))
        else:
            card_emb = torch.zeros(self._card_embed_dim, device=device)
            card_scalar_out = torch.zeros(16, device=device)
            card_eff_out = torch.zeros(16, device=device)
        if feats['patron_id'] is not None:
            patron_t = torch.tensor([feats['patron_id']], dtype=torch.long, device=device)
            patron_emb = self.patron_embed(patron_t).squeeze(0)
            favor_t = torch.tensor([[float(feats['patron_favor'])]], dtype=torch.float32, device=device)
            favor_out = F.relu(self.patron_favor_mlp(favor_t)).squeeze(0)
        else:
            patron_emb = torch.zeros(16, device=device)
            favor_out = torch.zeros(4, device=device)
        if feats['target_agent_hp'] is not None:
            atk_t = torch.tensor([[float(feats['target_agent_hp'])]], dtype=torch.float32, device=device)
            atk_out = F.relu(self.attack_context_mlp(atk_t)).squeeze(0)
        else:
            atk_out = torch.zeros(4, device=device)
        if feats['combo_count_after_buy'] is not None:
            buy_t = torch.tensor([[float(feats['combo_count_after_buy'])]], dtype=torch.float32, device=device)
            buy_out = F.relu(self.buy_context_mlp(buy_t)).squeeze(0)
        else:
            buy_out = torch.zeros(4, device=device)
        if feats['choice_card_ids'] is not None and len(feats['choice_card_ids']) > 0:
            choice_ids_t = torch.tensor(feats['choice_card_ids'], dtype=torch.long, device=device)
            choice_card_out = self.card_embedding(choice_ids_t).mean(dim=0)
        else:
            choice_card_out = torch.zeros(self._card_embed_dim, device=device)
        if feats['choice_effect_vecs'] is not None and feats['choice_effect_vecs'].shape[0] > 0:
            vecs = feats['choice_effect_vecs'].to(device)
            choice_eff_out = F.relu(self.choice_effect_mlp(vecs)).mean(dim=0)
        else:
            choice_eff_out = torch.zeros(16, device=device)
        fused = torch.cat([move_type_out, card_emb, card_scalar_out, card_eff_out, patron_emb, favor_out, atk_out, buy_out, choice_card_out, choice_eff_out])
        h = self.input_norm(fused)
        h = F.gelu(self.fusion1(h))
        return self.fusion2(self.hidden_norm(h))

    def encode_moves(self, move_features_list: List[Dict[str, Any]], device: str) -> torch.Tensor:
        return torch.stack([self.forward(f, device) for f in move_features_list])

    def forward_collated(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        dtype = self.fusion1.weight.dtype
        move_type_out = self.move_type_embed(batch['move_type'])
        card_p = batch['card_present'].unsqueeze(-1).to(dtype)
        card_emb = self.card_embedding(batch['card_id']) * card_p
        card_scalar_out = F.relu(self.card_scalar_mlp(batch['card_scalars'])) * card_p
        card_eff_out = F.relu(self.card_effects_mlp(batch['card_effects'])) * card_p
        patron_p = batch['patron_present'].unsqueeze(-1).to(dtype)
        patron_emb = self.patron_embed(batch['patron_id']) * patron_p
        favor_out = F.relu(self.patron_favor_mlp(batch['patron_favor'])) * patron_p
        atk_p = batch['attack_present'].unsqueeze(-1).to(dtype)
        atk_out = F.relu(self.attack_context_mlp(batch['attack_hp'])) * atk_p
        buy_p = batch['buy_present'].unsqueeze(-1).to(dtype)
        buy_out = F.relu(self.buy_context_mlp(batch['buy_combo'])) * buy_p
        c_mask = batch['choice_mask']
        cc_emb = self.card_embedding(batch['choice_ids'])
        choice_card_out = (cc_emb * c_mask.unsqueeze(-1).to(dtype)).sum(dim=2) / c_mask.sum(dim=2).clamp(min=1).unsqueeze(-1)
        e_mask = batch['choice_eff_mask']
        ce = F.relu(self.choice_effect_mlp(batch['choice_eff_vecs']))
        choice_eff_out = (ce * e_mask.unsqueeze(-1).to(dtype)).sum(dim=2) / e_mask.sum(dim=2).clamp(min=1).unsqueeze(-1)
        fused = torch.cat([move_type_out, card_emb, card_scalar_out, card_eff_out, patron_emb, favor_out, atk_out, buy_out, choice_card_out, choice_eff_out], dim=-1)
        h = self.input_norm(fused)
        h = F.gelu(self.fusion1(h))
        return self.fusion2(self.hidden_norm(h))
