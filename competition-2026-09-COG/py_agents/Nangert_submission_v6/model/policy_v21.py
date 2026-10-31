from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.distributions import Categorical
from .move_encoder import MoveEncoder, collate_move_features, slice_move_batch
from .recurrent_policy import Hidden, RecurrentPolicyNetwork
from .state_encoder_v21 import StateEncoderV21, collate_tensor_dicts_v21, slice_state_batch_v21

def _empty_move_features() -> Dict[str, Any]:
    return {'move_type': 6, 'card_id': None, 'card_cost': None, 'card_type': None, 'card_deck': None, 'card_hp': None, 'card_taunt': None, 'card_effects': None, 'target_agent_hp': None, 'combo_count_after_buy': None, 'patron_id': None, 'patron_favor': None, 'choice_card_ids': None, 'choice_card_feats': None, 'choice_effect_vecs': None}

class PolicyNetworkV21(RecurrentPolicyNetwork):
    architecture = 'v2_1'
    uses_previous_action = True

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.state_encoder = StateEncoderV21(config)
        self.move_encoder = MoveEncoder(config, self.state_encoder.get_card_embedding())
        cell_cls = nn.GRU if self._cell_type == 'gru' else nn.LSTM
        self.temporal = cell_cls(input_size=config.state_embed_dim + config.move_embed_dim, hidden_size=config.temporal_hidden_dim, num_layers=config.temporal_layers)
        nn.init.normal_(self.temporal_fusion[-1].weight, std=0.001)
        nn.init.zeros_(self.temporal_fusion[-1].bias)

    @staticmethod
    def collate_states(tensor_dicts, device='cpu'):
        return collate_tensor_dicts_v21(tensor_dicts, device)

    @staticmethod
    def slice_states(batch, idx, device):
        return slice_state_batch_v21(batch, idx, device)

    def collate_previous_actions(self, actions: List[Optional[Dict[str, Any]]], device='cpu') -> Dict[str, Any]:
        dummy = _empty_move_features()
        batch = collate_move_features([[action if action is not None else dummy] for action in actions], device)
        batch['previous_action_present'] = torch.tensor([action is not None for action in actions], dtype=torch.bool, device=device)
        return batch

    def slice_previous_actions(self, batch, idx, device):
        present = batch['previous_action_present']
        moves = {key: value for key, value in batch.items() if key != 'previous_action_present'}
        out = slice_move_batch(moves, idx, device)
        out['previous_action_present'] = present[idx].to(device)
        return out

    def encode_recurrent_inputs(self, state_batch: Dict[str, torch.Tensor], previous_action_batch: Optional[Dict[str, torch.Tensor]]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        state_emb = self.state_encoder.forward_collated(state_batch)
        if previous_action_batch is None:
            previous_emb = torch.zeros(state_emb.shape[0], self.move_encoder.fusion2.out_features, dtype=state_emb.dtype, device=state_emb.device)
        else:
            previous_emb = self.move_encoder.forward_collated(previous_action_batch)[:, 0]
            present = previous_action_batch['previous_action_present']
            previous_emb = previous_emb * present.unsqueeze(1).to(previous_emb.dtype)
        return (state_emb, torch.cat([state_emb, previous_emb], dim=-1))

    def _single_recurrent_input(self, state_emb: torch.Tensor, previous_action: Optional[Dict[str, Any]]) -> torch.Tensor:
        if previous_action is None:
            previous_emb = torch.zeros(self.move_encoder.fusion2.out_features, dtype=state_emb.dtype, device=state_emb.device)
        else:
            previous_emb = self.move_encoder.forward(previous_action, str(self.device))
        return torch.cat([state_emb, previous_emb])

    def forward(self, tensor_dict: Dict[str, torch.Tensor], move_features_list: List[Dict[str, Any]], hidden: Hidden, previous_action: Optional[Dict[str, Any]]=None) -> Tuple[torch.Tensor, torch.Tensor, Hidden]:
        state_emb = self.state_encoder(tensor_dict)
        recurrent_input = self._single_recurrent_input(state_emb, previous_action)
        output, new_hidden = self.temporal(recurrent_input.view(1, 1, -1), hidden)
        ctx = self.fuse(state_emb, output[0, 0])
        move_embs = self.move_encoder.encode_moves(move_features_list, str(self.device))
        logits = self.scorer(ctx, move_embs) / self.temperature
        return (logits, ctx, new_hidden)

    def select_action(self, tensor_dict: Dict[str, torch.Tensor], move_features_list: List[Dict[str, Any]], hidden: Hidden, deterministic: bool=False, previous_action: Optional[Dict[str, Any]]=None) -> Tuple[int, torch.Tensor, torch.Tensor, Hidden]:
        logits, _, new_hidden = self.forward(tensor_dict, move_features_list, hidden, previous_action=previous_action)
        dist = Categorical(logits=logits)
        action = logits.argmax() if deterministic else dist.sample()
        return (action.item(), dist.log_prob(action), dist.entropy(), new_hidden)
