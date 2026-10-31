import hashlib
import re
from typing import Dict, Iterable, Optional, Tuple
import torch
from .card_registry import get_card_id
from .encode_effects_string import EFFECT_DIM, encode_effects_list
CARD_FEAT_DIM: int = 8
AGENT_FEAT_DIM: int = 11
_NUM_PATRONS: int = 10
_NUM_BOARD_STATES: int = 4
PUBLIC_TOKEN_BUCKETS: int = 2048
_UNIQUE_ID_RE = re.compile('UniqueId:\\s*\\d+', re.IGNORECASE)
_CARD_TYPE_TO_ONEHOT_IDX: Dict[int, int] = {0: 0, 1: 1, 2: 2, 3: 3, 4: 0, 5: 0}

def _to_int(x) -> int:
    return getattr(x, 'value', x)

def stable_public_token(value: object) -> int:
    text = _UNIQUE_ID_RE.sub('UniqueId:<id>', str(value)).strip().upper()
    digest = hashlib.blake2b(text.encode('utf-8'), digest_size=8).digest()
    return 1 + int.from_bytes(digest, 'little') % (PUBLIC_TOKEN_BUCKETS - 1)

def _encode_public_tokens(values: Iterable[object], device: str) -> torch.Tensor:
    return torch.tensor([stable_public_token(value) for value in values], dtype=torch.long, device=device)

def _encode_card_feats(card, device: str) -> torch.Tensor:
    type_oh = [0.0] * 4
    type_oh[_CARD_TYPE_TO_ONEHOT_IDX.get(_to_int(card.type), 0)] = 1.0
    feats = [float(card.cost), float(card.hp), float(card.taunt), *type_oh, float(_to_int(card.deck))]
    return torch.tensor(feats, dtype=torch.float32, device=device)

def _encode_card_zone(cards: list, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if not cards:
        return (torch.zeros(0, dtype=torch.long, device=device), torch.zeros(0, CARD_FEAT_DIM, dtype=torch.float32, device=device))
    ids = torch.tensor([get_card_id(c.name) for c in cards], dtype=torch.long, device=device)
    feats = torch.stack([_encode_card_feats(c, device) for c in cards])
    return (ids, feats)

def _encode_agent_feats(agent, device: str) -> torch.Tensor:
    card_feats = _encode_card_feats(agent.representing_card, device)
    extra = torch.tensor([float(agent.currentHP), float(agent.activated), float(agent.representing_card.hp)], dtype=torch.float32, device=device)
    return torch.cat([card_feats, extra])

def _encode_agent_zone(agents: list, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if not agents:
        return (torch.zeros(0, dtype=torch.long, device=device), torch.zeros(0, AGENT_FEAT_DIM, dtype=torch.float32, device=device))
    ids = torch.tensor([get_card_id(a.representing_card.name) for a in agents], dtype=torch.long, device=device)
    feats = torch.stack([_encode_agent_feats(a, device) for a in agents])
    return (ids, feats)

def _encode_patron_states(patron_states, my_id: int, device: str) -> torch.Tensor:
    result = torch.zeros(_NUM_PATRONS, 3, dtype=torch.float32, device=device)
    for patron_id, favor in patron_states.patrons.items():
        pid = _to_int(patron_id)
        fav = _to_int(favor)
        if not 0 <= pid < _NUM_PATRONS:
            continue
        if fav == my_id:
            col = 0
        elif fav in (0, 1):
            col = 1
        elif fav == 2:
            col = 2
        else:
            continue
        result[pid, col] = 1.0
    return result

def _compute_combo_counts(current_player, device: str) -> torch.Tensor:
    counts = torch.zeros(_NUM_PATRONS, dtype=torch.float32, device=device)
    for card in current_player.played:
        deck_id = _to_int(card.deck)
        if 0 <= deck_id < _NUM_PATRONS:
            counts[deck_id] += 1.0
    for agent in current_player.agents:
        deck_id = _to_int(agent.representing_card.deck)
        if 0 <= deck_id < _NUM_PATRONS:
            counts[deck_id] += 1.0
    return counts

def _encode_choice_context(game_state, device: str) -> Dict[str, torch.Tensor]:
    if game_state.pending_choice is None:
        return {'choice_scalars': torch.zeros(4, dtype=torch.float32, device=device), 'choice_card_ids': torch.zeros(0, dtype=torch.long, device=device), 'choice_card_feats': torch.zeros(0, CARD_FEAT_DIM, dtype=torch.float32, device=device), 'choice_effect_vecs': torch.zeros(0, EFFECT_DIM, dtype=torch.float32, device=device)}
    choice = game_state.pending_choice
    type_oh = [0.0, 0.0]
    choice_type = _to_int(choice.type)
    if 0 <= choice_type <= 1:
        type_oh[choice_type] = 1.0
    scalars = torch.tensor([float(choice.max_choices), float(choice.min_choices), *type_oh], dtype=torch.float32, device=device)
    if choice_type == 0 and hasattr(choice.possible_options, 'possible_cards'):
        card_ids, card_feats = _encode_card_zone(choice.possible_options.possible_cards, device)
    else:
        card_ids = torch.zeros(0, dtype=torch.long, device=device)
        card_feats = torch.zeros(0, CARD_FEAT_DIM, dtype=torch.float32, device=device)
    if choice_type == 1 and hasattr(choice.possible_options, 'possible_effects'):
        effect_vecs = encode_effects_list(choice.possible_options.possible_effects).to(device)
    else:
        effect_vecs = torch.zeros(0, EFFECT_DIM, dtype=torch.float32, device=device)
    return {'choice_scalars': scalars, 'choice_card_ids': card_ids, 'choice_card_feats': card_feats, 'choice_effect_vecs': effect_vecs}

def game_state_to_tensor_dict(game_state, device: str='cpu') -> Dict[str, torch.Tensor]:
    cp = game_state.current_player
    ep = game_state.enemy_player
    current_scalars = torch.tensor([float(cp.coins), float(cp.power), float(cp.prestige), float(cp.patron_calls), float(len(cp.hand)), float(len(cp.played)), float(len(cp.cooldown_pile)), float(len(cp.draw_pile)), float(len(cp.agents)), float(sum((a.currentHP for a in cp.agents))), float(sum((1 for a in cp.agents if a.representing_card.taunt))), float(sum((1 for a in cp.agents if a.activated)))], dtype=torch.float32, device=device)
    enemy_scalars = torch.tensor([float(ep.coins), float(ep.power), float(ep.prestige), float(len(ep.agents)), float(sum((a.currentHP for a in ep.agents))), float(sum((1 for a in ep.agents if a.representing_card.taunt))), float(len(ep.hand_and_draw)), float(len(ep.played)), float(len(ep.cooldown_pile)), float(sum((1 for a in ep.agents if a.activated)))], dtype=torch.float32, device=device)
    board_oh = [0.0] * _NUM_BOARD_STATES
    bs = getattr(game_state.board_state, 'value', game_state.board_state)
    if 0 <= bs < _NUM_BOARD_STATES:
        board_oh[bs] = 1.0
    game_scalars = torch.tensor([*board_oh, float(len(game_state.upcoming_effects)), float(len(game_state.completed_actions))], dtype=torch.float32, device=device)
    patron_states = _encode_patron_states(game_state.patron_states, _to_int(cp.player_id), device)
    hand_ids, hand_feats = _encode_card_zone(cp.hand, device)
    played_ids, played_feats = _encode_card_zone(cp.played, device)
    cooldown_ids, cooldown_feats = _encode_card_zone(cp.cooldown_pile, device)
    known_ids, known_feats = _encode_card_zone(cp.known_upcoming_draws, device)
    tavern_ids, tavern_feats = _encode_card_zone(game_state.tavern_available_cards, device)
    opp_played_ids, opp_played_feats = _encode_card_zone(ep.played, device)
    opp_known_ids, opp_known_feats = _encode_card_zone(ep.cooldown_pile, device)
    agents_ids, agents_feats = _encode_agent_zone(cp.agents, device)
    opp_agents_ids, opp_agents_feats = _encode_agent_zone(ep.agents, device)
    combo_counts = _compute_combo_counts(cp, device)
    choice_dict = _encode_choice_context(game_state, device)
    return {'current_scalars': current_scalars, 'enemy_scalars': enemy_scalars, 'game_scalars': game_scalars, 'patron_states': patron_states, 'hand_ids': hand_ids, 'hand_feats': hand_feats, 'played_ids': played_ids, 'played_feats': played_feats, 'cooldown_ids': cooldown_ids, 'cooldown_feats': cooldown_feats, 'known_ids': known_ids, 'known_feats': known_feats, 'tavern_ids': tavern_ids, 'tavern_feats': tavern_feats, 'opp_played_ids': opp_played_ids, 'opp_played_feats': opp_played_feats, 'opp_known_ids': opp_known_ids, 'opp_known_feats': opp_known_feats, 'agents_ids': agents_ids, 'agents_feats': agents_feats, 'opp_agents_ids': opp_agents_ids, 'opp_agents_feats': opp_agents_feats, 'combo_counts': combo_counts, **choice_dict}

def game_state_to_tensor_dict_v21(game_state, device: str='cpu', completed_actions_since_last: Optional[int]=None) -> Dict[str, torch.Tensor]:
    td = game_state_to_tensor_dict(game_state, device=device)
    cp = game_state.current_player
    ep = game_state.enemy_player
    draw_ids, draw_feats = _encode_card_zone(list(getattr(cp, 'draw_pile', ())), device)
    opp_hand_draw_ids, opp_hand_draw_feats = _encode_card_zone(list(getattr(ep, 'hand_and_draw', ())), device)
    completed = list(getattr(game_state, 'completed_actions', ()))
    cursor = 0 if completed_actions_since_last is None else max(0, min(int(completed_actions_since_last), len(completed)))
    new_actions = completed[cursor:]
    upcoming = list(getattr(game_state, 'upcoming_effects', ()))
    next_turn = list(getattr(game_state, 'start_of_next_turn_effects', ()))
    choice = getattr(game_state, 'pending_choice', None)
    context = getattr(choice, 'context', 'NO_CHOICE') if choice else 'NO_CHOICE'
    follow_up = getattr(choice, 'choice_follow_up', 'NO_FOLLOW_UP') if choice else 'NO_FOLLOW_UP'
    td.update({'draw_ids': draw_ids, 'draw_feats': draw_feats, 'opp_hand_draw_ids': opp_hand_draw_ids, 'opp_hand_draw_feats': opp_hand_draw_feats, 'public_event_ids': _encode_public_tokens(new_actions, device), 'upcoming_effect_ids': _encode_public_tokens(upcoming, device), 'next_turn_effect_ids': _encode_public_tokens(next_turn, device), 'choice_context_id': torch.tensor(stable_public_token(context), dtype=torch.long, device=device), 'choice_followup_id': torch.tensor(stable_public_token(follow_up), dtype=torch.long, device=device), 'public_scalars': torch.tensor([float(_to_int(cp.player_id)), float(len(completed)), float(len(new_actions)), float(len(upcoming)), float(len(next_turn))], dtype=torch.float32, device=device)})
    return td
