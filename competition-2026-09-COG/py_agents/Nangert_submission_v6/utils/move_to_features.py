from typing import Dict, List, Tuple, Any
from .card_registry import get_card_id
from .encode_effects_string import encode_effects_list
PLAY_CARD = 0
ACTIVATE_AGENT = 1
ATTACK = 2
BUY_CARD = 3
CALL_PATRON = 4
MAKE_CHOICE = 5
END_TURN = 6

def _to_int(x) -> int:
    return getattr(x, 'value', x)

def _find_card_by_unique_id(game_state: Any, unique_id: int) -> Any:
    cp = game_state.current_player
    ep = game_state.enemy_player
    for card in cp.hand:
        if card.unique_id == unique_id:
            return card
    for card in cp.played:
        if card.unique_id == unique_id:
            return card
    for card in cp.cooldown_pile:
        if card.unique_id == unique_id:
            return card
    for card in cp.known_upcoming_draws:
        if card.unique_id == unique_id:
            return card
    for agent in cp.agents:
        if agent.representing_card.unique_id == unique_id:
            return agent.representing_card
    for agent in ep.agents:
        if agent.representing_card.unique_id == unique_id:
            return agent.representing_card
    for card in ep.played:
        if card.unique_id == unique_id:
            return card
    for card in ep.cooldown_pile:
        if card.unique_id == unique_id:
            return card
    for card in game_state.tavern_cards:
        if card.unique_id == unique_id:
            return card
    if hasattr(game_state, 'tavern_available_cards'):
        for card in game_state.tavern_available_cards:
            if card.unique_id == unique_id:
                return card
    return None

def _find_agent_by_card_unique_id(game_state: Any, unique_id: int) -> Any:
    for agent in game_state.enemy_player.agents:
        if agent.representing_card.unique_id == unique_id:
            return agent
    for agent in game_state.current_player.agents:
        if agent.representing_card.unique_id == unique_id:
            return agent
    return None

def _count_deck_cards(current_player: Any, deck_id: int) -> int:
    count = 0
    did = _to_int(deck_id)
    for card in current_player.played:
        if _to_int(card.deck) == did:
            count += 1
    for agent in current_player.agents:
        if _to_int(agent.representing_card.deck) == did:
            count += 1
    return count

def move_to_features(move: Any, game_state: Any) -> Dict[str, Any]:
    cmd = _to_int(move.command)
    result: Dict[str, Any] = {'move_type': cmd, 'card_id': None, 'card_cost': None, 'card_type': None, 'card_deck': None, 'card_hp': None, 'card_taunt': None, 'card_effects': None, 'target_agent_hp': None, 'combo_count_after_buy': None, 'patron_id': None, 'patron_favor': None, 'choice_card_ids': None, 'choice_card_feats': None, 'choice_effect_vecs': None}
    if hasattr(move, 'cardUniqueId'):
        card = _find_card_by_unique_id(game_state, move.cardUniqueId)
        if card is not None:
            result['card_id'] = get_card_id(card.name)
            result['card_cost'] = card.cost
            result['card_type'] = _to_int(card.type)
            result['card_deck'] = _to_int(card.deck)
            result['card_hp'] = card.hp
            result['card_taunt'] = int(card.taunt)
            result['card_effects'] = card.effects if card.effects else None
        if cmd == ATTACK:
            agent = _find_agent_by_card_unique_id(game_state, move.cardUniqueId)
            if agent is not None:
                result['target_agent_hp'] = agent.currentHP
        if cmd == BUY_CARD and card is not None:
            result['combo_count_after_buy'] = _count_deck_cards(game_state.current_player, card.deck) + 1
    elif hasattr(move, 'patronId'):
        result['patron_id'] = _to_int(move.patronId)
        favor_val = game_state.patron_states.patrons.get(move.patronId, 2)
        fav_int = _to_int(favor_val)
        my_id = _to_int(game_state.current_player.player_id)
        if fav_int == my_id:
            result['patron_favor'] = 1
        elif fav_int == 2:
            result['patron_favor'] = 0
        else:
            result['patron_favor'] = -1
    elif hasattr(move, 'cardsUniqueIds'):
        card_ids: List[int] = []
        card_feats: List[Tuple[int, int, int, int, int]] = []
        for uid in move.cardsUniqueIds:
            card = _find_card_by_unique_id(game_state, uid)
            if card is not None:
                card_ids.append(get_card_id(card.name))
                card_feats.append((card.cost, _to_int(card.type), _to_int(card.deck), card.hp, int(card.taunt)))
        result['choice_card_ids'] = card_ids
        result['choice_card_feats'] = card_feats
    elif hasattr(move, 'effects'):
        result['choice_effect_vecs'] = encode_effects_list(move.effects)
    return result

def batch_move_features(moves: List[Any], game_state: Any) -> List[Dict[str, Any]]:
    return [move_to_features(m, game_state) for m in moves]
