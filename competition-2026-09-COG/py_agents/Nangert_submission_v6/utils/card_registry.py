import json
from pathlib import Path
from typing import Dict, Tuple
_CARDS_PATH = Path(__file__).parent / 'cards.json'
with open(_CARDS_PATH, 'r') as f:
    _RAW_CARDS = json.load(f)
CARD_NAME_TO_ID: Dict[str, int] = {card['Name']: card['id'] for card in _RAW_CARDS}
_CARD_DATA: Dict[str, dict] = {card['Name']: card for card in _RAW_CARDS}
NUM_CARDS: int = len(CARD_NAME_TO_ID)

def get_card_id(card_name: str) -> int:
    return CARD_NAME_TO_ID[card_name]

def get_card_features(card_name: str) -> Tuple[int, int, int]:
    card = _CARD_DATA[card_name]
    cost = card['Cost']
    taunt = int(card.get('Taunt', False))
    hp = card['HP']
    return (cost, taunt, hp)
