import re
from typing import List
import torch
EFFECT_TYPES: List[str] = ['Coin', 'Power', 'Prestige', 'OppLosePrestige', 'Remove', 'Acquire', 'Destroy', 'Draw', 'Discard', 'Return', 'Toss', 'KnockOut', 'Patron', 'CreateSummersetSacking', 'Heal', 'KnockOutAll', 'ReturnAgent', 'Donate']
_EFFECT_TO_IDX = {name: i for i, name in enumerate(EFFECT_TYPES)}
EFFECT_DIM: int = len(EFFECT_TYPES) + 1
_EFFECT_PATTERN = re.compile('^(\\w+)\\s+(\\d+)$')

def encode_effect(effect_str: str) -> torch.Tensor:
    result = torch.zeros(EFFECT_DIM)
    if not effect_str:
        return result
    parts = re.split('\\s+AND\\s+|\\s+OR\\s+', effect_str)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        match = _EFFECT_PATTERN.match(part)
        if match:
            type_name = match.group(1)
            amount = int(match.group(2))
            idx = _EFFECT_TO_IDX.get(type_name)
            if idx is not None:
                result[idx] += 1.0
                result[-1] += float(amount)
    return result

def encode_effects_list(effects: List[str]) -> torch.Tensor:
    if not effects:
        return torch.zeros(0, EFFECT_DIM)
    return torch.stack([encode_effect(e) for e in effects])

def encode_effects_combined(effects: List[str]) -> torch.Tensor:
    if not effects:
        return torch.zeros(EFFECT_DIM)
    return encode_effects_list(effects).sum(dim=0)
