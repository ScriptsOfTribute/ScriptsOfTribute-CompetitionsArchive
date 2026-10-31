from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
from scripts_of_tribute.base_ai import BaseAI
from .model.policy_v21 import PolicyNetworkV21
from .model_config import SubmissionModelConfig
from .utils.game_state_to_vector import game_state_to_tensor_dict_v21
from .utils.move_to_features import batch_move_features
WEIGHTS_PATH = Path(__file__).resolve().parent / 'policy_weights.pt'
PATRON_PREFERENCE: tuple = ('RAJHIN', 'PELIN', 'ORGNUM', 'ANSEI', 'SAINT_ALESSIA', 'RED_EAGLE', 'PSIJIC', 'HLAALU', 'DUKE_OF_CROWS')
BOT_NAME = 'NAgent_4'

class CompetitionBotV6(BaseAI):

    def __init__(self, bot_name: str=BOT_NAME, device: str='cpu') -> None:
        super().__init__(bot_name=bot_name)
        blob = torch.load(WEIGHTS_PATH, map_location='cpu', weights_only=True)
        self.device = device
        self.policy = PolicyNetworkV21(SubmissionModelConfig(**blob['model_config']))
        self.policy.load_state_dict(blob['policy_state_dict'])
        self.policy.to(torch.device(device))
        self.policy.eval()
        self._hidden: Optional[Any] = None
        self._previous_action: Optional[Dict[str, Any]] = None
        self._completed_actions: Tuple[str, ...] = ()

    def pregame_prepare(self) -> None:
        self._hidden = None
        self._previous_action = None
        self._completed_actions = ()

    def select_patron(self, available_patrons: List[Any]) -> Any:
        by_name = {getattr(patron, 'name', str(patron)): patron for patron in available_patrons}
        for name in PATRON_PREFERENCE:
            if name in by_name:
                return by_name[name]
        return available_patrons[0]

    def play(self, game_state: Any, possible_moves: List[Any], remaining_time: int=10000) -> Any:
        try:
            completed = tuple((str(action) for action in getattr(game_state, 'completed_actions', ())))
            seen = len(self._completed_actions)
            cursor = seen if completed[:seen] == self._completed_actions else 0
            tensor_dict = game_state_to_tensor_dict_v21(game_state, device=self.device, completed_actions_since_last=cursor)
            move_features = batch_move_features(possible_moves, game_state)
            if self._hidden is None:
                self._hidden = self.policy.initial_hidden()
            with torch.no_grad():
                action_idx, _, _, hidden = self.policy.select_action(tensor_dict, move_features, self._hidden, deterministic=True, previous_action=self._previous_action)
            self._hidden = hidden
            self._completed_actions = completed
            self._previous_action = move_features[action_idx]
            return possible_moves[action_idx]
        except Exception as exc:
            print(f'[CompetitionBotV6] move selection failed ({exc!r}); falling back to a safe legal move', flush=True)
            return _fallback_move(possible_moves)

    def game_end(self, end_game_state: Any, final_state: Any) -> None:
        pass

def _fallback_move(possible_moves: List[Any]) -> Any:
    for move in possible_moves:
        command = getattr(move, 'command', None)
        if getattr(command, 'name', str(command)) == 'END_TURN':
            return move
    return possible_moves[-1]
