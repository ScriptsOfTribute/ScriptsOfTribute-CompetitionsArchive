import random
import logging
import sys
import os
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

import numpy as np
import torch
from pathlib import Path
from typing import List

from scripts_of_tribute.base_ai import BaseAI, PatronId, GameState, BasicMove
from scripts_of_tribute.board import EndGameState

from nn.BetterNet_v17 import BetterNetV17
from state_encoder.game_state_to_vector_v5 import game_state_to_tensor_dict_v5
from state_encoder.move_to_metadata import move_to_metadata
from state_encoder.move_to_tensor_v3 import move_to_tensor_v3, MOVE_FEAT_DIM

# tournament addition
from ..extensions import safe_play

PREFERRED_ORDER = [
    PatronId.ORGNUM,
    PatronId.SAINT_ALESSIA,
    PatronId.ANSEI,
    PatronId.RAJHIN,
    PatronId.PELIN,
    PatronId.DUKE_OF_CROWS
]

class CompetitionBotV4(BaseAI):
    def __init__(
        self,
        bot_name: str = "NAgent_2",
        evaluate: bool = True,
        instant_moves: bool = True,
        remove_end_turn: bool = True,
    ):
        super().__init__(bot_name=bot_name)
        self.logger = logging.getLogger(self.__class__.__name__)
        PKG_DIR = Path(__file__).resolve().parent  # py_agents/Nangert_submission_v4
        self.model_path = PKG_DIR / "model" / "v17" / "tribute_net_v26.pt"
        self.instant_moves = instant_moves
        self.remove_end_turn = remove_end_turn

        model = BetterNetV17(hidden_dim=128, num_moves=10, num_cards=125)
        if self.model_path.exists():
            self._load_state(model, self.model_path, self.model_path.name)
        else:
            self.logger.warning("Primary model not found; using random initialization.")

        self.model = model
        self.model.eval()

        self.evaluate = evaluate
        self.hidden = None
        self.order = None

    def _load_state(
        self, model: torch.nn.Module, path: Path, name: str
    ) -> bool:
        if path.exists():
            state = torch.load(path, map_location="cpu")
            model.load_state_dict(state)
            self.logger.info("Loaded %s model from %s", name, path)
            return True
        self.logger.warning(
            "No %s model found at %s; using random initialization.", name, path
        )
        return False

    def pregame_prepare(self) -> None:
        pass

    def select_patron(self, available_patrons: List[PatronId]) -> PatronId:
        for pref in PREFERRED_ORDER:
            if pref in available_patrons:
                return pref

        return random.choice(available_patrons)

    @safe_play(fallback="last")
    def play(
        self,
        game_state: GameState,
        possible_moves: List[BasicMove],
        remaining_time: int,
    ) -> BasicMove:
        obs = game_state_to_tensor_dict_v5(game_state)
        obs = {k: v.unsqueeze(0) for k, v in obs.items()}

        move_metadata = [move_to_metadata(m, game_state) for m in possible_moves]
        move_tensors = [move_to_tensor_v3(m, game_state) for m in possible_moves]

        max_moves = 10
        move_dim = MOVE_FEAT_DIM

        if len(move_metadata) >= max_moves:
            move_metas_padded = move_metadata[:max_moves]
        else:
            pad_meta = {
                "move_type": None,
                "card_id": None,
                "patron_id": None,
                "effect_vec": None
            }
            move_metas_padded = move_metadata + [pad_meta] * (max_moves - len(move_metadata))

        if len(move_tensors) >= max_moves:
            batch = move_tensors[:max_moves]
        else:
            padding = [torch.zeros(move_dim) for _ in range(max_moves - len(move_tensors))]
            batch = move_tensors + padding

        move_batch = torch.stack(batch, dim=0).unsqueeze(0)

        with torch.no_grad():
            logits, value, new_hidden = self.model(obs, move_metas_padded, move_batch, self.hidden)

        self.hidden = new_hidden

        probs = torch.softmax(logits, dim=-1).cpu().numpy().flatten()
        probs = probs[:len(possible_moves)]
        total = probs.sum()

        if total > 0:
            probs /= total
            if self.evaluate:
                idx = int(probs.argmax())
            else:
                idx = int(np.random.choice(len(probs), p=probs))
        else:
            idx = 0

        chosen_move = possible_moves[idx]
        return chosen_move

    def game_end(self, end_game_state: EndGameState, final_state: GameState) -> None:
        pass
