import pathlib
import random
import sys
import traceback
from typing import Dict

import numpy as np
import torch
from scripts_of_tribute.base_ai import BaseAI
from scripts_of_tribute.board import CurrentPlayer, GameState
from scripts_of_tribute.enums import MoveEnum, PatronId
from scripts_of_tribute.move import BasicMove

from .models.card_registry import CardRegistry
from .models.move_encoder import MoveEncoder
from .models.state_encoder import StateEncoder
from .models.VeiNet import VeiNet

from ..extensions import safe_play

class Vei(BaseAI):

    def serialize_move(self, mv: BasicMove) -> dict:
        d = {"type": int(mv.command.value)}
        if hasattr(mv, "cardUniqueId"):
            d["cid"] = self.card_registry.uid2cid[mv.cardUniqueId]
        if hasattr(mv, "patronId"):
            d["patron"] = int(mv.patronId.value)
        if hasattr(mv, "cardsUniqueIds"):
            d["cids"] = [self.card_registry.uid2cid[u] for u in mv.cardsUniqueIds]
        if hasattr(mv, "effects"):
            d["eff"]  = mv.effects
        return d
    
    _PATRON_PRIORITY = [
        PatronId.DUKE_OF_CROWS,
        PatronId.HLAALU,
        PatronId.PELIN,
        PatronId.ANSEI,
        PatronId.SAINT_ALESSIA,
        PatronId.RED_EAGLE,
        PatronId.RAJHIN,
        PatronId.ORGNUM,
        PatronId.PSIJIC,
    ]

    def __init__(
            self,
            bot_name: str='',
            weights: str | None = None,
            traj_path: str | None = None,
            tag: float = 0.0
        ):
        super().__init__("Vei")
        self.device = torch.device("cpu")
        self.card_registry = CardRegistry()
        self.weights = "NEWweights.pt"
        self._steps: list[dict] = []
        self._traj_path = traj_path
        self.player_id = None
        self._tag = tag
        self.crashed = False

    def pregame_prepare(self):
        self._steps.clear()
        if hasattr(self, "net"):
            return
        try:
            self.net = VeiNet().to(self.device)
            self.encoder = StateEncoder(self.device)
            self.move_encoder = MoveEncoder(device=self.device).to(self.device)
            if self.weights and pathlib.Path(self.weights).is_file():
                raw = torch.load(self.weights, map_location=self.device)

                # 1) ——— MoveEncoder ———
                me_ckpt = {k[13:]: v for k, v in raw.items()
                        if k.startswith("move_encoder.")}
                self.move_encoder.load_state_dict(me_ckpt, strict=False)

                # 2) ——— VeiNet ———
                net_ckpt = {k.replace("backbone.", ""): v
                            for k, v in raw.items()
                            if not k.startswith("move_encoder.")}
                model_sd = self.net.state_dict()

                compatible = {k: v for k, v in net_ckpt.items()
                            if k in model_sd and v.shape == model_sd[k].shape}

                # print(f"[Load] VeiNet tensors   ok:{len(compatible)} / {len(model_sd)}")
                model_sd.update(compatible)
                self.net.load_state_dict(model_sd)
            self.net.eval()
        except Exception as e:
            print("Exception in Vei.pregame_prepare():", e, file=sys.stderr, flush=True)
            self.crashed = True
            traceback.print_exc()
            return


    def select_patron(self, available_patrons):
        for pid in self._PATRON_PRIORITY:
            if pid in available_patrons:
                return pid
        return random.choice(available_patrons)
    
    @safe_play(fallback="last")
    def play(self, game_state: GameState, possible_moves, remaining_time):
        if self.crashed:
            return possible_moves[-1]
        if self.player_id is None:
            self.player_id = game_state.current_player.player_id
        try:
            feats = self.encoder(game_state)
            with torch.no_grad():
                logits, V = self.compute_forward(game_state, feats, possible_moves)

            probs = logits.softmax(0).cpu().numpy()
            idx   = np.random.choice(len(possible_moves), p=probs)
            mv    = possible_moves[idx]

            return mv

        except Exception as e:
            print("Exception in Vei.play():", e, file=sys.stderr, flush=True)
            self.crashed = True
            traceback.print_exc()
            return possible_moves[-1]
        
    def game_end(self, end_game_state, final_state):
        pass


    def compute_forward(self, game_state, feats, possible_moves):
        move_vecs = self.move_encoder(possible_moves)
        logits, V = self.net.forward_state(feats, move_vecs)
        bias = torch.zeros_like(logits)
        for i, m in enumerate(possible_moves):
            if m.command in [MoveEnum.PLAY_CARD, MoveEnum.BUY_CARD, MoveEnum.ACTIVATE_AGENT]:
                cid = self.card_registry.uid2cid.get(m.cardUniqueId)
                if cid is not None:
                    wwr = self.card_registry.weighted_win_rate.get(cid, 0.5)
                    bias[i] += (wwr - 0.5) * 0.9
        logits = logits + bias

        try:
            good_moves = sum(1 for m in possible_moves if m.command in [
                MoveEnum.PLAY_CARD, MoveEnum.ACTIVATE_AGENT])
            end_idx = next(i for i, m in enumerate(possible_moves) if m.command == MoveEnum.END_TURN)
            if end_idx < logits.size(0) and good_moves > 0:
                logits[end_idx] -= 0.1 * good_moves
        except (StopIteration, IndexError):
            pass

        return logits, V
