import torch
import torch.nn as nn
from typing import Dict, Tuple, List, Optional

from nn.CardEmbedding import CardEmbedding
from nn.CrossAttentionScorer import CrossAttentionScorer
from nn.EffectsEmbedding import EffectsEmbedding
from nn.PatronEmbedding import PatronEmbedding
from nn.ResidualMLP import ResidualMLP
from nn.TavernSelfAttention import TavernSelfAttention
from state_encoder.move_to_tensor_v3 import MOVE_FEAT_DIM


class BetterNetV17(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_moves: int = 10, num_cards: int = 256, attn_heads: int = 4) -> None:
        super().__init__()

        # Feature dims
        self.move_feat_dim = MOVE_FEAT_DIM
        self.current_player_dim = 8
        self.enemy_player_dim = 6
        self.patron_dim = 10 * 2

        # Encoders
        self.move_encoder = nn.Sequential(
            nn.Linear(self.move_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.cur_player_encoder = nn.Sequential(
            nn.Linear(self.current_player_dim, hidden_dim),
            nn.ReLU(),
            ResidualMLP(hidden_dim, hidden_dim),
        )
        self.enemy_player_encoder = nn.Sequential(
            nn.Linear(self.enemy_player_dim, hidden_dim),
            nn.ReLU(),
            ResidualMLP(hidden_dim, hidden_dim),
        )
        self.patron_encoder = nn.Sequential(
            nn.Linear(self.patron_dim, hidden_dim),
            nn.ReLU(),
            ResidualMLP(hidden_dim, hidden_dim),
        )

        # Shared embeddings
        self.card_embedding = CardEmbedding(num_cards=num_cards, embed_dim=hidden_dim, scalar_feat_dim=3)
        self.patron_embedding = PatronEmbedding(num_patrons=10, embed_dim=hidden_dim)
        self.effects_embedding = EffectsEmbedding(embed_dim=hidden_dim)

        # Attentions
        self.tavern_available_attention = TavernSelfAttention(hidden_dim, hidden_dim)
        self.tavern_cards_attention = TavernSelfAttention(hidden_dim, hidden_dim)

        # Fusion & LSTM
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 9, hidden_dim * 4),
            nn.ReLU(),
            ResidualMLP(hidden_dim * 4, hidden_dim * 4),
        )
        self.lstm = nn.LSTM(
            input_size=hidden_dim * 4,
            hidden_size=256,
            batch_first=True,
        )

        # Keep this around for dims/compatibility with existing code
        self.policy_proj = nn.Linear(256, hidden_dim)

        # Move fusion: gated + bi-interaction
        self.move_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),  # bias initialized below to favor meta early
        )
        self.move_bi = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            ResidualMLP(hidden_dim, hidden_dim),
        )
        self.move_ln = nn.LayerNorm(hidden_dim)

        # Cross-attention scorer
        self.move_cross_attn = CrossAttentionScorer(
            context_dim=256,
            move_dim=hidden_dim,
            attn_dim=hidden_dim,
            num_heads=attn_heads
        )

        # Heads
        self.value_head = nn.Linear(hidden_dim * 2, 1)
        self.num_moves = num_moves

        # --- init: bias gate towards meta path (g ≈ 0) so it’s safe before training the fusion ---
        with torch.no_grad():
            self.move_gate[-1].bias.fill_(-2.0)

    # --------------------
    # Public forward
    # --------------------
    def forward(
            self,
            obs: Dict[str, torch.Tensor],
            move_metas: Optional[List[dict]] = None,
            move_tensor: Optional[torch.Tensor] = None,
            hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ):
        D = self.policy_proj.out_features  # == hidden_dim

        def embed_mean(field: str, B: int, T: int) -> torch.Tensor:
            ids = obs[f"{field}_ids"].view(B * T, -1)
            feats = obs[f"{field}_feats"].view(B * T, -1, 3)
            if ids.size(1) == 0:
                return torch.zeros(B, T, D, device=ids.device)
            embedded = self.card_embedding(ids, feats).mean(dim=1)
            return embedded.view(B, T, -1)

        current_shape = obs["current_player"].shape

        # ---------------- TRAINING PATH (no move inputs) ----------------
        if len(current_shape) == 3:
            cur_encoded = self.cur_player_encoder(obs["current_player"])
            opp_encoded = self.enemy_player_encoder(obs["enemy_player"])
            patron_encoded = self.patron_encoder(obs["patron_tensor"].flatten(start_dim=-2))

            B, T, N = obs["tavern_available_ids"].shape
            tav_avail = self.card_embedding(
                obs["tavern_available_ids"].view(B * T, N),
                obs["tavern_available_feats"].view(B * T, N, -1)
            )
            tav_avail_attn = self.tavern_available_attention(tav_avail).view(B, T, -1)

            hand_enc = embed_mean("hand", B, T)
            draw_enc = embed_mean("played", B, T)
            played_enc = embed_mean("known", B, T)
            opp_cooldown_enc = embed_mean("agents", B, T)
            opp_draw_enc = embed_mean("opp_agents", B, T)

            context = self.fusion(torch.cat([
                cur_encoded,
                opp_encoded,
                patron_encoded,
                tav_avail_attn,
                hand_enc,
                draw_enc,
                played_enc,
                opp_cooldown_enc,
                opp_draw_enc
            ], dim=-1))

            lstm_out, _ = self.lstm(context)
            values = self.value_head(lstm_out).squeeze(-1)
            return lstm_out, values

        # ---------------- INFERENCE PATH (single step with moves) ----------------
        elif move_metas is not None and move_tensor is not None and  len(current_shape) == 2:
            B = 1
            device = obs["current_player"].device

            # Encoders for obs
            cur_encoded = self.cur_player_encoder(obs["current_player"]).unsqueeze(1)
            opp_encoded = self.enemy_player_encoder(obs["enemy_player"]).unsqueeze(1)
            patron_encoded = self.patron_encoder(obs["patron_tensor"].flatten(start_dim=-2).unsqueeze(1))

            tav_avail_attn = self.tavern_available_attention(
                self.card_embedding(obs["tavern_available_ids"], obs["tavern_available_feats"])
            ).view(B, 1, -1)

            hand_enc = embed_mean("hand", B, 1)
            draw_enc = embed_mean("played", B, 1)
            played_enc = embed_mean("known", B, 1)
            opp_cooldown_enc = embed_mean("agents", B, 1)
            opp_draw_enc = embed_mean("opp_agents", B, 1)

            context = self.fusion(torch.cat([
                cur_encoded,
                opp_encoded,
                patron_encoded,
                tav_avail_attn,
                hand_enc,
                draw_enc,
                played_enc,
                opp_cooldown_enc,
                opp_draw_enc
            ], dim=-1))

            lstm_out, new_hidden = self.lstm(context, hidden)
            value = self.value_head(lstm_out).squeeze(-1).squeeze(-1)
            final_hidden = lstm_out[:, -1, :]

            # --- moves: meta branch ---
            meta_emb = torch.stack([
                self._embed_move_meta(m, device).squeeze(0) for m in move_metas
            ], dim=0).unsqueeze(0)

            # --- moves: dense feature branch ---
            feat_emb = self.move_encoder(move_tensor)

            # --- fuse branches ---
            fused_moves = self._fuse_move_embeddings(meta_emb, feat_emb)

            # --- score with cross-attention ---
            logits = self.move_cross_attn(final_hidden, fused_moves)

            return logits, value, new_hidden

        else:
            raise ValueError(
                "Forward received unexpected inputs:\n"
                f"- current_player.shape={current_shape}\n"
                f"- move_metas is {'set' if move_metas is not None else 'None'}\n"
                f"- move_tensor is {'set' if move_tensor is not None else 'None'}"
            )

    # --------------------
    # Helpers
    # --------------------
    def _fuse_move_embeddings(self, meta_emb: torch.Tensor, feat_emb: torch.Tensor) -> torch.Tensor:
        x_cat = torch.cat([meta_emb, feat_emb], dim=-1)
        gate = torch.sigmoid(self.move_gate(x_cat))
        gated = gate * feat_emb + (1.0 - gate) * meta_emb

        bi = torch.cat([meta_emb, feat_emb, meta_emb * feat_emb, torch.abs(meta_emb - feat_emb)], dim=-1)
        bi_out = self.move_bi(bi)

        fused = self.move_ln(gated + bi_out)
        return fused

    def _embed_move_meta(self, meta: dict, device: torch.device) -> torch.Tensor:
        if meta.get("card_id", -1) is not None and meta["card_id"] >= 0:
            card_id = torch.tensor([meta["card_id"]], dtype=torch.long, device=device)
            feats = torch.zeros((1, 3), device=device)
            return self.card_embedding(card_id, feats)  # [1,D]

        if meta.get("patron_id", -1) is not None and meta["patron_id"] >= 0:
            patron_id = torch.tensor([meta["patron_id"]], dtype=torch.long, device=device)
            return self.patron_embedding(patron_id)  # [1,D]

        if meta.get("effect_vec") is not None and isinstance(meta["effect_vec"], torch.Tensor):
            return self.effects_embedding(meta["effect_vec"].to(device).unsqueeze(0))

        # Fallback: zero vector
        return torch.zeros((1, self.policy_proj.out_features), device=device)