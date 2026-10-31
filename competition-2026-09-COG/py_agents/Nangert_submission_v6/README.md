# Nangert_submission_v6

- **Agent name:** `NAgent_4`
- **Entry point:** `Nangert_submission_v6.competition_bot.CompetitionBotV6`
- **Author:** Nangert
- **Dependencies:** `requirements.txt` (torch, scripts-of-tribute)

A Set-Transformer policy network trained by self-play PPO (~100k games).
No search: one forward pass scores all legal moves jointly with the game
state and the argmax is played; a GRU carries public-observation context
across each game (reset per game). CPU-only, ~10 ms per decision,
~240 MB process peak.
