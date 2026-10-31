# PatronZero

**Author:** Martin Niederl, Johannes Kepler University Linz

PatronZero uses a compact policy/value network to evaluate the visible game state and currently legal moves. The state representation covers resources, prestige, patrons, card zones, and agents; each move is represented by its action type and associated card, patron, or choice. The model was initialized through imitation learning and subsequently developed with AlphaZero-inspired self-play and mixed-opponent training.

At decision time, the network can either rank the legal moves on its own or guide one of two current-turn search methods designed for imperfect information. The PC-PIMC variant searches several sampled hidden states independently and combines their recommendations, while the single-observer ISMCTS variant shares one information-set tree across those samples and accounts for actions that are not available in every determinization. Keeping these alternatives in the same agent makes it possible to compare inexpensive neural action selection with information-set search under the same learned model.

The available self-play training time before the submission deadline was limited, so the current model represents an early stage of the approach rather than its full potential.
