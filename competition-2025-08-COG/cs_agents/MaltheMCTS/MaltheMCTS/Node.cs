using Bots;
using ScriptsOfTribute;
using ScriptsOfTribute.Board.Cards;
using ScriptsOfTribute.Serializers;
using SimpleBots.src.MaltheMCTS.Utility.HeuristicScoring;
using System.Linq;

namespace MaltheMCTS;

public class Node
{
    public Dictionary<Move, Edge> MoveToChildNode;
    public int VisitCount = 0;
    public double TotalScore = 0;
    public int GameStateHash { get; private set; }
    public SeededGameState GameState { get; private set; }
    public List<Move> PossibleMoves;

    internal MaltheMCTS Bot;

    private List<UniqueCard> CardsInHandRanked;
    private List<UniqueCard> CardsInCooldownRanked;
    private List<UniqueCard> CardsInTavernRanked;
    private List<UniqueCard> CardsInDrawPileRanked;
    private List<UniqueCard> CardsPlayedRanked;

    /// <summary>
    /// Only used when SimulateMultipleTurns is disabled. It is a copy of this node, but representing the current score/visits of the node if end_turn is played, but without
    /// affecting the state with the card draws that happens on end_turn, since with this feature disabled, we do not want this to be part of our simulations.
    /// </summary>
    private Node? endNode;

    public Node(SeededGameState gameState, List<Move> possibleMoves, MaltheMCTS bot)
    {
        GameState = gameState;
        PossibleMoves = possibleMoves;
        Bot = bot;
        ApplyInstantMoves();
        if (gameState.GameEndState == null)
        {
            FilterMoves();
        }
        MoveToChildNode = new Dictionary<Move, Edge>();
    }

    public virtual void Visit(out double score, HashSet<Node> visitedNodes)
    {

        if (visitedNodes.Contains(this))
        {
            score = Score();
            TotalScore += score;
            VisitCount++;
            return;
        }

        visitedNodes.Add(this);

        var playerId = GameState.CurrentPlayer.PlayerID;

        if (GameState.GameEndState == null)
        {
            if (VisitCount == 0)
            {
                ApplyInstantMoves();
                score = Score();
            }
            else if (PossibleMoves.Count > MoveToChildNode.Count)
            {
                var expandedEdge = Expand();
                expandedEdge.Child.Visit(out score, visitedNodes);
                expandedEdge.VisitCount++;
            }
            else
            {
                var selectedEdge = Select();
                selectedEdge.Child.Visit(out score, visitedNodes);
                selectedEdge.VisitCount++;

                if (selectedEdge.Child.GameState.CurrentPlayer.PlayerID != playerId)
                {
                    score *= -1; // this assumes the score is representing a winrate in a zero-sum-game format
                }
            }
        }
        else
        {
            score = Score();
        }

        TotalScore += score;
        VisitCount++;
    }


    internal Edge Expand()
    {
        foreach (var currMove in PossibleMoves)
        {
            Node newChild = null;

            if (!MoveToChildNode.Keys.Any(m => m.IsIdentical(currMove)))
            {
                if (!Bot.Settings.SIMULATE_MULTIPLE_TURNS && currMove.Command == CommandEnum.END_TURN)
                {
                    newChild = new EndNode(GameState, PossibleMoves, Bot);
                }
                else if ((Bot.Settings.INCLUDE_PLAY_MOVE_CHANCE_NODES && currMove.IsStochastic(GameState))
                    || Bot.Settings.INCLUDE_END_TURN_CHANCE_NODES && currMove.Command == CommandEnum.END_TURN)
                {
                    newChild = new ChanceNode(GameState, this, currMove, Bot);
                }
                else
                {
                    ulong randomSeed = (ulong)Utility.Rng.Next();
                    var (newGameState, newPossibleMoves) = GameState.ApplyMove(currMove, randomSeed);
                    newChild = Utility.FindOrBuildNode(newGameState, this, newPossibleMoves, Bot);
                }

                if (newChild != null &&
                !Bot.Settings.SIMULATE_MULTIPLE_TURNS &&
                newChild.PossibleMoves.Count == 1 &&
                newChild.PossibleMoves[0].Command == CommandEnum.END_TURN)
                {
                    newChild = new EndNode(GameState, PossibleMoves, Bot);
                }

                var newEdge = new Edge(newChild, 0);
                MoveToChildNode.Add(currMove, newEdge);
                return newEdge;
            }
        }

        throw new Exception("Expand was unexpectedly called on a node that was fully expanded");
    }

    internal double Score()
    {
        var gameState = RollOutTillEndOfTurn();
        return HeuristicScoring.Score(gameState, Bot.PredictionEngine);
    }

    private SeededGameState RollOutTillEndOfTurn()
    {
        var rolloutPossibleMoves = PossibleMoves.ToList();
        var gameState = GameState;

        while (gameState.GameEndState != null && (rolloutPossibleMoves.Count > 1 || rolloutPossibleMoves[0].Command != CommandEnum.END_TURN))
        {
            if (Bot.Settings.FORCE_DELAY_TURN_END_IN_ROLLOUT)
            {
                rolloutPossibleMoves.RemoveAll(m => m.Command == CommandEnum.END_TURN);
            }

            var chosenIndex = Utility.Rng.Next(rolloutPossibleMoves.Count);
            var randomMove = rolloutPossibleMoves[chosenIndex];

            if (randomMove.Command == CommandEnum.END_TURN)
            {
                return gameState;
            }
            else
            {
                (gameState, rolloutPossibleMoves) = gameState.ApplyMove(randomMove);
            }
        }

        return gameState;
    }

    internal double Rollout()
    {
        double result = 0;
        var rolloutGameState = GameState;
        var rolloutPlayerId = rolloutGameState.CurrentPlayer.PlayerID;
        var rolloutPossibleMoves = new List<Move>(PossibleMoves);

        // TODO also apply the playing obvious moves in here, possibly
        while (rolloutGameState.GameEndState == null)
        {
            if (Bot.Settings.FORCE_DELAY_TURN_END_IN_ROLLOUT)
            {
                if (rolloutPossibleMoves.Count > 1)
                {
                    rolloutPossibleMoves.RemoveAll(Move => Move.Command == CommandEnum.END_TURN);
                }
            }
            var chosenIndex = Utility.Rng.Next(rolloutPossibleMoves.Count);
            var moveToMake = rolloutPossibleMoves[chosenIndex];

            var (newGameState, newPossibleMoves) = rolloutGameState.ApplyMove(moveToMake);
            rolloutGameState = newGameState;
            rolloutPossibleMoves = Utility.RemoveDuplicateMoves(newPossibleMoves);
        }

        if (rolloutGameState.GameEndState.Winner != PlayerEnum.NO_PLAYER_SELECTED)
        {
            if (rolloutGameState.GameEndState.Winner == rolloutPlayerId)
            {
                result += 1;
            }
            else
            {
                result -= 1;
            }
        }

        return result;
    }

    internal virtual Edge Select()
    {
        double maxConfidence = -double.MaxValue;
        var highestConfidenceChild = MoveToChildNode.First().Value;

        foreach (var childEdge in MoveToChildNode.Values)
        {
            double confidence = GetUCBScore(childEdge);
            if (confidence > maxConfidence)
            {
                maxConfidence = confidence;
                highestConfidenceChild = childEdge;
            }
        }

        return highestConfidenceChild;
    }

    public double GetUCBScore(Edge edge)
    {
        if (Bot.Settings.UPDATED_TREE_REUSE)
        {
            var simulatedTotalScore = (edge.Child.TotalScore / edge.Child.VisitCount) * edge.VisitCount;
            double exploitation = simulatedTotalScore / edge.VisitCount;
            double exploration = Bot.Settings.UCT_EXPLORATION_CONSTANT * Math.Sqrt(Math.Log(VisitCount) / edge.VisitCount);
            return exploitation + exploration;
        }
        else
        {
            double exploitation = edge.Child.TotalScore / edge.Child.VisitCount;
            double exploration = Bot.Settings.UCT_EXPLORATION_CONSTANT * Math.Sqrt(Math.Log(VisitCount) / edge.Child.VisitCount);
            return exploitation + exploration;
        }
    }

    internal void ApplyInstantMoves()
    {
        foreach (var currMove in PossibleMoves)
        {
            if (currMove.IsInstantPlay())
            {
                (GameState, var possibleMoves) = GameState.ApplyMove(currMove, (ulong)Utility.Rng.Next());
                PossibleMoves = possibleMoves;
                FilterMoves();
                ApplyInstantMoves();
                break;
            }
        }

        CardsInHandRanked = null;
        CardsInCooldownRanked = null;
        CardsInTavernRanked = null;
        CardsInDrawPileRanked = null;
        CardsPlayedRanked = null;
        GameStateHash = GameState.GenerateHash();
    }

    private void FilterMoves()
    {
        PossibleMoves = Utility.RemoveDuplicateMoves(PossibleMoves);

        #region additionalFiltering
        if (Bot.Settings.ADDITIONAL_MOVE_FILTERING)
        {
            switch (GameState.BoardState)
            {
                case ScriptsOfTribute.Board.CardAction.BoardState.CHOICE_PENDING:
                    switch (GameState.PendingChoice!.ChoiceFollowUp)
                    {
                        case ChoiceFollowUp.ENACT_CHOSEN_EFFECT:
                        case ChoiceFollowUp.ACQUIRE_CARDS:
                        case ChoiceFollowUp.REFRESH_CARDS:
                        case ChoiceFollowUp.TOSS_CARDS:
                        case ChoiceFollowUp.KNOCKOUT_AGENTS:
                        case ChoiceFollowUp.COMPLETE_HLAALU:
                        case ChoiceFollowUp.COMPLETE_PELLIN:
                        case ChoiceFollowUp.COMPLETE_PSIJIC:
                        case ChoiceFollowUp.REPLACE_CARDS_IN_TAVERN:
                            break;
                        case ChoiceFollowUp.DESTROY_CARDS:
                            var cardsInHandAndPlayed = GameState.CurrentPlayer.Played.Concat(GameState.CurrentPlayer.Hand);
                            SetBewildermentGoldChoiceMoves(cardsInHandAndPlayed);
                            break;
                        case ChoiceFollowUp.DISCARD_CARDS:
                            SetBewildermentGoldChoiceMoves(GameState.CurrentPlayer.Hand);
                            break;
                        case ChoiceFollowUp.COMPLETE_TREASURY:
                            cardsInHandAndPlayed = GameState.CurrentPlayer.Played.Concat(GameState.CurrentPlayer.Hand);
                            SetBewildermentGoldChoiceMoves(cardsInHandAndPlayed);
                            break;
                    }
                    break;
                case ScriptsOfTribute.Board.CardAction.BoardState.NORMAL:
                    // Limit to play all cards before buying from tavern or activating patrons
                    bool canPlayCards = GameState.CurrentPlayer.Hand.Count > 0;
                    if (canPlayCards)
                    {
                        PossibleMoves = PossibleMoves.Where(m => m.Command == CommandEnum.PLAY_CARD).ToList();
                    }
                    break;
                case ScriptsOfTribute.Board.CardAction.BoardState.START_OF_TURN_CHOICE_PENDING:
                    switch (GameState.PendingChoice!.ChoiceFollowUp)
                    {
                        case ChoiceFollowUp.DISCARD_CARDS:
                            var cardsHandAndPlayed = GameState.CurrentPlayer.Played.Concat(GameState.CurrentPlayer.Hand);
                            SetBewildermentGoldChoiceMoves(cardsHandAndPlayed);
                            break;
                        default:
                            throw new NotImplementedException("Unexpected choice follow up: " + GameState.PendingChoice!.ChoiceFollowUp);
                    }
                    break;
                // Complete treasury seems to be a patron choice, so not sure that the complete treasury enum value is for
                case ScriptsOfTribute.Board.CardAction.BoardState.PATRON_CHOICE_PENDING:
                    var InHandAndPlayed = GameState.CurrentPlayer.Played.Concat(GameState.CurrentPlayer.Hand);
                    // Hlaalu cant destroy 0 cost cards like gold or bewilderment
                    if (GameState.PendingChoice?.ChoiceFollowUp == ChoiceFollowUp.COMPLETE_TREASURY)
                    {
                        SetBewildermentGoldChoiceMoves(InHandAndPlayed);
                    }
                    break;
            }
        }
        #endregion

        if (Bot.Settings.CHOICE_BRANCH_LIMIT != null && PossibleMoves.Count > Bot.Settings.CHOICE_BRANCH_LIMIT)
        {

            switch (GameState.BoardState)
            {
                case ScriptsOfTribute.Board.CardAction.BoardState.CHOICE_PENDING:
                    switch (GameState.PendingChoice!.ChoiceFollowUp)
                    {
                        case ChoiceFollowUp.ENACT_CHOSEN_EFFECT:
                            Console.WriteLine("UNEXPECTED BRANCH LIMIT VIOLATION (enact chosen effect)");
                            break;
                        case ChoiceFollowUp.ACQUIRE_CARDS:
                            if (CardsInTavernRanked == null)
                            {
                                CardsInTavernRanked = Utility.RankCardsInGameState(GameState, GameState.TavernAvailableCards);
                            }
                            // Aquire in this patch, always is a maximum of 1 card
                            var maxPrice = PossibleMoves.Max(m =>
                            {
                                var move = (MakeChoiceMoveUniqueCard)m;
                                return move.Choices.Count > 0 ? move.Choices[0].Cost : 0;
                            });
                            var allowedCards = CardsInTavernRanked.Where(c => c.Cost <= maxPrice);
                            var topTavernCards = allowedCards.Take(Bot.Settings.CHOICE_BRANCH_LIMIT!.Value - 1);
                            PossibleMoves = PossibleMoves.Where(m =>
                                (m as MakeChoiceMoveUniqueCard).Choices.Count == 0
                                || topTavernCards.Any(c => (m as MakeChoiceMoveUniqueCard).Choices[0].CommonId == c.CommonId)
                                ).ToList();
                            break;
                        case ChoiceFollowUp.DESTROY_CARDS:
                            if (CardsPlayedRanked == null) // In SoT, the destroy also allows to destroy from hand, but to assist bot, i exclude this, cause its almost always best to play the card first
                            {
                                CardsPlayedRanked = Utility.RankCardsInGameState(GameState, GameState.CurrentPlayer.Played);
                            }
                            int maxAmount = PossibleMoves.Max(m => (m as MakeChoiceMoveUniqueCard).Choices.Count);
                            if (maxAmount == 1)
                            {
                                var worstPlayedPileCards = CardsPlayedRanked.TakeLast(Bot.Settings.CHOICE_BRANCH_LIMIT!.Value - 1);
                                PossibleMoves = PossibleMoves.Where(m =>
                                (m as MakeChoiceMoveUniqueCard).Choices.Count == 0
                                || worstPlayedPileCards.Any(c => (m as MakeChoiceMoveUniqueCard).Choices[0].CommonId == c.CommonId))
                                .ToList();
                            }
                            else // Here the possible destroy amount is 2, since thats the max in the patch
                            {
                                PossibleMoves = Utility.GetRankedCardCombinationMoves(PossibleMoves, CardsPlayedRanked.AsEnumerable().Reverse().ToList(), Bot.Settings.CHOICE_BRANCH_LIMIT!.Value, true);
                            }
                            break;
                        case ChoiceFollowUp.DISCARD_CARDS:
                            // Discard in this patch is always 1 card
                            if (CardsInHandRanked == null)
                            {
                                CardsInHandRanked = Utility.RankCardsInGameState(GameState, GameState.CurrentPlayer.Hand.ToList());
                            }
                            var bottumHandCards = CardsInHandRanked.TakeLast(Bot.Settings.CHOICE_BRANCH_LIMIT!.Value).ToList();
                            PossibleMoves = PossibleMoves.Where(m =>
                                    bottumHandCards.Any(c => (m as MakeChoiceMoveUniqueCard).Choices[0].CommonId == c.CommonId))
                                .ToList();
                            break;
                        case ChoiceFollowUp.REFRESH_CARDS: //Means moving cards from cooldown to top of drawpile
                            //if (CardsInCooldownRanked == null)
                            //{
                            //    CardsInCooldownRanked = Utility.RankCardsInGameState(GameState, GameState.CurrentPlayer.CooldownPile);
                            //}
                            // Skips this. As its bit more complex. Excludes empty moves in SoT (unlike ToT, as far as i can see) refresh requires a minimum of 1
                            // But can also allow more than 2
                            break;
                        case ChoiceFollowUp.TOSS_CARDS:
                        // Not included in this patch
                        case ChoiceFollowUp.KNOCKOUT_AGENTS:
                            // Theoretically it could make sense to leave agents up to stop opponent from playing even stronger agents, but i see this as purely theoretical and
                            // not something that actually happens in games, so to optimize the MCTS-search, i exclude any moves that does not knockout the maximum amount of agents
                            int allowedAmount = PossibleMoves.Max(m => (m as MakeChoiceMoveUniqueCard).Choices.Count);
                            int knockoutCount = Math.Min(GameState.EnemyPlayer.Agents.Count, allowedAmount);
                            PossibleMoves = PossibleMoves.Where(m => (m as MakeChoiceMoveUniqueCard).Choices.Count == knockoutCount).ToList();
                            // FUTURE consider checking branch limit here too (but other method cant be used, since these a serializedAgents not uniqueCard), but branching factor likely wont get
                            // excessive here
                            break;
                        case ChoiceFollowUp.COMPLETE_HLAALU:
                            Console.WriteLine("UNEXPECTED BRANCH LIMIT VIOLATION (COMPLETE_HLAALU)");
                            break;
                        case ChoiceFollowUp.COMPLETE_PELLIN:
                            Console.WriteLine("UNEXPECTED BRANCH LIMIT VIOLATION (COMPLETE_PELLIN)");
                            break;
                        case ChoiceFollowUp.COMPLETE_PSIJIC:
                            Console.WriteLine("UNEXPECTED BRANCH LIMIT VIOLATION (COMPLETE_PSIJIC)");
                            break;
                        case ChoiceFollowUp.COMPLETE_TREASURY:
                            if (CardsPlayedRanked == null) // In SoT, the destroy also allows to destroy from hand, but to assist bot, i exclude this, cause its almost best to play the card first
                            {
                                CardsPlayedRanked = Utility.RankCardsInGameState(GameState, GameState.CurrentPlayer.Played);
                            }
                            // Treasury is just a single destroy
                            var worstPlayedCards = CardsPlayedRanked.TakeLast(Bot.Settings.CHOICE_BRANCH_LIMIT!.Value - 1);
                            PossibleMoves = PossibleMoves.Where(m =>
                            worstPlayedCards.Any(c => (m as MakeChoiceMoveUniqueCard).Choices[0].CommonId == c.CommonId)
                            || (m as MakeChoiceMoveUniqueCard).Choices.Count == 0)
                            .ToList();
                            break;
                        case ChoiceFollowUp.REPLACE_CARDS_IN_TAVERN:
                            // Not sure here how to make some good logic. Instead, just makes some random moves available
                            var indexes = new HashSet<int>();
                            while (indexes.Count < Bot.Settings.CHOICE_BRANCH_LIMIT!.Value)
                            {
                                indexes.Add(Utility.Rng.Next(PossibleMoves.Count));
                            }
                            PossibleMoves = indexes.Select(i => PossibleMoves[i]).ToList();
                            break;
                    }
                    break;
                case ScriptsOfTribute.Board.CardAction.BoardState.NORMAL:
                    // Here i probably do not want to limit
                    break;
                case ScriptsOfTribute.Board.CardAction.BoardState.START_OF_TURN_CHOICE_PENDING:
                    switch (GameState.PendingChoice!.ChoiceFollowUp)
                    {
                        case ChoiceFollowUp.DISCARD_CARDS: // Always only 1 in this patch
                            if (CardsInHandRanked == null)
                            {
                                CardsInHandRanked = Utility.RankCardsInGameState(GameState, GameState.CurrentPlayer.Hand);
                            }
                            var worstCards = CardsInHandRanked.TakeLast(Bot.Settings.CHOICE_BRANCH_LIMIT!.Value);
                            PossibleMoves = PossibleMoves.Where(m =>
                                    worstCards.Any(c => (m as MakeChoiceMoveUniqueCard).Choices[0].CommonId == c.CommonId))
                                .ToList();
                            break;
                        default:
                            Console.WriteLine("UNKNOWN choice type: " + GameState.PendingChoice!.ChoiceFollowUp);
                            break;
                    }
                    break;
                case ScriptsOfTribute.Board.CardAction.BoardState.PATRON_CHOICE_PENDING:
                    var listAtThisPoint = PossibleMoves.ToList();
                    if (CardsPlayedRanked == null)
                    {
                        CardsPlayedRanked = Utility.RankCardsInGameState(GameState, GameState.CurrentPlayer.Played);
                    }
                    // In SoT, the destroy also allows to destroy from hand, but to assist bot, i exclude this, cause its almost always best to play the card first
                    // Atm Patron choice is only a single selection, so thats why i just use index [0] on choices
                    var destroyableCards = PossibleMoves.Where(m => GameState.CurrentPlayer.Played.Any(c => c.CommonId == (m as MakeChoiceMoveUniqueCard).Choices[0].CommonId))
                            .Select(m => (m as MakeChoiceMoveUniqueCard).Choices[0].CommonId).ToList();

                    var bottumDestroyAble = CardsPlayedRanked.Where(c => destroyableCards.Contains(c.CommonId)).Select(c => c.CommonId).ToList();
                    PossibleMoves = PossibleMoves.Where(
                        m =>
                        {
                            return
                            bottumDestroyAble.Contains((m as MakeChoiceMoveUniqueCard).Choices[0].CommonId)
                            ||
                            (m as MakeChoiceMoveUniqueCard).Choices.Count == 0;
                        }
                    ).ToList();
                    if (PossibleMoves.Count == 0)
                    {
                        // This is because for in SoT (unlike ToT) you can use the Hlaalu effect even with no eligible cards in play, but only in hand
                        // And since i only look at played cards, it might be that none of them are eligible. In this edge case, i just do no filtering
                        PossibleMoves = listAtThisPoint;
                    }
                    break;
            }
        }


        PossibleMoves = Utility.RemoveDuplicateMoves(PossibleMoves);
    }

    private void SetBewildermentGoldChoiceMoves(IEnumerable<UniqueCard> cardPool)
    {
        int maxAmount = PossibleMoves.Max(m => (m as MakeChoiceMoveUniqueCard).Choices.Count);
        var bewilderments = cardPool.Count(c => c.CommonId == CardId.BEWILDERMENT);
        if (bewilderments > 0)
        {
            if (bewilderments >= maxAmount)
            {
                PossibleMoves.RemoveAll(m => !(m as MakeChoiceMoveUniqueCard).Choices.All(m => m.CommonId == CardId.BEWILDERMENT));
            }
            else
            {
                PossibleMoves.RemoveAll(m => !(m as MakeChoiceMoveUniqueCard).Choices.Any(m => m.CommonId == CardId.BEWILDERMENT));
            }
        }

        int remainingAmount = maxAmount - bewilderments;

        if (remainingAmount > 0)
        {
            var gold = cardPool.Count(c => c.CommonId == CardId.GOLD);
            if (gold > 0)
            {
                PossibleMoves.RemoveAll(m => !(m as MakeChoiceMoveUniqueCard).Choices.Any(c => c.CommonId == CardId.GOLD));
            }
        }
    }

    public class Edge
    {
        public Node Child;
        public int VisitCount;

        public Edge(Node child, int visitCount)
        {
            Child = child;
            VisitCount = visitCount;
        }
    }
}
