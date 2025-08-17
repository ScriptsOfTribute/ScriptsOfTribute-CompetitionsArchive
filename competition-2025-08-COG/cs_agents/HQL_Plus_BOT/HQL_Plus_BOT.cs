using ScriptsOfTribute;
using ScriptsOfTribute.AI;
using ScriptsOfTribute.Board;
using ScriptsOfTribute.Board.Cards;
using ScriptsOfTribute.Serializers;

namespace Bots;

//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
//// This was added by the competition organizers to solve the problem with agents
//// using this internal `List` extension.
//public static class Extensions
//{
//    public static T PickRandom<T>(this List<T> source, SeededRandom rng)
//    {
//        return source[rng.Next() % source.Count];
//    }
//}
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

public class HQL_Plus_BOT : AI
{
    private HQL_Plus_QL ql = new HQL_Plus_QL();

    private static Random random = new Random();
    private readonly SeededRandom rng = new SeededRandom((ulong)random.Next(1000));

    public override PatronId SelectPatron(List<PatronId> availablePatrons, int round)
    {
        if (availablePatrons.Count == 8 || availablePatrons.Count == 5)
        {
            if (availablePatrons.Contains(PatronId.ORGNUM))
            {
                return PatronId.ORGNUM;
            }
            else if (availablePatrons.Contains(PatronId.DUKE_OF_CROWS))
            {
                return PatronId.DUKE_OF_CROWS;
            }
            else if (availablePatrons.Contains(PatronId.RAJHIN))
            {
                return PatronId.RAJHIN;
            }
            else if (availablePatrons.Contains(PatronId.RED_EAGLE))
            {
                return PatronId.RED_EAGLE;
            }
        }
        else
        {
            if (availablePatrons.Contains(PatronId.ORGNUM))
            {
                return PatronId.ORGNUM;
            }
            //else if (availablePatrons.Contains(PatronId.DUKE_OF_CROWS))
            //{
            //    return PatronId.DUKE_OF_CROWS;
            //}
            else if (availablePatrons.Contains(PatronId.RAJHIN))
            {
                return PatronId.RAJHIN;
            }
            //else if (availablePatrons.Contains(PatronId.RED_EAGLE))
            //{
            //    return PatronId.RED_EAGLE;
            //}
        }
        return availablePatrons.PickRandom(rng);

    }
    public bool HQL_Plus_ShouldTradeCard(UniqueCard card)
    {
        return (card.Type == CardType.STARTER && card.Cost == 0) || card.Type == CardType.CURSE;
    }

    public bool HQL_Plus_ShouldUseTreasury(SeededGameState game_state, List<SimplePatronMove> patron_moves)
    {
        foreach (var move in patron_moves)
        {
            if (move.PatronId == PatronId.TREASURY)
            {
                var cards = game_state.CurrentPlayer.Hand.Concat(game_state.CurrentPlayer.Played);

                UniqueCard result = cards.First();
                foreach (var card in cards)
                {
                    if (HQL_Plus_ShouldTradeCard(card))
                    {
                        return true;
                    }
                }
            }
        }

        return false;
    }

    public UniqueCard HQL_Plus_PickCardForTreasury(SeededGameState game_state)
    {
        var cards = game_state.CurrentPlayer.Hand.Concat(game_state.CurrentPlayer.Played);

        UniqueCard result = cards.First();
        foreach (var card in cards)
        {
            if (HQL_Plus_ShouldTradeCard(card))
            {
                return card;
            }
        }

        return result;
    }

    private int nodes_limit = 15;

    public class HQL_Plus_SearchNode
    {
        public Move root_move;
        public SeededGameState node_sgs;
        public List<Move>? possible_moves;
        public int heuristic_score;

        public HQL_Plus_SearchNode(Move move, SeededGameState sgs, List<Move>? moves, int score)
        {
            root_move = move;
            node_sgs = sgs;
            possible_moves = moves;
            heuristic_score = score;
        }

    }

    private Move? HQL_Plus_FindNoChoiceMove(SeededGameState sgs, List<Move> possible_moves)
    {
        foreach (var move in possible_moves)
        {
            if (move is SimpleCardMove card)
            {
                if (HQL_Plus_NoChoiceMoves.ShouldPlay(card.Card.CommonId))
                {
                    // var (new_state, new_moves) = sgs.ApplyMove(move);
                    // if (new_state.PendingChoice == null)
                    // {
                    //     return move;
                    // }
                    return move;
                }
            }
        }

        return null;
    }

    public void HQL_Plus_PlaySimpleMovesOnNodeUntilChoice(HQL_Plus_SearchNode node)
    {
        if (node.possible_moves is null)
        {
            return;
        }

        var move = HQL_Plus_FindNoChoiceMove(node.node_sgs, node.possible_moves);
        while (move is not null)
        {
            var (new_sgs, new_possible_moves) = node.node_sgs.ApplyMove(move, (ulong)rng.Next());
            node.node_sgs = new_sgs;
            node.possible_moves = new_possible_moves;

            move = HQL_Plus_FindNoChoiceMove(node.node_sgs, node.possible_moves);
        }

        node.heuristic_score = ql.HQL_Plus_Heuristic(node.node_sgs);
    }

    private Move HQL_Plus_BestMoveFromSearch(SeededGameState sgs, List<Move> possible_moves)
    {
        List<HQL_Plus_SearchNode> all_nodes = new List<HQL_Plus_SearchNode>();

        foreach (Move move in possible_moves)
        {
            if (move.Command != CommandEnum.END_TURN)
            {
                var (newGameState, newPossibleMoves) = sgs.ApplyMove(move, (ulong)rng.Next());
                all_nodes.Add(new HQL_Plus_SearchNode(move, newGameState, newPossibleMoves, ql.HQL_Plus_Heuristic(newGameState)));
            }
            else
            {
                all_nodes.Add(new HQL_Plus_SearchNode(move, sgs, null, ql.HQL_Plus_Heuristic(sgs)));
            }
        }

        List<HQL_Plus_SearchNode> end_states_nodes = new List<HQL_Plus_SearchNode>();
        List<HQL_Plus_SearchNode> best_actual_nodes = new List<HQL_Plus_SearchNode>();

        foreach (HQL_Plus_SearchNode node in all_nodes)
        {
            if (node.possible_moves is null)
            {
                end_states_nodes.Add(node);
            }
            else
            {
                if (node.possible_moves is null)
                {
                    end_states_nodes.Add(node);
                }
                else
                {
                    best_actual_nodes.Add(node);
                }
            }
        }

        while (best_actual_nodes.Count > 0)
        {
            all_nodes = new List<HQL_Plus_SearchNode>();

            foreach (HQL_Plus_SearchNode node in best_actual_nodes)
            {
                foreach (Move move in node.possible_moves)
                {
                    if (move.Command != CommandEnum.END_TURN)
                    {
                        var (newGameState, newPossibleMoves) = node.node_sgs.ApplyMove(move, (ulong)rng.Next());
                        all_nodes.Add(new HQL_Plus_SearchNode(node.root_move, newGameState, newPossibleMoves, ql.HQL_Plus_Heuristic(newGameState)));
                    }
                    else
                    {
                        all_nodes.Add(new HQL_Plus_SearchNode(node.root_move, node.node_sgs, null, ql.HQL_Plus_Heuristic(node.node_sgs)));
                    }
                }
            }

            best_actual_nodes = new List<HQL_Plus_SearchNode>();

            foreach (HQL_Plus_SearchNode node in all_nodes)
            {
                if (node.possible_moves is null)
                {
                    end_states_nodes.Add(node);
                    continue;
                }

                if (best_actual_nodes.Count < nodes_limit)
                {
                    HQL_Plus_PlaySimpleMovesOnNodeUntilChoice(node);
                    best_actual_nodes.Add(node);
                }
                else
                {
                    int min_score = best_actual_nodes[0].heuristic_score;
                    int min_index = 0;

                    for (int i = 0; i < nodes_limit; i++)
                    {
                        if (min_score > best_actual_nodes[i].heuristic_score)
                        {
                            min_score = best_actual_nodes[i].heuristic_score;
                            min_index = i;
                        }
                    }

                    if (min_score < node.heuristic_score)
                    {
                        best_actual_nodes[min_index] = node;
                    }
                }
            }
        }

        Move best_move = end_states_nodes[0].root_move;
        int best_score = end_states_nodes[0].heuristic_score;

        foreach (HQL_Plus_SearchNode node in end_states_nodes)
        {
            if (best_score < node.heuristic_score)
            {
                best_score = node.heuristic_score;
                best_move = node.root_move;
            }
        }

        return best_move;
    }

    public void HQL_Plus_HandleEndTurn(SeededGameState sgs)
    {
        ql.HQL_Plus_IncrementTurnCounter();
        ql.HQL_Plus_SaveGainedCards(sgs);
        ql.HQL_Plus_UpdateQValuesForPlayedCardsAtEndOfTurn(sgs);
    }

    public void HQL_Plus_HandleEndPlay(SeededGameState sgs, Move best_move)
    {
        ql.HQL_Plus_SavePlayedCardIfApplicable(best_move);
        ql.HQL_Plus_UpdateDeckCardsCounter(sgs);
    }

    public override Move Play(GameState game_state, List<Move> possibleMoves, TimeSpan remainingTime)
    {
        SeededGameState sgs = game_state.ToSeededGameState((ulong)rng.Next());

        HQL_Plus_Stage stage = ql.HQL_Plus_TransformGameStateToStages(sgs);

        if (possibleMoves.Count == 1 && possibleMoves[0].Command == CommandEnum.END_TURN)
        {
            HQL_Plus_HandleEndPlay(sgs, possibleMoves[0]);
            HQL_Plus_HandleEndTurn(sgs);
            return Move.EndTurn();
        }

        var action_agent_moves = possibleMoves.Where(m => m.Command == CommandEnum.PLAY_CARD ||
            m.Command == CommandEnum.ACTIVATE_AGENT).ToList();

        var buy_moves = possibleMoves.Where(m => m.Command == CommandEnum.BUY_CARD).ToList();
        var patron_moves = possibleMoves.Where(m => m.Command == CommandEnum.CALL_PATRON).ToList().ConvertAll(m => (SimplePatronMove)m);

        Move best_move = possibleMoves[0];

        if (action_agent_moves.Count != 0)
        {
            best_move = action_agent_moves[0];
            var no_choice_move = HQL_Plus_FindNoChoiceMove(sgs, action_agent_moves);
            if (no_choice_move is not null)
            {
                best_move = no_choice_move;
            }
        }
        // else if (buy_moves.Count != 0)
        // {
        //     // best_move = ql.PickBuyMove(sgs, buy_moves);
        // }
        else
        {
            for (int i = 0; i < buy_moves.Count; i++)
            {
                if (buy_moves[i] is SimpleCardMove buy_card)
                {
                    if (buy_card.Card.Type is CardType.CONTRACT_ACTION)
                    {
                        var (new_state, new_moves) = sgs.ApplyMove(buy_moves[i]);

                        var old_score = ql.HQL_Plus_Heuristic(sgs);
                        var new_score = ql.HQL_Plus_Heuristic(new_state);

                        if (old_score > new_score)
                        {
                            possibleMoves.RemoveAll(m => m == buy_moves[i]);
                        }
                    }
                }
            }

            for (int i = possibleMoves.Count - 1; i >= 0; i--)
            {
                if (possibleMoves[i] is SimplePatronMove patron)
                {
                    if (patron.PatronId == PatronId.DUKE_OF_CROWS && stage != HQL_Plus_Stage.Late && sgs.CurrentPlayer.Coins < 7)
                    {
                        possibleMoves.RemoveAt(i);
                        break;
                    }
                }
            }
            // for (int i = possibleMoves.Count - 1; i >= 0; i--)
            // {
            //     if (possibleMoves[i] is SimplePatronMove patron)
            //     {
            //         if (patron.PatronId == PatronId.ORGNUM && (stage != Stage.Late || stage != Stage.Middle))
            //         {
            //             possibleMoves.RemoveAt(i);
            //             break;
            //         }
            //     }
            // }



            // START OF NEW ENSEMBLE LOGIC
            const int numberOfSearches = 7;

            var moveVotes = new Dictionary<Move, int>();

            for (int i = 0; i < numberOfSearches; i++)
            {
                Move move = HQL_Plus_BestMoveFromSearch(sgs, possibleMoves);

                if (moveVotes.ContainsKey(move))
                {
                    moveVotes[move]++;
                }
                else
                {
                    moveVotes[move] = 1;
                }

            }

            // Find the move with the most votes.
            if (moveVotes.Count > 0)
            {
                best_move = moveVotes.OrderByDescending(kvp => kvp.Value).First().Key;
            }
            else
            {
                best_move = possibleMoves.FirstOrDefault() ?? Move.EndTurn();
            }
        }

        HQL_Plus_HandleEndPlay(sgs, best_move);
        if (best_move.Command == CommandEnum.END_TURN)
        {
            HQL_Plus_HandleEndTurn(sgs);
        }

        return best_move;
    }

    public override void GameEnd(EndGameState state, FullGameState? final_board_state)
    {
        ql.HQL_Plus_ResetVariables();

        ql.HQL_Plus_SaveQTableToFile();
    }
}