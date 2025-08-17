using ScriptsOfTribute;
using ScriptsOfTribute.Board;
using ScriptsOfTribute.Board.Cards;
using ScriptsOfTribute.Serializers;
using System.Diagnostics;

namespace Bots;

using QKey = Tuple<CardId, int>;

public class HQL_Plus_PatronsCount
{
    public int player_patrons = 0;
    public int neutral_patron = 0;
    public int enemy_patrons = 0;

    public HQL_Plus_PatronsCount(int p, int n, int e)
    {
        player_patrons = p;
        neutral_patron = n;
        enemy_patrons = e;
    }
}

public class HQL_Plus_QL
{
    private int turn_counter = 0;
    private static double explorationChance = 0.5;
    private static double learningRate = 0.1;
    private static double discountFactor = 0.5;

    // action - card to buy, state = stages and combo
    // key = {0 - card id, 1 - enemy prestige + turns finished, 3 - combo for card's deck}

    // action - played action or activated agent from specific deck
    // state - number of cards from specific deck
    // value - reward for playing that move
    private Dictionary<QKey, double> qTable1 = new Dictionary<QKey, double>();
    private Dictionary<QKey, double> qTable2 = new Dictionary<QKey, double>();
    private Random doubleQLearningRandom = new Random();

    public int[] deck_cards_counters = new int[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

    public int actions_before_this_turn_count = 0;
    public HashSet<UniqueCard> played_cards_this_turn = new HashSet<UniqueCard>();
    public SortedSet<CardId> gained_cards = new SortedSet<CardId>();

    public HQL_Plus_QL()
    {
        //using (var sw = new StreamWriter(HQL_Plus_FilePaths.errorFile, append: true))
        //{
        //    // sw.WriteLine("Start of game");
        //    // sw.WriteLine(DateTime.Now + "\n\n");
        //}

        //using (var sw = new StreamWriter(HQL_Plus_FilePaths.tmpFile, append: false))
        //{
        //    // sw.WriteLine("Start of game");
        //    // sw.WriteLine(DateTime.Now + "\n");
        //}

        HQL_Plus_ReadQTableFromFile();
    }

    public void HQL_Plus_ReadQTableFromFile()
    {
        using (var streamReader = new StreamReader(HQL_Plus_FilePaths.qTable1Path))
        {
            string raw_line;
            while ((raw_line = streamReader.ReadLine()) != null)
            {
                string key_value_str = String.Join("", raw_line.Split('(', ')', ' '));
                string[] action_state_value = key_value_str.Split(':');

                CardId action = (CardId)int.Parse(action_state_value[0]);
                int state = int.Parse(action_state_value[1]);
                double value = double.Parse(action_state_value[2].Replace(',', '.'), System.Globalization.CultureInfo.InvariantCulture);

                if (!qTable1.TryAdd(Tuple.Create(action, state), value))
                {
                    // ErrorFileWriteLine($"Didn't add key to qTable");
                    // ErrorFileWriteLine($"action = {action}");
                    // ErrorFileWriteLine($"state = {state}");
                    // ErrorFileWriteLine($"value = {value}");
                }
            }
        }

        using (var streamReader = new StreamReader(HQL_Plus_FilePaths.qTable2Path))
        {
            string raw_line;
            while ((raw_line = streamReader.ReadLine()) != null)
            {
                string key_value_str = String.Join("", raw_line.Split('(', ')', ' '));
                string[] action_state_value = key_value_str.Split(':');

                CardId action = (CardId)int.Parse(action_state_value[0]);
                int state = int.Parse(action_state_value[1]);
                double value = double.Parse(action_state_value[2].Replace(',', '.'), System.Globalization.CultureInfo.InvariantCulture);

                if (!qTable2.TryAdd(Tuple.Create(action, state), value))
                {
                    // ErrorFileWriteLine($"Didn't add key to qTable");
                    // ErrorFileWriteLine($"action = {action}");
                    // ErrorFileWriteLine($"state = {state}");
                    // ErrorFileWriteLine($"value = {value}");
                }
            }
        }

    }

    public void HQL_Plus_SaveQTableToFile()
    {
        //using (var sw = new StreamWriter(HQL_Plus_FilePaths.qTable1Path))
        //{
        //    foreach (var (key, value) in qTable1)
        //    {
        //        sw.WriteLine(((int)key.Item1).ToString() + " : " + key.Item2.ToString() + " : " + value.ToString());
        //    }
        //}

        //using (var sw = new StreamWriter(HQL_Plus_FilePaths.qTable2Path))
        //{
        //    foreach (var (key, value) in qTable2)
        //    {
        //        sw.WriteLine(((int)key.Item1).ToString() + " : " + key.Item2.ToString() + " : " + value.ToString());
        //    }
        //}
    }

    public void HQL_Plus_ErrorFileWriteLine(string msg)
    {
        using (var sw = new StreamWriter(HQL_Plus_FilePaths.errorFile, append: true))
        {
            sw.WriteLine(msg);
        }
    }

    public void HQL_Plus_TmpFileWriteLine(string msg)
    {
        using (var sw = new StreamWriter(HQL_Plus_FilePaths.tmpFile, append: true))
        {
            sw.WriteLine(msg);
        }
    }

    public double HQL_Plus_TryToGetQValue(CardId played_card, int deck_cards, bool useTable1 = true)
    {
        if (deck_cards > HQL_Plus_Consts.max_deck_cards)
        {
            deck_cards = HQL_Plus_Consts.max_deck_cards;
        }
        QKey key = Tuple.Create(played_card, deck_cards);
        double q_value = 0;

        var targetTable = useTable1 ? qTable1 : qTable2;

        if (targetTable.TryGetValue(key, out q_value))
        {
            return q_value;
        }
        else
        {
            // ErrorFileWriteLine("Didn't get key value from qTable");
            // ErrorFileWriteLine($"action = {played_card}");
            // ErrorFileWriteLine($"state = {deck_cards}");

            return 5;
        }
    }

    public int HQL_Plus_TryToGetQValueToInt(CardId played_card, int deck_cards)
    {

        // Average q-values from both tables for action selection
        double q_value1 = HQL_Plus_TryToGetQValue(played_card, deck_cards, true);
        double q_value2 = HQL_Plus_TryToGetQValue(played_card, deck_cards, false);

        double avg_q_value = (q_value1 + q_value2) / 2.0;

        int q_value = (int)avg_q_value; // HQL_Plus_TryToGetQValue(played_card, deck_cards); // try to round and not cast to int
        if (q_value == 0)
        {
            q_value = 1;
        }

        return q_value;
    }

    public HQL_Plus_Stage HQL_Plus_TransformGameStateToStages(SeededGameState sgs)
    {
        Func<int, HQL_Plus_Stage> FindStage = x =>
        {
            switch (x)
            {
                case >= 0 and < 10:
                    return HQL_Plus_Stage.Start;
                case >= 10 and < 20:
                    return HQL_Plus_Stage.Early;
                case >= 20 and < 30:
                    return HQL_Plus_Stage.Middle;
                case >= 30:
                    return HQL_Plus_Stage.Late;
                default:
                    using (var sw = new StreamWriter(HQL_Plus_FilePaths.errorFile, append: true))
                    {
                        // sw.WriteLine("Unexpected prestige value in TransfromGameStateToGrade() = " + x.ToString());
                        // sw.WriteLine(DateTime.Now + "\n");
                    }
                    return HQL_Plus_Stage.Late;
            }
        };

        int sum = sgs.EnemyPlayer.Prestige + turn_counter;

        HQL_Plus_Stage stage = FindStage(sum);

        if (sgs.EnemyPlayer.Prestige > 40)
        {
            stage = HQL_Plus_Stage.End;
        }

        return stage;
    }

    public void HQL_Plus_IncrementTurnCounter()
    {
        ++turn_counter;
    }

    public void HQL_Plus_UpdateDeckCardsCounter(SeededGameState sgs)
    {
        List<UniqueCard> all_cards = sgs.CurrentPlayer.Hand.Concat(sgs.CurrentPlayer.Played.Concat(sgs.CurrentPlayer.CooldownPile.Concat(sgs.CurrentPlayer.DrawPile))).ToList();
        int[] updated_counter = new int[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
        foreach (var card in all_cards)
        {
            if (card.Type != CardType.STARTER)
                updated_counter[(int)card.Deck]++;
        }

        for (int i = 0; i < updated_counter.Count(); ++i)
        {
            if (updated_counter[i] > HQL_Plus_Consts.max_deck_cards)
            {
                updated_counter[i] = HQL_Plus_Consts.max_deck_cards;
            }
        }
        deck_cards_counters = updated_counter;
    }

    public void HQL_Plus_ResetVariables()
    {
        turn_counter = 0;

        deck_cards_counters = new int[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

        actions_before_this_turn_count = 0;
        played_cards_this_turn = new HashSet<UniqueCard>();
        gained_cards = new SortedSet<CardId>();
    }

    public double HQL_Plus_MaxQValueFromNewState(SeededGameState sgs, int deck_id, bool useTable1ForSelection)
    {
        double result = 0;
        var tavern_cards = sgs.TavernCards.Concat(sgs.TavernAvailableCards);
        HashSet<CardId> deck_cards = new HashSet<CardId>();

        foreach (var card in tavern_cards)
        {
            //if (card.Deck == PatronId.PELIN)
            //{
            deck_cards.Add(card.CommonId);
            //}
        }

        // If no PELIN cards available, return 0 (same as original behavior)
        if (deck_cards.Count == 0)
        {
            return 0;
        }

        // Double Q-Learning: Find best action using one table, evaluate with the other
        CardId bestAction = deck_cards.First();
        double bestActionValue = double.MinValue;

        // Find best action using the selection table
        foreach (var possible_card in deck_cards)
        {
            double selection_value = HQL_Plus_TryToGetQValue(possible_card, deck_cards_counters[deck_id] + 1, useTable1ForSelection);
            if (selection_value > bestActionValue)
            {
                bestActionValue = selection_value;
                bestAction = possible_card;
            }
        }

        // Return value of best action from the opposite table
        result = HQL_Plus_TryToGetQValue(bestAction, deck_cards_counters[deck_id] + 1, !useTable1ForSelection);

        return result;
    }

    public void HQL_Plus_CalculateNewQValue(SeededGameState sgs, CardId played_card, int card_score, int deck_id)
    {
        // Randomly choose which table to update
        bool updateTable1 = doubleQLearningRandom.NextDouble() < 0.5;

        QKey q_key = Tuple.Create(played_card, deck_cards_counters[deck_id]);

        if (updateTable1)
        {
            // update table 1 : use table 1 for action selection, table 2 for value estimation
            double current_q_value = HQL_Plus_TryToGetQValue(played_card, deck_cards_counters[deck_id], true);
            double next_state_value = HQL_Plus_MaxQValueFromNewState(sgs, deck_id, true);

            double new_q_value = (1.0 - learningRate) * current_q_value + learningRate * (card_score + discountFactor * next_state_value);

            qTable1[q_key] = new_q_value;
        }
        else
        {
            // Update table 2: use table 2 for action selection, table 1 for value estimation
            double current_q_value = HQL_Plus_TryToGetQValue(played_card, deck_cards_counters[deck_id], false);
            double next_state_value = HQL_Plus_MaxQValueFromNewState(sgs, deck_id, false);

            double new_q_value = (1.0 - learningRate) * current_q_value + learningRate * (card_score + discountFactor * next_state_value);

            qTable2[q_key] = new_q_value;
        }
    }

    // Pick best by value move or explore other move
    // Return weakest move, if didn't pick any
    // public Move PickBuyMove(SeededGameState sgs, List<Move> buy_moves)
    // {
    //     Random random = new Random();

    //     List<Tuple<int, double>> moves_values = new List<Tuple<int, double>>();
    //     for (int i = 0; i < buy_moves.Count; ++i)
    //     {
    //         QKey key = ConstructQTableKey(sgs, buy_moves[i]);
    //         double q_value = TryToGetQValue(key);
    //         moves_values.Add(Tuple.Create(i, q_value));
    //     }

    //     // Sort in descending order
    //     moves_values.Sort((x, y) => y.Item2.CompareTo(x.Item2));

    //     Move result = buy_moves[moves_values.First().Item1];

    //     // if (random.NextDouble() < explorationChance)
    //     // {
    //     //     result = buy_moves[random.Next(buy_moves.Count)];
    //     // }

    //     var card_move = (SimpleCardMove)result;
    //     IncrementDeckCardsCounter(card_move.Card.Deck);

    //     return result;
    // }

    public int HQL_Plus_ScoreForCompletedAction(CompletedAction action, SeededGameState sgs)
    {
        int result = 0;

        int power_prestige_w = 2;
        int coin_w = 1;
        int average_card_w = 3;
        int max_cost = 10;
        int patron_w = 6;

        switch (action.Type)
        {
            case CompletedActionType.ACQUIRE_CARD:
                {
                    if (action.TargetCard is not null)
                    {
                        result += action.TargetCard.Cost;
                    }
                    break;
                }
            case CompletedActionType.GAIN_COIN:
                {
                    result += action.Amount * coin_w;
                    break;
                }
            case CompletedActionType.GAIN_POWER:
                {
                    result += action.Amount * power_prestige_w;
                    break;
                }
            case CompletedActionType.GAIN_PRESTIGE:
                {
                    result += action.Amount * power_prestige_w;
                    break;
                }
            case CompletedActionType.OPP_LOSE_PRESTIGE:
                {
                    result += Math.Abs(action.Amount) * power_prestige_w;
                    break;
                }
            case CompletedActionType.REPLACE_TAVERN:
                {
                    // combo numbers?
                    break;
                }
            case CompletedActionType.DESTROY_CARD:
                {
                    if (action.TargetCard is not null)
                    {
                        result += (max_cost - action.TargetCard.Cost) / 2;
                    }
                    break;
                }
            case CompletedActionType.DRAW:
                {
                    result += average_card_w;
                    break;
                }
            case CompletedActionType.DISCARD:
                {
                    result += average_card_w;
                    break;
                }
            case CompletedActionType.REFRESH:
                {
                    if (action.TargetCard is not null)
                    {
                        result += action.TargetCard.Cost;
                    }
                    break;
                }
            case CompletedActionType.KNOCKOUT:
                {
                    // if (action.TargetCard is not null)
                    // {
                    //     if (action.TargetCard.HP > 0)
                    //     {
                    //         result += action.TargetCard.HP * power_prestige_w;
                    //     }
                    // }
                    break;
                }
            case CompletedActionType.ADD_PATRON_CALLS:
                {
                    result += patron_w;
                    break;
                }
            case CompletedActionType.ADD_SUMMERSET_SACKING:
                {
                    result += 2;
                    break;
                }
            case CompletedActionType.HEAL_AGENT:
                {
                    result += action.Amount * power_prestige_w;
                    break;
                }
            default:
                {
                    // Other actions type don't bring value from played card.
                    break;
                }
        }

        return result;
    }

    public void HQL_Plus_SaveGainedCards(SeededGameState sgs)
    {
        List<UniqueCard> all_cards = sgs.CurrentPlayer.Hand.Concat(sgs.CurrentPlayer.Played.Concat(sgs.CurrentPlayer.CooldownPile.Concat(sgs.CurrentPlayer.DrawPile))).ToList();

        foreach (var card in all_cards)
        {
            if (card.Type != CardType.STARTER && card.Type != CardType.CURSE)
            {
                if (!gained_cards.Contains(card.CommonId))
                {
                    gained_cards.Add(card.CommonId);
                }
            }
        }
    }

    public void HQL_Plus_SavePlayedCardIfApplicable(Move move)
    {
        if (move is SimpleCardMove card && gained_cards.Contains(card.Card.CommonId))
        {
            played_cards_this_turn.Add(card.Card);
        }
    }

    public void HQL_Plus_UpdateQValuesForPlayedCardsAtEndOfTurn(SeededGameState sgs)
    {
        Dictionary<CardId, int> card_id_to_turn_score = new Dictionary<CardId, int>();

        List<CompletedAction> actions_completed_this_turn = sgs.CompletedActions.Skip(Math.Max(0, actions_before_this_turn_count)).ToList();

        foreach (var action in actions_completed_this_turn)
        {
            if (action.SourceCard is not null && gained_cards.Contains(action.SourceCard.CommonId))
            {
                int action_score = HQL_Plus_ScoreForCompletedAction(action, sgs);
                if (card_id_to_turn_score.ContainsKey(action.SourceCard.CommonId))
                {
                    card_id_to_turn_score[action.SourceCard.CommonId] += action_score;
                }
                else
                {
                    card_id_to_turn_score[action.SourceCard.CommonId] = action_score;
                }
            }
        }

        foreach (var (card_id, score) in card_id_to_turn_score)
        {
            var unique_card = GlobalCardDatabase.Instance.GetCard(card_id);
            HQL_Plus_CalculateNewQValue(sgs, card_id, score, (int)unique_card.Deck);
        }

        played_cards_this_turn = new HashSet<UniqueCard>();
        actions_before_this_turn_count = sgs.CompletedActions.Count;
    }

    public int HQL_Plus_Heuristic(SeededGameState sgs)
    {
        int stage = (int)HQL_Plus_TransformGameStateToStages(sgs);
        int player_score = 0;
        int enemy_score = 0;

        player_score += sgs.CurrentPlayer.Prestige * HQL_Plus_Consts.prestige_weight[stage];
        player_score += sgs.CurrentPlayer.Power * HQL_Plus_Consts.power_weight[stage];
        player_score += sgs.CurrentPlayer.Coins * HQL_Plus_Consts.coins_weight[stage];

        enemy_score += sgs.EnemyPlayer.Prestige * HQL_Plus_Consts.prestige_weight[stage];
        enemy_score += sgs.EnemyPlayer.Power * HQL_Plus_Consts.power_weight[stage];
        enemy_score += sgs.EnemyPlayer.Coins * HQL_Plus_Consts.coins_weight[stage];

        int enemy_patrons = 0;

        foreach (var (key, value) in sgs.PatronStates.All)
        {
            if (key == PatronId.TREASURY)
            {
                continue;
            }
            if (value == sgs.CurrentPlayer.PlayerID)
            {
                player_score += HQL_Plus_Consts.patron_weight[stage];
                switch (key)
                {
                    case PatronId.ANSEI:
                        player_score += HQL_Plus_Consts.ansei_weight[stage];
                        break;
                    case PatronId.DUKE_OF_CROWS:
                        player_score += HQL_Plus_Consts.crow_weight[stage];
                        break;
                    case PatronId.ORGNUM:
                        player_score += HQL_Plus_Consts.orgnum_weight[stage];
                        break;
                    default:
                        break;
                }
            }
            else if (value == sgs.EnemyPlayer.PlayerID)
            {
                ++enemy_patrons;
                enemy_score += HQL_Plus_Consts.patron_weight[stage];
                switch (key)
                {
                    case PatronId.ANSEI:
                        enemy_score += HQL_Plus_Consts.ansei_weight[stage];
                        break;
                    case PatronId.DUKE_OF_CROWS:
                        enemy_score += HQL_Plus_Consts.crow_weight[stage];
                        break;
                    case PatronId.ORGNUM:
                        enemy_score += HQL_Plus_Consts.orgnum_weight[stage];
                        break;
                    default:
                        break;
                }
            }
        }

        if (enemy_patrons == 2)
        {
            enemy_score += HQL_Plus_Consts.patron_weight[stage] * 2;
        }
        else if (enemy_patrons == 3)
        {
            enemy_score += HQL_Plus_Consts.patron_weight[stage] * 4;
        }
        else if (enemy_patrons == 4)
        {
            enemy_score += HQL_Plus_Consts.patron_weight[stage] * 100;
        }

        List<UniqueCard> all_cards = sgs.CurrentPlayer.Hand.Concat(sgs.CurrentPlayer.Played.Concat(sgs.CurrentPlayer.CooldownPile.Concat(sgs.CurrentPlayer.DrawPile))).ToList();
        int[] player_combo = new int[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
        List<UniqueCard> all_enemy_cards = sgs.EnemyPlayer.Hand.Concat(sgs.EnemyPlayer.DrawPile).Concat(sgs.CurrentPlayer.Played.Concat(sgs.CurrentPlayer.CooldownPile)).ToList();
        int[] enemy_combo = new int[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

        foreach (var card in all_cards)
        {
            if (card.CommonId == CardId.GOLD)
            {
                player_score += HQL_Plus_Consts.gold_card_weight[stage];
            }
            else
            {
                int q_value = HQL_Plus_TryToGetQValueToInt(card.CommonId, deck_cards_counters[(int)card.Deck]);
                player_score += q_value * HQL_Plus_Consts.card_weight[stage];
                player_combo[(int)card.Deck]++;
            }
        }

        foreach (var card in all_enemy_cards)
        {
            enemy_combo[(int)card.Deck]++;
        }

        foreach (var combo in player_combo)
        {
            player_score += HQL_Plus_Consts.combo_weight[stage] * combo;
        }
        foreach (var combo in enemy_combo)
        {
            enemy_score += HQL_Plus_Consts.combo_weight[stage] * combo;
        }

        foreach (var agent in sgs.CurrentPlayer.Agents)
        {
            int q_value = HQL_Plus_TryToGetQValueToInt(agent.RepresentingCard.CommonId, deck_cards_counters[(int)agent.RepresentingCard.Deck]);
            player_score += HQL_Plus_Consts.active_agent_weight[stage] * q_value + agent.CurrentHp * HQL_Plus_Consts.hp_weight[stage];
        }

        foreach (var agent in sgs.EnemyPlayer.Agents)
        {
            int q_value = HQL_Plus_TryToGetQValueToInt(agent.RepresentingCard.CommonId, deck_cards_counters[(int)agent.RepresentingCard.Deck]);
            enemy_score += HQL_Plus_Consts.active_agent_weight[stage] * q_value + agent.CurrentHp * HQL_Plus_Consts.hp_weight[stage];
        }

        return player_score - enemy_score;
    }
}