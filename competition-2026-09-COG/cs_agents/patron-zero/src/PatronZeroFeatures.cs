using ScriptsOfTribute;
using ScriptsOfTribute.Board;
using ScriptsOfTribute.Board.Cards;
using ScriptsOfTribute.Serializers;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;

namespace PatronZeroSubmission;

internal static class PatronZeroFeatures
{
    public static int StateDim => 11 + PatronCount * PlayerCount + CardCount * 16;
    public static int MoveDim(bool choiceTokensEnabled)
        => CommandCount + CardCount + 1 + PatronCount + 1 + 1 + (choiceTokensEnabled ? ChoiceTokenBuckets : 0);
    private const int CommandCount = 7;
    private const int PlayerCount = 3;
    private const int ChoiceTokenBuckets = 64;
    private const int PatronCount = 10;
    private static readonly int CardCount = Enum.GetValues<CardId>().Cast<int>().Max() + 1;
    private static readonly float[] ScalarScales = [3, 2, 50, 50, 50, 10, 50, 50, 50, 10, 100];

    public static float[] EncodeState(GameState state)
    {
        var values = new float[StateDim];
        var index = 0;
        var scalars = new float[]
        {
            (int)state.BoardState,
            (int)state.CurrentPlayer.PlayerID,
            state.CurrentPlayer.Prestige,
            state.CurrentPlayer.Power,
            state.CurrentPlayer.Coins,
            state.CurrentPlayer.PatronCalls,
            state.EnemyPlayer.Prestige,
            state.EnemyPlayer.Power,
            state.EnemyPlayer.Coins,
            state.TavernAvailableCards.Count,
            state.TavernCards.Count,
        };
        for (var scalar = 0; scalar < scalars.Length; scalar++)
        {
            values[index++] = scalars[scalar] / ScalarScales[scalar];
        }

        foreach (var (patron, favor) in state.PatronStates.All)
        {
            var offset = index + (int)patron * PlayerCount + (int)favor;
            if (offset >= index && offset < index + PatronCount * PlayerCount)
            {
                values[offset] = 1;
            }
        }
        index += PatronCount * PlayerCount;

        AddCardCounts(values, ref index, state.CurrentPlayer.Hand, 0.25f);
        AddCardCounts(values, ref index, state.CurrentPlayer.DrawPile, 0.25f);
        AddCardCounts(values, ref index, state.CurrentPlayer.CooldownPile, 0.25f);
        AddCardCounts(values, ref index, state.CurrentPlayer.Played, 0.25f);
        AddCardCounts(values, ref index, state.CurrentPlayer.KnownUpcomingDraws, 0.25f);
        AddCardCounts(values, ref index, state.EnemyPlayer.HandAndDraw, 0.25f);
        AddCardCounts(values, ref index, state.EnemyPlayer.CooldownPile, 0.25f);
        AddCardCounts(values, ref index, state.EnemyPlayer.Played, 0.25f);
        AddCardCounts(values, ref index, state.TavernAvailableCards, 1.0f);
        AddCardCounts(values, ref index, state.TavernCards, 0.25f);
        AddAgentFeatures(values, ref index, state.CurrentPlayer.Agents);
        AddAgentFeatures(values, ref index, state.EnemyPlayer.Agents);
        return values;
    }

    public static float[] EncodeState(SeededGameState state)
    {
        var values = new float[StateDim];
        var index = 0;
        var scalars = new float[]
        {
            (int)state.BoardState,
            (int)state.CurrentPlayer.PlayerID,
            state.CurrentPlayer.Prestige,
            state.CurrentPlayer.Power,
            state.CurrentPlayer.Coins,
            state.CurrentPlayer.PatronCalls,
            state.EnemyPlayer.Prestige,
            state.EnemyPlayer.Power,
            state.EnemyPlayer.Coins,
            state.TavernAvailableCards.Count,
            state.TavernCards.Count,
        };
        for (var scalar = 0; scalar < scalars.Length; scalar++)
        {
            values[index++] = scalars[scalar] / ScalarScales[scalar];
        }

        foreach (var (patron, favor) in state.PatronStates.All)
        {
            var offset = index + (int)patron * PlayerCount + (int)favor;
            if (offset >= index && offset < index + PatronCount * PlayerCount)
            {
                values[offset] = 1;
            }
        }
        index += PatronCount * PlayerCount;

        AddCardCounts(values, ref index, state.CurrentPlayer.Hand, 0.25f);
        AddCardCounts(values, ref index, state.CurrentPlayer.DrawPile, 0.25f);
        AddCardCounts(values, ref index, state.CurrentPlayer.CooldownPile, 0.25f);
        AddCardCounts(values, ref index, state.CurrentPlayer.Played, 0.25f);
        AddCardCounts(values, ref index, state.CurrentPlayer.KnownUpcomingDraws, 0.25f);
        AddCardCounts(values, ref index, state.EnemyPlayer.Hand.Concat(state.EnemyPlayer.DrawPile), 0.25f);
        AddCardCounts(values, ref index, state.EnemyPlayer.CooldownPile, 0.25f);
        AddCardCounts(values, ref index, state.EnemyPlayer.Played, 0.25f);
        AddCardCounts(values, ref index, state.TavernAvailableCards, 1.0f);
        AddCardCounts(values, ref index, state.TavernCards, 0.25f);
        AddAgentFeatures(values, ref index, state.CurrentPlayer.Agents);
        AddAgentFeatures(values, ref index, state.EnemyPlayer.Agents);
        return values;
    }

    public static float[] EncodeMoves(IReadOnlyList<Move> moves, bool choiceTokensEnabled)
    {
        var moveDim = MoveDim(choiceTokensEnabled);
        var values = new float[moves.Count * moveDim];
        for (var moveIndex = 0; moveIndex < moves.Count; moveIndex++)
        {
            var move = moves[moveIndex];
            var offset = moveIndex * moveDim;
            values[offset + (int)move.Command] = 1;
            offset += CommandCount;

            var card = ReferencedCard(move);
            values[offset + (card >= 0 && card < CardCount ? card : CardCount)] = 1;
            offset += CardCount + 1;

            var patron = move is SimplePatronMove patronMove ? (int)patronMove.PatronId : -1;
            values[offset + (patron >= 0 && patron < PatronCount ? patron : PatronCount)] = 1;
            offset += PatronCount + 1;
            values[offset] = MathF.Log(1 + ChoiceCount(move)) / 8.0f;
            if (choiceTokensEnabled)
            {
                offset++;
                foreach (var bucket in ChoiceTokenIndices(move.ToString()))
                {
                    values[offset + bucket] = 1;
                }
            }
        }
        return values;
    }

    private static void AddCardCounts(float[] values, ref int index, IEnumerable<UniqueCard> cards, float scale)
    {
        foreach (var card in cards)
        {
            var cardId = (int)card.CommonId;
            if (cardId >= 0 && cardId < CardCount)
            {
                values[index + cardId] += scale;
            }
        }
        index += CardCount;
    }

    private static void AddAgentFeatures(float[] values, ref int index, IEnumerable<SerializedAgent> agents)
    {
        foreach (var agent in agents)
        {
            var cardId = (int)agent.RepresentingCard.CommonId;
            if (cardId < 0 || cardId >= CardCount)
            {
                continue;
            }
            values[index + cardId] += 0.25f;
            values[index + CardCount + cardId] += agent.CurrentHp / 20.0f;
            values[index + CardCount * 2 + cardId] += agent.Activated ? 0.25f : 0.0f;
        }
        index += CardCount * 3;
    }

    private static int ReferencedCard(Move move)
        => move switch
        {
            SimpleCardMove cardMove => (int)cardMove.Card.CommonId,
            MakeChoiceMove<UniqueCard> choice when choice.Choices.Count == 1 => (int)choice.Choices[0].CommonId,
            _ => -1,
        };

    private static int ChoiceCount(Move move)
        => move switch
        {
            MakeChoiceMove<UniqueCard> choice => choice.Choices.Count,
            MakeChoiceMove<UniqueEffect> choice => choice.Choices.Count,
            _ => 0,
        };

    private static IEnumerable<int> ChoiceTokenIndices(string moveText)
    {
        if (!moveText.StartsWith("CHOICE", StringComparison.Ordinal))
        {
            yield break;
        }
        var tokens = Regex.Matches(moveText.ToUpperInvariant(), "[A-Z0-9]+")
            .Select(match => match.Value)
            .Skip(1)
            .Distinct(StringComparer.Ordinal);
        foreach (var token in tokens)
        {
            var digest = SHA256.HashData(Encoding.UTF8.GetBytes(token));
            yield return (digest[0] | digest[1] << 8) % ChoiceTokenBuckets;
        }
    }
}
