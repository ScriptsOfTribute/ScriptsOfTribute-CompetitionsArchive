using System.Globalization;
using ScriptsOfTribute;
using ScriptsOfTribute.Board;
using ScriptsOfTribute.Board.CardAction;
using ScriptsOfTribute.Board.Cards;
using ScriptsOfTribute.Serializers;

namespace PatronZeroSubmission;

internal sealed record SearchSettings(
    int Simulations,
    int Worlds,
    int MaxChildren,
    double Exploration,
    double DecisionMilliseconds);

internal interface INeuralSearch
{
    int Search(
        GameState visibleState,
        IReadOnlyList<Move> legalMoves,
        ulong baseSeed,
        int decisionIndex,
        SearchSettings settings);
}

internal interface ISearchEvaluator
{
    PatronZeroModelOutput Evaluate(GameState state, IReadOnlyList<Move> moves);
    PatronZeroModelOutput Evaluate(SeededGameState state, IReadOnlyList<Move> moves);
}

internal static class SearchMath
{
    public static double[] Softmax(IReadOnlyList<float> logits)
    {
        if (logits.Count == 0 || logits.Any(logit => !float.IsFinite(logit)))
        {
            throw new ArgumentException("Policy logits must be non-empty and finite.", nameof(logits));
        }
        var max = logits.Max();
        var values = logits.Select(value => Math.Exp(value - max)).ToArray();
        var sum = values.Sum();
        return sum <= 0 || double.IsNaN(sum)
            ? Enumerable.Repeat(1.0 / logits.Count, logits.Count).ToArray()
            : values.Select(value => value / sum).ToArray();
    }

    public static IReadOnlyList<int> CandidateIndices(
        IReadOnlyList<double> priors,
        IReadOnlyList<Move> moves,
        int maxChildren,
        bool isRoot)
    {
        if (priors.Count != moves.Count)
        {
            throw new ArgumentException("Every legal move must have exactly one policy prior.");
        }
        if (maxChildren < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(maxChildren), "The candidate cap must be positive.");
        }

        var actionKeys = MoveKeyMap.Create(moves).Keys;
        var ordered = Enumerable.Range(0, priors.Count)
            .OrderByDescending(index => priors[index])
            .ThenBy(index => actionKeys[index], StringComparer.Ordinal)
            .ToArray();
        // The real root must retain every legal action; only deeper simulated
        // states may use the candidate cap.
        if (isRoot || ordered.Length <= maxChildren)
        {
            return ordered;
        }

        var selected = ordered.Take(maxChildren).ToList();
        var seen = selected.ToHashSet();
        // Current-turn search must always retain the explicit horizon action.
        // Keeping it only at the real root can force a search branch past the
        // intended END_TURN boundary after child pruning.
        var endTurn = Array.FindIndex(moves.ToArray(), move => move.Command == CommandEnum.END_TURN);
        if (endTurn >= 0 && seen.Add(endTurn))
        {
            if (selected.Count >= maxChildren)
            {
                selected.RemoveAt(selected.Count - 1);
            }
            selected.Add(endTurn);
        }
        return selected;
    }

    public static int SelectMoveIndex(
        IReadOnlyList<int> visits,
        IReadOnlyList<double> valueSums,
        IReadOnlyList<Move> moves)
    {
        if (moves.Count == 0
            || visits.Count != moves.Count
            || valueSums.Count != moves.Count)
        {
            throw new ArgumentException("Visit and value statistics must match the legal move count.");
        }
        var keys = MoveKeyMap.Create(moves).Keys;
        if (visits.Any(count => count > 0))
        {
            return Enumerable.Range(0, moves.Count)
                .OrderByDescending(index => visits[index])
                .ThenByDescending(index => visits[index] == 0
                    ? double.NegativeInfinity
                    : valueSums[index] / visits[index])
                .ThenBy(index => keys[index], StringComparer.Ordinal)
                .First();
        }
        return Enumerable.Range(0, moves.Count)
            .Where(index => moves[index].Command != CommandEnum.END_TURN)
            .OrderBy(index => keys[index], StringComparer.Ordinal)
            .DefaultIfEmpty(0)
            .First();
    }

    public static int SelectPolicyMove(
        IReadOnlyList<double> policy,
        IReadOnlyList<Move> moves)
    {
        if (moves.Count == 0 || policy.Count != moves.Count)
        {
            throw new ArgumentException("Policy statistics must match a non-empty legal move list.");
        }
        var keys = MoveKeyMap.Create(moves).Keys;
        return Enumerable.Range(0, moves.Count)
            .OrderByDescending(index => policy[index])
            .ThenBy(index => keys[index], StringComparer.Ordinal)
            .First();
    }

    public static void ValidateTreeSearchInputs(
        IReadOnlyList<Move> legalMoves,
        SearchSettings settings)
    {
        if (legalMoves.Count == 0)
        {
            throw new ArgumentException("Tree search requires at least one legal move.", nameof(legalMoves));
        }
        if (settings.Simulations < 1 || settings.Worlds < 1 || settings.MaxChildren < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(settings), "Search counts and the child cap must be positive.");
        }
        if (!double.IsFinite(settings.Exploration) || settings.Exploration < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(settings), "Exploration must be finite and non-negative.");
        }
        if (!double.IsFinite(settings.DecisionMilliseconds) || settings.DecisionMilliseconds < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(settings), "The decision budget must be finite and non-negative.");
        }
    }

    public static double ValueForRoot(float currentPlayerValue, PlayerEnum statePlayer, PlayerEnum rootPlayer)
        => statePlayer == rootPlayer ? currentPlayerValue : -currentPlayerValue;

    public static double TerminalValue(EndGameState endState, PlayerEnum rootPlayer)
        => endState.Winner == rootPlayer ? 1.0
            : endState.Winner == PlayerEnum.NO_PLAYER_SELECTED ? 0.0
            : -1.0;

    public static ulong SimulationSeed(ulong baseSeed, int decisionIndex, int simulationIndex)
    {
        if (decisionIndex < 0 || simulationIndex < 0)
        {
            throw new ArgumentOutOfRangeException(
                decisionIndex < 0 ? nameof(decisionIndex) : nameof(simulationIndex));
        }
        return unchecked(
            baseSeed
            + (ulong)decisionIndex * 1_000_003UL
            + (ulong)simulationIndex * 97_409UL
            + 17UL);
    }
}

internal static class DeterminizationSidecar
{
    public static SeededGameState Create(GameState visibleState, ulong seed)
    {
        var expectedPrefix = visibleState.CurrentPlayer.KnownUpcomingDraws
            .Select(CardIdentity)
            .ToArray();
        var world = visibleState.ToSeededGameState(seed);
        var actualPrefix = world.CurrentPlayer.DrawPile
            .Take(expectedPrefix.Length)
            .Select(CardIdentity)
            .ToArray();
        if (!expectedPrefix.AsSpan().SequenceEqual(actualPrefix))
        {
            throw new InvalidOperationException("Determinization changed the visible known-upcoming draw prefix.");
        }

        // The official SeededGameState constructor preserves prefix order but
        // drops the metadata count. Rehydrate only the fresh sampled copy so
        // subsequent ApplyMove calls and model features retain the visible fact.
        world.CurrentPlayer.KnownUpcomingDraws.Clear();
        world.CurrentPlayer.KnownUpcomingDraws.AddRange(
            world.CurrentPlayer.DrawPile.Take(expectedPrefix.Length));
        Validate(world);
        return world;
    }

    public static void Validate(SeededGameState state)
    {
        var metadata = state.CurrentPlayer.KnownUpcomingDraws.Select(CardIdentity).ToArray();
        var prefix = state.CurrentPlayer.DrawPile.Take(metadata.Length).Select(CardIdentity).ToArray();
        if (!metadata.AsSpan().SequenceEqual(prefix))
        {
            throw new InvalidOperationException("Known-upcoming metadata no longer matches the sampled draw prefix.");
        }
    }

    private static string CardIdentity(UniqueCard card)
        => $"{(int)card.CommonId}:{card.UniqueId.Value}";
}

// SO-ISMCTS shares nodes only when these exact, fairly observable facts match.
// The neural feature vector alone is intentionally compressed and is not a safe key.
internal sealed class ObservationKey : IEquatable<ObservationKey>
{
    private readonly float[] _state;
    private readonly string[] _legalMoveKeys;
    private readonly string[] _semanticFacts;
    private readonly int _hashCode;

    private ObservationKey(float[] state, string[] legalMoveKeys, string[] semanticFacts)
    {
        _state = state;
        _legalMoveKeys = legalMoveKeys;
        _semanticFacts = semanticFacts;
        var hash = new HashCode();
        foreach (var value in state)
        {
            hash.Add(BitConverter.SingleToInt32Bits(value));
        }
        foreach (var key in legalMoveKeys)
        {
            hash.Add(key, StringComparer.Ordinal);
        }
        foreach (var fact in semanticFacts)
        {
            hash.Add(fact, StringComparer.Ordinal);
        }
        _hashCode = hash.ToHashCode();
    }

    public static ObservationKey From(GameState state, IReadOnlyList<Move> moves)
        => new(
            PatronZeroFeatures.EncodeState(state),
            MoveKeyMap.Create(moves).Keys.OrderBy(key => key, StringComparer.Ordinal).ToArray(),
            SemanticFacts(state));

    public static ObservationKey From(SeededGameState state, IReadOnlyList<Move> moves)
        => new(
            PatronZeroFeatures.EncodeState(state),
            MoveKeyMap.Create(moves).Keys.OrderBy(key => key, StringComparer.Ordinal).ToArray(),
            SemanticFacts(state));

    public bool Equals(ObservationKey? other)
        => other is not null
            && _hashCode == other._hashCode
            && _state.AsSpan().SequenceEqual(other._state)
            && _legalMoveKeys.AsSpan().SequenceEqual(other._legalMoveKeys)
            && _semanticFacts.AsSpan().SequenceEqual(other._semanticFacts);

    public override bool Equals(object? obj) => obj is ObservationKey other && Equals(other);

    public override int GetHashCode() => _hashCode;

    private static string[] SemanticFacts(GameState state)
        =>
        [
            $"board:{state.BoardState}",
            $"current:{state.CurrentPlayer.PlayerID}:{state.CurrentPlayer.Power}:{state.CurrentPlayer.Coins}:{state.CurrentPlayer.Prestige}:{state.CurrentPlayer.PatronCalls}",
            $"current-hand:{CardMultiset(state.CurrentPlayer.Hand)}",
            $"current-draw:{CardMultiset(state.CurrentPlayer.DrawPile)}",
            $"current-cooldown:{CardMultiset(state.CurrentPlayer.CooldownPile)}",
            $"current-played:{CardMultiset(state.CurrentPlayer.Played)}",
            $"current-known:{string.Join(",", state.CurrentPlayer.KnownUpcomingDraws.Select(CardIdentity))}",
            $"current-agents:{AgentSet(state.CurrentPlayer.Agents)}",
            $"enemy:{state.EnemyPlayer.PlayerID}:{state.EnemyPlayer.Power}:{state.EnemyPlayer.Coins}:{state.EnemyPlayer.Prestige}",
            $"enemy-hand-draw:{CardMultiset(state.EnemyPlayer.HandAndDraw)}",
            $"enemy-cooldown:{CardMultiset(state.EnemyPlayer.CooldownPile)}",
            $"enemy-played:{CardMultiset(state.EnemyPlayer.Played)}",
            $"enemy-agents:{AgentSet(state.EnemyPlayer.Agents)}",
            $"tavern-available:{CardMultiset(state.TavernAvailableCards)}",
            $"tavern-deck:{CardMultiset(state.TavernCards)}",
            $"patrons:{PatronIdentity(state.PatronStates)}",
            $"choice:{ChoiceIdentity(state.PendingChoice)}",
            $"combos:{ComboIdentity(state.ComboStates)}",
            $"upcoming:{EffectSequence(state.UpcomingEffects)}",
            $"next-turn:{EffectSequence(state.StartOfNextTurnEffects)}",
        ];

    private static string[] SemanticFacts(SeededGameState state)
        =>
        [
            $"board:{state.BoardState}",
            $"current:{state.CurrentPlayer.PlayerID}:{state.CurrentPlayer.Power}:{state.CurrentPlayer.Coins}:{state.CurrentPlayer.Prestige}:{state.CurrentPlayer.PatronCalls}",
            $"current-hand:{CardMultiset(state.CurrentPlayer.Hand)}",
            $"current-draw:{CardMultiset(state.CurrentPlayer.DrawPile)}",
            $"current-cooldown:{CardMultiset(state.CurrentPlayer.CooldownPile)}",
            $"current-played:{CardMultiset(state.CurrentPlayer.Played)}",
            $"current-known:{string.Join(",", state.CurrentPlayer.KnownUpcomingDraws.Select(CardIdentity))}",
            $"current-agents:{AgentSet(state.CurrentPlayer.Agents)}",
            $"enemy:{state.EnemyPlayer.PlayerID}:{state.EnemyPlayer.Power}:{state.EnemyPlayer.Coins}:{state.EnemyPlayer.Prestige}",
            $"enemy-hand-draw:{CardMultiset(state.EnemyPlayer.Hand.Concat(state.EnemyPlayer.DrawPile))}",
            $"enemy-cooldown:{CardMultiset(state.EnemyPlayer.CooldownPile)}",
            $"enemy-played:{CardMultiset(state.EnemyPlayer.Played)}",
            $"enemy-agents:{AgentSet(state.EnemyPlayer.Agents)}",
            $"tavern-available:{CardMultiset(state.TavernAvailableCards)}",
            $"tavern-deck:{CardMultiset(state.TavernCards)}",
            $"patrons:{PatronIdentity(state.PatronStates)}",
            $"choice:{ChoiceIdentity(state.PendingChoice)}",
            $"combos:{ComboIdentity(state.ComboStates)}",
            $"upcoming:{EffectSequence(state.UpcomingEffects)}",
            $"next-turn:{EffectSequence(state.StartOfNextTurnEffects)}",
        ];

    private static string ChoiceIdentity(SerializedChoice? choice)
    {
        if (choice is null)
        {
            return "none";
        }
        var options = choice.Type == Choice.DataType.CARD
            ? choice.PossibleCards.Select(CardIdentity)
            : choice.PossibleEffects.Select(effect =>
                $"{(int)effect.Type}:{effect.Amount}:{effect.Combo}:{CardIdentity(effect.ParentCard)}");
        return $"{choice.Type}:{choice.MinChoices}:{choice.MaxChoices}:{choice.Context}:{choice.ChoiceFollowUp}:{string.Join(",", options.OrderBy(value => value, StringComparer.Ordinal))}";
    }

    private static string CardIdentity(UniqueCard card)
        => $"{(int)card.CommonId}:{card.UniqueId.Value}";

    private static string CardMultiset(IEnumerable<UniqueCard> cards)
        => string.Join(",", cards.Select(CardIdentity).OrderBy(value => value, StringComparer.Ordinal));

    private static string AgentSet(IEnumerable<SerializedAgent> agents)
        => string.Join(",", agents
            .Select(agent => $"{CardIdentity(agent.RepresentingCard)}:{agent.CurrentHp}:{agent.Activated}")
            .OrderBy(value => value, StringComparer.Ordinal));

    private static string PatronIdentity(PatronStates patrons)
        => string.Join(",", patrons.All
            .OrderBy(pair => pair.Key)
            .Select(pair => $"{(int)pair.Key}:{(int)pair.Value}"));

    private static string ComboIdentity(ComboStates combos)
        => string.Join(";", combos.All
            .OrderBy(pair => pair.Key)
            .Select(pair => $"{(int)pair.Key}:{pair.Value.CurrentCombo}:"
                + string.Join("/", pair.Value.All.Select(EffectSequence))));

    private static string EffectSequence(IEnumerable<UniqueBaseEffect> effects)
        => string.Join(",", effects.Select(EffectIdentity));

    private static string EffectIdentity(UniqueBaseEffect effect)
        => effect switch
        {
            UniqueEffect simple => $"effect:{(int)simple.Type}:{simple.Amount}:{simple.Combo}:{CardIdentity(simple.ParentCard)}",
            UniqueEffectOr either => $"or:{either.Combo}:{CardIdentity(either.ParentCard)}:{string.Join("|", new[] { EffectIdentity(either.GetLeft()), EffectIdentity(either.GetRight()) }.OrderBy(value => value, StringComparer.Ordinal))}",
            _ => $"{effect.GetType().FullName}:{effect}",
        };
}

internal sealed class MoveKeyMap
{
    private MoveKeyMap(string[] keys, IReadOnlyDictionary<string, Move> movesByKey)
    {
        Keys = keys;
        MovesByKey = movesByKey;
    }

    public string[] Keys { get; }
    public IReadOnlyDictionary<string, Move> MovesByKey { get; }

    public static MoveKeyMap Create(IReadOnlyList<Move> moves)
    {
        var entries = moves
            .Select((move, index) => new MoveEntry(index, BaseKey(move), PhysicalTieBreak(move)))
            .ToArray();
        var keys = new string[moves.Count];
        foreach (var group in entries.GroupBy(entry => entry.BaseKey, StringComparer.Ordinal))
        {
            var ordered = group.OrderBy(entry => entry.PhysicalTieBreak, StringComparer.Ordinal).ToArray();
            for (var occurrence = 0; occurrence < ordered.Length; occurrence++)
            {
                // Keep the occurrence component even for a singleton, so a
                // key does not change merely because an equivalent sibling is
                // available in another determinization.
                keys[ordered[occurrence].Index] = $"{ordered[occurrence].BaseKey}#{occurrence}";
            }
        }

        return new MoveKeyMap(
            keys,
            Enumerable.Range(0, moves.Count).ToDictionary(index => keys[index], index => moves[index], StringComparer.Ordinal));
    }

    private static string BaseKey(Move move)
        => move switch
        {
            SimpleCardMove cardMove => $"{move.Command}:card:{(int)cardMove.Card.CommonId}:{cardMove.Card.UniqueId.Value}",
            SimplePatronMove patronMove => $"{move.Command}:patron:{(int)patronMove.PatronId}",
            MakeChoiceMove<UniqueCard> cardChoice => $"{move.Command}:cards:{string.Join(",", cardChoice.Choices.Select(card => $"{(int)card.CommonId}:{card.UniqueId.Value}").OrderBy(value => value, StringComparer.Ordinal))}",
            MakeChoiceMove<UniqueEffect> effectChoice => $"{move.Command}:effects:{string.Join(",", effectChoice.Choices.Select(effect => $"{(int)effect.Type}:{effect.Amount}:{effect.Combo}:{(int)effect.ParentCard.CommonId}:{effect.ParentCard.UniqueId.Value}").OrderBy(value => value, StringComparer.Ordinal))}",
            _ => move.Command.ToString(),
        };

    private static string PhysicalTieBreak(Move move)
        => move switch
        {
            SimpleCardMove cardMove => cardMove.Card.UniqueId.Value.ToString(CultureInfo.InvariantCulture),
            MakeChoiceMove<UniqueCard> cardChoice => string.Join(",", cardChoice.Choices.Select(card => card.UniqueId.Value).OrderBy(value => value)),
            _ => move.ToString() ?? string.Empty,
        };

    private sealed record MoveEntry(int Index, string BaseKey, string PhysicalTieBreak);
}
