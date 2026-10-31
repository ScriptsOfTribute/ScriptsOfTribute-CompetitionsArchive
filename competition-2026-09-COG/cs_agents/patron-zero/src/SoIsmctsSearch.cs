using System.Diagnostics;
using ScriptsOfTribute;
using ScriptsOfTribute.Board;
using ScriptsOfTribute.Serializers;

namespace PatronZeroSubmission;

internal sealed class SoIsmctsSearch(ISearchEvaluator evaluator) : INeuralSearch
{
    public int Search(
        GameState visibleState,
        IReadOnlyList<Move> legalMoves,
        ulong baseSeed,
        int decisionIndex,
        SearchSettings settings)
    {
        SearchMath.ValidateTreeSearchInputs(legalMoves, settings);
        var watch = Stopwatch.StartNew();
        var rootPlayer = visibleState.CurrentPlayer.PlayerID;
        var rootEvaluation = evaluator.Evaluate(visibleState, legalMoves);
        var visibleRootKey = ObservationKey.From(visibleState, legalMoves);
        var root = CreateNode(
            legalMoves,
            rootPlayer,
            rootEvaluation,
            settings,
            isRoot: true);

        var sampledWorldCount = Math.Min(settings.Worlds, settings.Simulations);
        var simulationsCompleted = 0;
        while (simulationsCompleted < settings.Simulations
               && (simulationsCompleted == 0
                   || settings.DecisionMilliseconds <= 0
                   || watch.Elapsed.TotalMilliseconds < settings.DecisionMilliseconds))
        {
            var worldIndex = simulationsCompleted % sampledWorldCount;
            var seed = SearchMath.SimulationSeed(baseSeed, decisionIndex, worldIndex);
            var state = DeterminizationSidecar.Create(visibleState, seed);
            if (!visibleRootKey.Equals(ObservationKey.From(state, legalMoves)))
            {
                throw new InvalidOperationException("SO-ISMCTS sampled root changed the acting player's fair observation.");
            }
            Simulate(root, state, legalMoves.ToList(), settings);
            simulationsCompleted++;
        }

        var rootMoveMap = MoveKeyMap.Create(legalMoves);
        var rootIndexByKey = Enumerable.Range(0, legalMoves.Count)
            .ToDictionary(index => rootMoveMap.Keys[index], index => index, StringComparer.Ordinal);
        var visits = new int[legalMoves.Count];
        var valueSums = new double[legalMoves.Count];
        foreach (var edge in root.Edges.Values)
        {
            if (!rootIndexByKey.TryGetValue(edge.ActionKey, out var index))
            {
                continue;
            }
            visits[index] = edge.Visits;
            valueSums[index] = edge.ValueSum;
        }
        var minimumAvailability = root.Edges.Values.Min(edge => edge.Availability);
        var maximumAvailability = root.Edges.Values.Max(edge => edge.Availability);
        if (visits.Sum() != simulationsCompleted)
        {
            throw new InvalidOperationException(
                $"SO-ISMCTS accounted for {visits.Sum()} of {simulationsCompleted} completed root visits.");
        }
        if (minimumAvailability != simulationsCompleted
            || maximumAvailability != simulationsCompleted)
        {
            throw new InvalidOperationException(
                $"SO-ISMCTS root availability was {minimumAvailability}..{maximumAvailability}, expected {simulationsCompleted}.");
        }

        return SearchMath.SelectMoveIndex(visits, valueSums, legalMoves);
    }

    private void Simulate(
        InformationSetNode root,
        SeededGameState state,
        List<Move> moves,
        SearchSettings settings)
    {
        var node = root;
        var path = new List<InformationSetEdge>();
        double leafValue;

        while (true)
        {
            if (state.GameEndState is not null)
            {
                leafValue = SearchMath.TerminalValue(state.GameEndState, root.RootPlayer);
                break;
            }
            if (node.Edges.Count == 0 || moves.Count == 0)
            {
                leafValue = node.PredictedValue;
                break;
            }

            var currentMoveMap = MoveKeyMap.Create(moves);
            var compatible = node.Edges.Values
                .Where(edge => currentMoveMap.MovesByKey.ContainsKey(edge.ActionKey))
                .ToArray();
            if (compatible.Length == 0)
            {
                leafValue = node.PredictedValue;
                break;
            }
            foreach (var edge in compatible)
            {
                edge.Availability++;
            }

            var selected = SelectEdge(node, compatible, settings.Exploration);
            var move = currentMoveMap.MovesByKey[selected.ActionKey];
            path.Add(selected);
            if (move.Command == CommandEnum.END_TURN)
            {
                var (afterEndTurn, _) = state.ApplyMove(move);
                if (afterEndTurn.GameEndState is not null)
                {
                    leafValue = SearchMath.TerminalValue(afterEndTurn.GameEndState, root.RootPlayer);
                }
                else
                {
                    leafValue = node.PredictedValue;
                }
                break;
            }
            var (nextState, nextMoves) = state.ApplyMove(move);
            DeterminizationSidecar.Validate(nextState);
            if (nextState.CurrentPlayer.PlayerID != root.RootPlayer)
            {
                leafValue = node.PredictedValue;
                break;
            }
            var observationKey = ObservationKey.From(nextState, nextMoves);
            if (selected.Children.TryGetValue(observationKey, out var existing))
            {
                node = existing;
            }
            else
            {
                PatronZeroModelOutput? evaluation = null;
                if (nextState.GameEndState is null && nextMoves.Count > 0)
                {
                    evaluation = evaluator.Evaluate(nextState, nextMoves);
                }
                node = CreateNode(
                    nextMoves,
                    root.RootPlayer,
                    evaluation,
                    settings,
                    isRoot: false,
                    countInitialAvailability: true,
                    terminalState: nextState.GameEndState,
                    statePlayer: nextState.CurrentPlayer.PlayerID);
                selected.Children.Add(observationKey, node);
                state = nextState;
                moves = nextMoves;
                leafValue = node.PredictedValue;
                break;
            }

            state = nextState;
            moves = nextMoves;
        }

        foreach (var edge in path)
        {
            edge.Visits++;
            edge.ValueSum += leafValue;
        }
    }

    private static InformationSetNode CreateNode(
        IReadOnlyList<Move> moves,
        PlayerEnum rootPlayer,
        PatronZeroModelOutput? evaluation,
        SearchSettings settings,
        bool isRoot,
        bool countInitialAvailability = false,
        EndGameState? terminalState = null,
        PlayerEnum? statePlayer = null)
    {
        double predictedValue;
        if (terminalState is not null)
        {
            predictedValue = SearchMath.TerminalValue(terminalState, rootPlayer);
        }
        else if (evaluation is null || moves.Count == 0)
        {
            predictedValue = 0;
        }
        else
        {
            predictedValue = SearchMath.ValueForRoot(
                evaluation.Value,
                statePlayer ?? rootPlayer,
                rootPlayer);
        }

        var node = new InformationSetNode(rootPlayer, predictedValue);
        if (terminalState is not null || evaluation is null || moves.Count == 0)
        {
            return node;
        }

        var moveMap = MoveKeyMap.Create(moves);
        var priors = SearchMath.Softmax(evaluation.Logits);
        foreach (var moveIndex in SearchMath.CandidateIndices(priors, moves, settings.MaxChildren, isRoot))
        {
            var actionKey = moveMap.Keys[moveIndex];
            node.Edges.Add(actionKey, new InformationSetEdge(
                actionKey,
                priors[moveIndex],
                countInitialAvailability ? 1 : 0));
        }
        NormalizePriors(node.Edges.Values);
        return node;
    }

    private static InformationSetEdge SelectEdge(
        InformationSetNode node,
        IReadOnlyList<InformationSetEdge> compatible,
        double exploration)
    {
        InformationSetEdge? best = null;
        var bestScore = double.NegativeInfinity;
        foreach (var edge in compatible)
        {
            var q = edge.Visits == 0 ? node.PredictedValue : edge.ValueSum / edge.Visits;
            var opportunityCount = Math.Max(1, edge.Availability);
            // Availability replaces parent visits in this neural PUCT analogue:
            // an action must not gain exploration pressure in worlds where it
            // was unavailable.
            var u = exploration * edge.Prior * Math.Sqrt(opportunityCount) / (1 + edge.Visits);
            var score = q + u;
            if (score > bestScore)
            {
                best = edge;
                bestScore = score;
            }
        }
        return best ?? compatible[0];
    }

    private static void NormalizePriors(IEnumerable<InformationSetEdge> edges)
    {
        var materialized = edges.ToArray();
        var sum = materialized.Sum(edge => edge.Prior);
        if (sum <= 0)
        {
            return;
        }
        foreach (var edge in materialized)
        {
            edge.Prior /= sum;
        }
    }

    private sealed class InformationSetNode(PlayerEnum rootPlayer, double predictedValue)
    {
        public PlayerEnum RootPlayer { get; } = rootPlayer;
        public double PredictedValue { get; } = predictedValue;
        public Dictionary<string, InformationSetEdge> Edges { get; } = new(StringComparer.Ordinal);
    }

    private sealed class InformationSetEdge(string actionKey, double prior, int availability)
    {
        public string ActionKey { get; } = actionKey;
        public double Prior { get; set; } = prior;
        public int Availability { get; set; } = availability;
        public int Visits { get; set; }
        public double ValueSum { get; set; }
        public Dictionary<ObservationKey, InformationSetNode> Children { get; } = [];
    }
}
