using System.Diagnostics;
using ScriptsOfTribute;
using ScriptsOfTribute.Board;
using ScriptsOfTribute.Serializers;

namespace PatronZeroSubmission;

internal sealed class PcPimcSearch(ISearchEvaluator evaluator) : INeuralSearch
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
        var combinedPolicy = new double[legalMoves.Count];
        var worlds = Math.Min(settings.Worlds, settings.Simulations);
        var roots = new List<PerfectInformationNode>(worlds);

        for (var worldIndex = 0; worldIndex < worlds; worldIndex++)
        {
            var worldSeed = SearchMath.SimulationSeed(baseSeed, decisionIndex, worldIndex);
            var sampledState = DeterminizationSidecar.Create(visibleState, worldSeed);
            if (!visibleRootKey.Equals(ObservationKey.From(sampledState, legalMoves)))
            {
                throw new InvalidOperationException("PC-PIMC sampled root changed the acting player's fair observation.");
            }
            var root = new PerfectInformationNode(
                sampledState,
                legalMoves.ToList(),
                rootPlayer);
            Expand(root, settings, rootEvaluation, isRoot: true);
            roots.Add(root);
        }

        var simulationsCompleted = 0;
        while (simulationsCompleted < settings.Simulations
               && (simulationsCompleted == 0
                   || settings.DecisionMilliseconds <= 0
                   || watch.Elapsed.TotalMilliseconds < settings.DecisionMilliseconds))
        {
            Simulate(roots[simulationsCompleted % roots.Count], settings);
            simulationsCompleted++;
        }

        var accountedVisits = roots.Sum(root => root.Edges.Sum(edge => edge.Visits));
        if (accountedVisits != simulationsCompleted)
        {
            throw new InvalidOperationException(
                $"PC-PIMC accounted for {accountedVisits} of {simulationsCompleted} simulations.");
        }

        foreach (var root in roots)
        {
            var rootVisits = root.Edges.Sum(edge => edge.Visits);
            if (rootVisits == 0)
            {
                continue;
            }
            foreach (var edge in root.Edges)
            {
                // PC-PIMC combines policies, so every sampled world contributes
                // a normalized root distribution rather than unscaled values.
                combinedPolicy[edge.RootMoveIndex] += edge.Visits / (double)rootVisits;
            }
        }

        return SearchMath.SelectPolicyMove(combinedPolicy, legalMoves);
    }

    private void Simulate(PerfectInformationNode root, SearchSettings settings)
    {
        var path = new List<PerfectInformationEdge>();
        var node = root;
        double leafValue;

        while (true)
        {
            if (node.State.GameEndState is not null)
            {
                leafValue = SearchMath.TerminalValue(node.State.GameEndState, node.RootPlayer);
                break;
            }
            if (!node.Expanded)
            {
                leafValue = Expand(node, settings, overrideEvaluation: null, isRoot: false);
                break;
            }
            if (node.Edges.Count == 0)
            {
                leafValue = node.PredictedValue;
                break;
            }

            var edge = SelectEdge(node, settings.Exploration);
            path.Add(edge);
            if (edge.Move.Command == CommandEnum.END_TURN)
            {
                var (afterEndTurn, _) = node.State.ApplyMove(edge.Move);
                if (afterEndTurn.GameEndState is not null)
                {
                    leafValue = SearchMath.TerminalValue(afterEndTurn.GameEndState, node.RootPlayer);
                }
                else
                {
                    // Do not inspect or evaluate the opponent's private state.
                    // END_TURN is the horizon edge, so use the pre-END node value.
                    leafValue = node.PredictedValue;
                }
                break;
            }
            if (edge.Child is null)
            {
                var (nextState, nextMoves) = node.State.ApplyMove(edge.Move);
                DeterminizationSidecar.Validate(nextState);
                if (nextState.CurrentPlayer.PlayerID != node.RootPlayer)
                {
                    leafValue = node.PredictedValue;
                    break;
                }
                edge.Child = new PerfectInformationNode(nextState, nextMoves, node.RootPlayer);
            }
            node = edge.Child;
        }

        if (path.Count == 0 || !ReferenceEquals(path[^1].Parent, node))
        {
            node.Visits++;
        }
        foreach (var edge in path)
        {
            edge.Parent.Visits++;
            edge.Visits++;
            edge.ValueSum += leafValue;
        }
    }

    private double Expand(
        PerfectInformationNode node,
        SearchSettings settings,
        PatronZeroModelOutput? overrideEvaluation,
        bool isRoot)
    {
        if (node.State.GameEndState is not null)
        {
            node.PredictedValue = SearchMath.TerminalValue(node.State.GameEndState, node.RootPlayer);
            node.Expanded = true;
            return node.PredictedValue;
        }
        if (node.Moves.Count == 0)
        {
            node.PredictedValue = 0;
            node.Expanded = true;
            return 0;
        }

        var evaluation = overrideEvaluation ?? evaluator.Evaluate(node.State, node.Moves);
        node.PredictedValue = SearchMath.ValueForRoot(
            evaluation.Value,
            node.State.CurrentPlayer.PlayerID,
            node.RootPlayer);
        var priors = SearchMath.Softmax(evaluation.Logits);
        foreach (var moveIndex in SearchMath.CandidateIndices(priors, node.Moves, settings.MaxChildren, isRoot))
        {
            node.Edges.Add(new PerfectInformationEdge(
                node,
                node.Moves[moveIndex],
                isRoot ? moveIndex : -1,
                priors[moveIndex]));
        }
        NormalizePriors(node.Edges);
        node.Expanded = true;
        return node.PredictedValue;
    }

    private static PerfectInformationEdge SelectEdge(PerfectInformationNode node, double exploration)
    {
        var sqrtVisits = Math.Sqrt(node.Visits + 1);
        PerfectInformationEdge? best = null;
        var bestScore = double.NegativeInfinity;
        foreach (var edge in node.Edges)
        {
            var q = edge.Visits == 0 ? node.PredictedValue : edge.ValueSum / edge.Visits;
            var u = exploration * edge.Prior * sqrtVisits / (1 + edge.Visits);
            var score = q + u;
            if (score > bestScore)
            {
                best = edge;
                bestScore = score;
            }
        }
        return best ?? node.Edges[0];
    }

    private static void NormalizePriors(IReadOnlyList<PerfectInformationEdge> edges)
    {
        var sum = edges.Sum(edge => edge.Prior);
        if (sum <= 0)
        {
            return;
        }
        foreach (var edge in edges)
        {
            edge.Prior /= sum;
        }
    }

    private sealed class PerfectInformationNode(
        SeededGameState state,
        List<Move> moves,
        PlayerEnum rootPlayer)
    {
        public SeededGameState State { get; } = state;
        public List<Move> Moves { get; } = moves;
        public PlayerEnum RootPlayer { get; } = rootPlayer;
        public List<PerfectInformationEdge> Edges { get; } = [];
        public bool Expanded { get; set; }
        public int Visits { get; set; }
        public double PredictedValue { get; set; }
    }

    private sealed class PerfectInformationEdge(
        PerfectInformationNode parent,
        Move move,
        int rootMoveIndex,
        double prior)
    {
        public PerfectInformationNode Parent { get; } = parent;
        public Move Move { get; } = move;
        public int RootMoveIndex { get; } = rootMoveIndex;
        public double Prior { get; set; } = prior;
        public int Visits { get; set; }
        public double ValueSum { get; set; }
        public PerfectInformationNode? Child { get; set; }
    }
}
