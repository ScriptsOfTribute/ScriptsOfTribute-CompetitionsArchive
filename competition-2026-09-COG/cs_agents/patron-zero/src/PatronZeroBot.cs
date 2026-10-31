using ScriptsOfTribute;
using ScriptsOfTribute.AI;
using ScriptsOfTribute.Board;
using ScriptsOfTribute.Board.CardAction;
using ScriptsOfTribute.Serializers;

namespace PatronZeroSubmission;

/// <summary>
/// Competition entry point. The default mode is the proven direct policy.
/// </summary>
public sealed class PatronZeroBot : AI, IDisposable
{
    private readonly PatronZeroModelEvaluator _model;
    private readonly INeuralSearch _search;
    private readonly SearchSettings _settings;
    private int _decisionIndex;

    public PatronZeroBot()
    {
        var assemblyDirectory = Path.GetDirectoryName(typeof(PatronZeroBot).Assembly.Location)
            ?? throw new InvalidOperationException("Could not locate the PatronZero assembly directory.");
        var modelPath = Path.Combine(assemblyDirectory, "model", "model.pt");
        _model = new PatronZeroModelEvaluator(modelPath);

        _search = new DirectPolicySearch(_model);
        // _search = new PcPimcSearch(_model);
        // _search = new SoIsmctsSearch(_model);

        var decisionMilliseconds = _search is DirectPolicySearch
            ? 0
            : 100;
        _settings = new SearchSettings(
            Simulations: 100_000,
            Worlds: 4,
            MaxChildren: 16,
            Exploration: 1.5,
            DecisionMilliseconds: decisionMilliseconds);
    }

    public override PatronId SelectPatron(List<PatronId> availablePatrons, int round)
        => availablePatrons[round % availablePatrons.Count];

    public override Move Play(GameState gameState, List<Move> possibleMoves, TimeSpan remainingTime)
    {
        if (possibleMoves.Count == 1)
        {
            return possibleMoves[0];
        }

        var effectiveSettings = _settings.DecisionMilliseconds <= 0
            ? _settings
            : _settings with
            {
                DecisionMilliseconds = Math.Max(
                    1,
                    Math.Min(_settings.DecisionMilliseconds, remainingTime.TotalMilliseconds - 100)),
            };
        try
        {
            var selectedMoveIndex = _search.Search(
                gameState,
                possibleMoves,
                baseSeed: 23_001,
                decisionIndex: _decisionIndex++,
                effectiveSettings);
            return possibleMoves[selectedMoveIndex];
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine($"PatronZeroBot failed while selecting a move: {exception}");
            throw;
        }
    }

    public override void GameEnd(EndGameState state, FullGameState? finalBoardState) { }

    public void Dispose() => _model.Dispose();
}
