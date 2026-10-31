using ScriptsOfTribute;
using ScriptsOfTribute.Board;
using ScriptsOfTribute.Serializers;
using System.Text.Json;
using TorchSharp;
using static TorchSharp.torch;

namespace PatronZeroSubmission;

internal sealed record PatronZeroModelOutput(float[] Logits, float Value);

internal sealed class PatronZeroModelEvaluator : ISearchEvaluator, IDisposable
{
    private readonly string _modelPath;
    private readonly bool _choiceTokensEnabled;
    private torch.jit.ScriptModule? _model;
    private bool _loadAttempted;

    public PatronZeroModelEvaluator(string modelPath)
    {
        _modelPath = modelPath;
        _choiceTokensEnabled = UsesChoiceTokens(modelPath);
    }

    public PatronZeroModelOutput Evaluate(GameState state, IReadOnlyList<Move> moves)
        => Evaluate(PatronZeroFeatures.EncodeState(state), moves);

    public PatronZeroModelOutput Evaluate(SeededGameState state, IReadOnlyList<Move> moves)
        => Evaluate(PatronZeroFeatures.EncodeState(state), moves);

    private PatronZeroModelOutput Evaluate(float[] stateValues, IReadOnlyList<Move> moves)
    {
        if (moves.Count == 0)
        {
            throw new ArgumentException("Cannot evaluate an empty legal-move list.", nameof(moves));
        }
        EnsureModel();

        var moveValues = PatronZeroFeatures.EncodeMoves(moves, _choiceTokensEnabled);
        var moveDim = PatronZeroFeatures.MoveDim(_choiceTokensEnabled);

        using var state = torch.tensor(
            stateValues,
            [1, PatronZeroFeatures.StateDim],
            dtype: ScalarType.Float32);
        using var moveTensor = torch.tensor(
            moveValues,
            [1, moves.Count, moveDim],
            dtype: ScalarType.Float32);
        using var mask = torch.ones(
            [1, moves.Count],
            dtype: ScalarType.Float32);

        var result = _model!.forward(state, moveTensor, mask);
        var (logits, value) = Unpack(result);
        using (logits)
        using (value)
        {
            var output = new PatronZeroModelOutput(
                logits.cpu().data<float>().ToArray(),
                value.cpu().data<float>().ToArray()[0]);
            if (output.Logits.Length != moves.Count
                || output.Logits.Any(logit => !float.IsFinite(logit))
                || !float.IsFinite(output.Value))
            {
                throw new InvalidDataException("The model returned invalid policy/value output.");
            }
            return output;
        }
    }

    public void Dispose()
    {
        _model?.Dispose();
        _model = null;
    }

    private void EnsureModel()
    {
        if (_model is not null)
        {
            return;
        }
        if (_loadAttempted)
        {
            throw new InvalidOperationException($"PatronZero model could not be loaded from '{_modelPath}'.");
        }

        _loadAttempted = true;
        if (!File.Exists(_modelPath))
        {
            throw new FileNotFoundException("PatronZero model not found.", _modelPath);
        }

        torch.set_num_threads(1);
        _model = torch.jit.load(_modelPath);
        _model.eval();
    }

    private static bool UsesChoiceTokens(string modelPath)
    {
        var configPath = Path.Combine(Path.GetDirectoryName(modelPath) ?? string.Empty, "model.json");
        if (!File.Exists(configPath))
        {
            return false;
        }
        using var document = JsonDocument.Parse(File.ReadAllText(configPath));
        return document.RootElement.TryGetProperty("model_version", out var version)
            && version.GetString() == "patron-zero-imitation-v3";
    }

    private static (Tensor logits, Tensor value) Unpack(object? result)
        => result switch
        {
            ValueTuple<Tensor, Tensor> tuple => tuple,
            Tuple<Tensor, Tensor> tuple => (tuple.Item1, tuple.Item2),
            object[] { Length: 2 } values when values[0] is Tensor logits && values[1] is Tensor value => (logits, value),
            _ => throw new InvalidOperationException($"Unexpected model output: {result?.GetType().FullName ?? "<null>"}.")
        };
}
