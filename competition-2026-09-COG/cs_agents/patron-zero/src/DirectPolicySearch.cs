using ScriptsOfTribute;
using ScriptsOfTribute.AI;
using ScriptsOfTribute.Board;
using ScriptsOfTribute.Serializers;

namespace PatronZeroSubmission;

internal sealed class DirectPolicySearch(ISearchEvaluator evaluator) : INeuralSearch
{
    public int Search(
        GameState visibleState,
        IReadOnlyList<Move> legalMoves,
        ulong baseSeed,
        int decisionIndex,
        SearchSettings settings)
    {
        _ = baseSeed;
        _ = decisionIndex;
        _ = settings;
        var output = evaluator.Evaluate(visibleState, legalMoves);
        if (legalMoves.Count == 0
            || output.Logits.Length != legalMoves.Count
            || output.Logits.Any(logit => !float.IsFinite(logit)))
        {
            throw new InvalidDataException("The policy output must contain one finite logit per legal move.");
        }
        var keys = MoveKeyMap.Create(legalMoves).Keys;
        var selected = Enumerable.Range(0, legalMoves.Count)
            .OrderByDescending(index => output.Logits[index])
            .ThenBy(index => keys[index], StringComparer.Ordinal)
            .First();
        return selected;
    }
}
