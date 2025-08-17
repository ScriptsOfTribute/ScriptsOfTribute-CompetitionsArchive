using System.Linq;
using System.Xml.Schema;
using ScriptsOfTribute;
using ScriptsOfTribute.Serializers;

namespace MaltheMCTS;

public class ChanceNode : Node
{
    public Node Parent;
    public Move AppliedMove;
    private Dictionary<Node, int> knownOutcomesWithVisitCount;
    public ChanceNode(SeededGameState gameState, Node parent, Move appliedMove, MaltheMCTS bot) : base(gameState, new List<Move>(), bot)
    {
        AppliedMove = appliedMove;
        if (bot.Settings.CHANCE_NODE_BRANCH_LIMIT != null)
        {
            knownOutcomesWithVisitCount = new();
        }
        Parent = parent;
    }

    public override void Visit(out double score, HashSet<Node> visitedNodes)
    {
        Node child;

        if (Bot.Settings.CHANCE_NODE_BRANCH_LIMIT != null)
        {
            if (knownOutcomesWithVisitCount.Count >= Bot.Settings.CHANCE_NODE_BRANCH_LIMIT.Value)
            {
                child = knownOutcomesWithVisitCount.MinBy(x => x.Value).Key;
            }
            else
            {
                (var newState, var newMoves) = Parent.GameState.ApplyMove(AppliedMove, (ulong)Utility.Rng.Next());
                child = Utility.FindOrBuildNode(newState, this, newMoves, Bot);
                var existingOutcome = knownOutcomesWithVisitCount.FirstOrDefault(x => x.Key.GameStateHash == child.GameStateHash && x.Key.GameState.IsIdentical(child.GameState)).Key;
                if (existingOutcome != null)
                {
                    child = existingOutcome;
                }
                else
                {
                    knownOutcomesWithVisitCount.Add(child, 0);
                }
            }
        }
        else
        {
            (var newState, var newMoves) = Parent.GameState.ApplyMove(AppliedMove, (ulong)Utility.Rng.Next());
            child = Utility.FindOrBuildNode(newState, this, newMoves, Bot);
        }

        child.Visit(out score, visitedNodes);

        if (Bot.Settings.CHANCE_NODE_BRANCH_LIMIT != null)
        {
            knownOutcomesWithVisitCount[child]++;
        }

        TotalScore += score;
        VisitCount++;
    }
}
