using ScriptsOfTribute;
using ScriptsOfTribute.Serializers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MaltheMCTS
{
    /// <summary>
    /// Only used when SimulateMultipleTurns is disabled. It is a copy of this node, but representing the current score/visits of the node if end_turn is played, but without
    /// affecting the state with the card draws that happens on end_turn, since with this feature disabled, i do not want this to be part of the simulations.
    /// </summary>
    public class EndNode : Node
    {
        public EndNode(SeededGameState gameState, List<Move> possibleMoves, MaltheMCTS bot) : base(gameState, possibleMoves, bot)
        {
        }

        public override void Visit(out double score, HashSet<Node> visitedNodes)
        {
            score = Score();
            TotalScore += score;
            VisitCount++;
        }
    }
}
