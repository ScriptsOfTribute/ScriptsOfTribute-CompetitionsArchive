using System.Diagnostics;
using Bots;
using Microsoft.ML;
using ScriptsOfTribute;
using ScriptsOfTribute.AI;
using ScriptsOfTribute.Board;
using ScriptsOfTribute.Serializers;
using SimpleBots.src.MaltheMCTS.Utility.HeuristicScoring;
using SimpleBots.src.MaltheMCTS.Utility.HeuristicScoring.ModelEvaluation;
using static SimpleBots.src.MaltheMCTS.Utility.HeuristicScoring.ModelEvaluation.EnsembledTreeModelEvaluation;

namespace MaltheMCTS;

public class MaltheMCTS : AI
{
    public Dictionary<int, List<Node>> NodeGameStateHashMap = new Dictionary<int, List<Node>>();
    public Settings Settings { get; set; }

    public PredictionEngine<GameStateFeatureSetCsvRow, ModelOutput> PredictionEngine;

    public string InstanceName;

    public MaltheMCTS(string? instanceName = null, Settings? settings = null) : base()
    {
        this.InstanceName = instanceName ?? "MaltheMCTS_" + Guid.NewGuid();
        Settings = settings ?? new Settings(); // Hardcoded
        PredictionEngine = EnsembledTreeModelEvaluation.GetPredictionEngine();
    }

    public MaltheMCTS() : base()
    {
        InstanceName = "MaltheMCTS_" + Guid.NewGuid();
        Settings = new Settings(); // Hardcoded
        PredictionEngine = EnsembledTreeModelEvaluation.GetPredictionEngine();
        Utility.CategorizeCards();
    }

    public override void PregamePrepare()
    {
        NodeGameStateHashMap = new Dictionary<int, List<Node>>();
    }

    public override void GameEnd(EndGameState state, FullGameState? finalBoardState)
    {
        //Console.WriteLine("@@@ Game ended because of " + state.Reason + " @@@");
        //Console.WriteLine("@@@ Winner was " + state.Winner + " @@@");
        //Console.WriteLine("Patrons were:");
        //finalBoardState?.Patrons.ForEach(p => Console.WriteLine(p));
    }

    public override Move Play(GameState gameState, List<Move> possibleMoves, TimeSpan remainingTime)
    {
        try
        {
            var instantPlay = FindInstantPlayMove(possibleMoves);
            if (instantPlay != null)
            {
                return instantPlay;
            }

            if (possibleMoves.Count == 1)
            {
                return possibleMoves[0];
            }

            ulong randomSeed = (ulong)Utility.Rng.Next();
            var seededGameState = gameState.ToSeededGameState(randomSeed);

            var rootNode = Utility.FindOrBuildNode(seededGameState, null, possibleMoves, this);

            var moveTimer = new Stopwatch();
            moveTimer.Start();
            int estimatedRemainingMovesInTurn = EstimateRemainingMovesInTurn(gameState, possibleMoves);
            double millisecondsForMove = (remainingTime.TotalMilliseconds / estimatedRemainingMovesInTurn) - Settings.ITERATION_COMPLETION_MILLISECONDS_BUFFER;
            while (moveTimer.ElapsedMilliseconds < millisecondsForMove)
            {
                rootNode.Visit(out double score, new HashSet<Node>());
            }

            if (rootNode.MoveToChildNode.Count == 0)
            {
                // No time for calculating move
                return possibleMoves.PickRandom(new SeededRandom());
            }

            var bestMove = rootNode.MoveToChildNode
                .OrderByDescending(moveNodePair => (moveNodePair.Value.Child.TotalScore / moveNodePair.Value.VisitCount))
                .FirstOrDefault()
                .Key;

            return Utility.FindOfficialMove(bestMove, possibleMoves);
        }
        catch
        {
            // Something went wrong while trying to compute move. Playing random move instead.
            return possibleMoves.PickRandom(new SeededRandom());
        }
    }

    private void SaveErrorLog(string errorMessage)
    {
        var filePath = InstanceName + "_Error.txt";

        string directoryPath = Path.GetDirectoryName(filePath);
        if (!string.IsNullOrEmpty(directoryPath))
        {
            Directory.CreateDirectory(directoryPath);
        }

        using (var writer = new StreamWriter(filePath, true))
        {
            writer.Write("\n");
            writer.Write(errorMessage);
        }
    }

    private bool CheckMoveLegality(Move moveToCheck, Node rootNode, GameState officialGameState, List<Move> officialPossiblemoves)
    {
        if (!officialPossiblemoves.Any(move => move.IsIdentical(moveToCheck)))
        {
            Console.WriteLine("----- ABOUT TO PERFORM ILLEGAL MOVE -----");
            Console.WriteLine("Our state:");
            rootNode?.GameState.Log();
            Console.WriteLine("Actual state:");
            officialGameState.ToSeededGameState((ulong)Utility.Rng.Next()).Log();
            Console.WriteLine("@@@@ Trying to play move:");
            moveToCheck.Log();
            Console.WriteLine("@@@@@@@ But available moves were:");
            officialPossiblemoves.ForEach(m => m.Log());
            Console.WriteLine("@@@@@@ But we thought moves were:");
            rootNode.PossibleMoves.ForEach(m => m.Log());

            return false;
        }

        return true;
    }

    private int EstimateRemainingMovesInTurn(GameState inputState, List<Move> inputPossibleMoves)
    {
        return EstimateRemainingMovesInTurn(inputState.ToSeededGameState((ulong)Utility.Rng.Next()), inputPossibleMoves);
    }

    private int EstimateRemainingMovesInTurn(SeededGameState inputState, List<Move> inputPossibleMoves)
    {

        var possibleMoves = new List<Move>(inputPossibleMoves);

        if (possibleMoves.Count == 1 && possibleMoves[0].Command == CommandEnum.END_TURN)
        {
            return 0;
        }

        possibleMoves.RemoveAll(x => x.Command == CommandEnum.END_TURN);

        int result = 1;
        SeededGameState currentState = inputState;
        List<Move> currentPossibleMoves = possibleMoves;

        while (currentPossibleMoves.Count > 0)
        {

            var instantPlay = FindInstantPlayMove(currentPossibleMoves);
            if (instantPlay != null)
            {
                (currentState, currentPossibleMoves) = currentState.ApplyMove(instantPlay);
            }
            else if (currentPossibleMoves.Count == 1)
            {
                (currentState, currentPossibleMoves) = currentState.ApplyMove(currentPossibleMoves[0]);
            }
            else
            {
                result++;
                (currentState, currentPossibleMoves) = currentState.ApplyMove(currentPossibleMoves[0]);
            }

            currentPossibleMoves.RemoveAll(x => x.Command == CommandEnum.END_TURN);
        }

        return result;
    }

    private Move FindInstantPlayMove(List<Move> possibleMoves)
    {
        if (possibleMoves.Count == 1)
        {
            // This can be different than "END_TURN" in cases where a choice needs to be made (between agents for example)
            // while there is only one agent available.
            return possibleMoves[0];
        }

        foreach (Move currMove in possibleMoves)
        {
            if (currMove.IsInstantPlay()) {
                return currMove;
            }
        }

        return null;
    }

    public override PatronId SelectPatron(List<PatronId> availablePatrons, int round)
    {
        if (availablePatrons.Contains(PatronId.HLAALU))
        {
            return PatronId.HLAALU;
        }
        if (availablePatrons.Contains(PatronId.DUKE_OF_CROWS))
        {
            return PatronId.DUKE_OF_CROWS;
        }
        if (availablePatrons.Contains(PatronId.ANSEI))
        {
            return PatronId.ANSEI;
        }
        if (availablePatrons.Contains(PatronId.RAJHIN))
        {
            return PatronId.RAJHIN;
        }
        return availablePatrons.PickRandom(new SeededRandom());
    }
}
