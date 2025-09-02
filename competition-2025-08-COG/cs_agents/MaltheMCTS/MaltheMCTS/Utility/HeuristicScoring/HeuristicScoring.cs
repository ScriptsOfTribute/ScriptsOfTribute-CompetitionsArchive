using Microsoft.ML;
using ScriptsOfTribute;
using ScriptsOfTribute.Board.Cards;
using ScriptsOfTribute.Serializers;
using SimpleBots.src.MaltheMCTS.Utility.HeuristicScoring.ModelEvaluation;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static SimpleBots.src.MaltheMCTS.Utility.HeuristicScoring.ModelEvaluation.EnsembledTreeModelEvaluation;

namespace SimpleBots.src.MaltheMCTS.Utility.HeuristicScoring
{
    public static class HeuristicScoring
    {
        private const double BASE_AGENT_STRENGTH_MULTIPLIER = 1;
        private const double AGENT_HP_VALUE_MULTIPLIER = 0.1;
        private const double CHOICE_WEIGHT = 0.75;

        /// <summary>
        /// To lower the amount of variables (hand strengths, patron moves available, coins, power, whether agents have been activated) the model needs to process,
        /// this model scores states just before ending turn
        /// </summary>
        public static double Score(SeededGameState gameState, PredictionEngine<GameStateFeatureSetCsvRow, ModelOutput> predictionEngine, bool endOfTurnExclusive = true)
        {
            // The manual model (null) does not return either 0 and 1 or -1 and 1, so this logic does not apply for it
            if (predictionEngine != null)
            {
                var winner = CheckWinner(gameState, endOfTurnExclusive);

                if (winner == gameState.CurrentPlayer.PlayerID)
                {
                    return 1;
                }
                else if (winner == gameState.EnemyPlayer.PlayerID)
                {
                    return 0; // Consider if i should measure score as win probability (0-1) or zero sum score
                }
            }

            var featureSet = FeatureSetUtility.BuildFeatureSet(gameState);

            return ModelEvaluation(featureSet, predictionEngine);
        }

        private static PlayerEnum CheckWinner(SeededGameState gameState, bool endOfTurnExclusiveEvaluation)
        {
            int currentPlayerPrestige = gameState.CurrentPlayer.Prestige;
            int opponentPrestige = gameState.EnemyPlayer.Prestige;

            if (currentPlayerPrestige >= 80)
            {
                return gameState.CurrentPlayer.PlayerID;
            }

            int opponentTaunt = gameState.EnemyPlayer.Agents.Where(a => a.RepresentingCard.Taunt).Sum(a => a.CurrentHp);
            int power = gameState.CurrentPlayer.Power;

            if (power - opponentTaunt + currentPlayerPrestige >= 80)
            {
                return gameState.CurrentPlayer.PlayerID;
            }

            if (endOfTurnExclusiveEvaluation && opponentPrestige >= 40 && opponentPrestige > currentPlayerPrestige)
            {
                return gameState.EnemyPlayer.PlayerID;
            }


            int patronCount = 0;

            foreach (var patron in gameState.PatronStates.All)
            {
                if (patron.Value == gameState.CurrentPlayer.PlayerID)
                {
                    patronCount++;
                }
            }

            if (patronCount >= 4)
            {
                return gameState.CurrentPlayer.PlayerID;
            }

            return PlayerEnum.NO_PLAYER_SELECTED;
        }

        private static double ModelEvaluation(GameStateFeatureSet featureSet, PredictionEngine<GameStateFeatureSetCsvRow, ModelOutput> predictionEngine)
        {
            if (predictionEngine == null)
            {
                return SimpleManualEvaluation.Evaluate(featureSet);
            }
            else
            {
                var csvFeatureSet = featureSet.ToCsvRow();
                return predictionEngine.Predict(csvFeatureSet).WinProbability;
            }
        }
    }
}