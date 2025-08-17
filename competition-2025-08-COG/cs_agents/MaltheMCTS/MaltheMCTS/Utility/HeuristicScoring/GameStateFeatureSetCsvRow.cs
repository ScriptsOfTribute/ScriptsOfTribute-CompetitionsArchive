using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleBots.src.MaltheMCTS.Utility.HeuristicScoring
{
    /// <summary>
    /// I had to turn everything into float, since Auto.ML.Net kept crashing on all other types, including Ints, with some non-interpretable error logs
    /// </summary>
    public class GameStateFeatureSetCsvRow
    {
        [LoadColumn(0)] public float CurrentPlayerPrestige { get; set; }
        [LoadColumn(1)] public float CurrentPlayerDeck_PrestigeStrength { get; set; }
        [LoadColumn(2)] public float CurrentPlayerDeck_PowerStrength { get; set; }
        [LoadColumn(3)] public float CurrentPlayerDeck_GoldStrength { get; set; }
        [LoadColumn(4)] public float CurrentPlayerDeck_MiscStrength { get; set; }
        [LoadColumn(5)] public float CurrentPlayerDeckComboProportion { get; set; }
        [LoadColumn(6)] public float CurrentPlayerAgent_PrestigeStrength { get; set; }
        [LoadColumn(7)] public float CurrentPlayerAgent_PowerStrength { get; set; }
        [LoadColumn(8)] public float CurrentPlayerAgent_GoldStrength { get; set; }
        [LoadColumn(9)] public float CurrentPlayerAgent_MiscStrength { get; set; }
        [LoadColumn(10)] public float CurrentPlayerPatronFavour { get; set; }
        [LoadColumn(11)] public float OpponentPrestige { get; set; }
        [LoadColumn(12)] public float OpponentDeck_PrestigeStrength { get; set; }
        [LoadColumn(13)] public float OpponentDeck_PowerStrength { get; set; }
        [LoadColumn(14)] public float OpponentDeck_GoldStrength { get; set; }
        [LoadColumn(15)] public float OpponentDeck_MiscStrength { get; set; }
        [LoadColumn(16)] public float OpponentAgent_PrestigeStrength { get; set; }
        [LoadColumn(17)] public float OpponentAgent_PowerStrength { get; set; }
        [LoadColumn(18)] public float OpponentAgent_GoldStrength { get; set; }
        [LoadColumn(19)] public float OpponentAgent_MiscStrength { get; set; }
        [LoadColumn(20)] public float OpponentPatronFavour { get; set; }
        //[LoadColumn(21)] public float WinProbability { get; set; }
    }
}
