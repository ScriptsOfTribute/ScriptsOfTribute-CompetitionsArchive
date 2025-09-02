using Microsoft.ML;
using Microsoft.ML.Data;

namespace SimpleBots.src.MaltheMCTS.Utility.HeuristicScoring.ModelEvaluation
{
    public static class EnsembledTreeModelEvaluation
    {
        private static readonly MLContext mlContext = new MLContext();

        public static PredictionEngine<GameStateFeatureSetCsvRow, ModelOutput>? GetPredictionEngine()
        {
            string basePath = "MaltheMCTS/Ensemble_Tree_Models/";

            //switch (modelType)
            //{
            //    case RegressionTrainer.FastForest:
            //        var fastForestModel = mlContext.Model.Load(basePath + "FastForest", out var _);
            //        return mlContext.Model.CreatePredictionEngine<GameStateFeatureSetCsvRow, ModelOutput>(fastForestModel);
            //    case RegressionTrainer.FastTree:
            //        var fastTreeModel = mlContext.Model.Load(basePath + "FastTree", out var _);
            //        return mlContext.Model.CreatePredictionEngine<GameStateFeatureSetCsvRow, ModelOutput>(fastTreeModel);
            //    case RegressionTrainer.FastTreeTweedie:
            //        var fastTreeTweedieModel = mlContext.Model.Load(basePath + "FastTreeTweedie", out var _);
            //        return mlContext.Model.CreatePredictionEngine<GameStateFeatureSetCsvRow, ModelOutput>(fastTreeTweedieModel);
            //    case RegressionTrainer.LightGbm:
                    var lightGbmModel = mlContext.Model.Load(basePath + "LightGbm", out var _);
                    return mlContext.Model.CreatePredictionEngine<GameStateFeatureSetCsvRow, ModelOutput>(lightGbmModel);
            //    default:
            //        throw new ArgumentException("Unexpected model type: " + modelType);
            //}
        }

        //public static PredictionEngine<GameStateFeatureSetCsvRow, ModelOutput> GetPredictionEngine(string modelPath, RegressionTrainer modeltype)
        //{
        //    var model = mlContext.Model.Load(modelPath + "/" + modeltype + "_model", out var _);
        //    return mlContext.Model.CreatePredictionEngine<GameStateFeatureSetCsvRow, ModelOutput>(model);
        //}

        public class ModelOutput
        {
            [ColumnName("Score")]
            public float WinProbability { get; set; }
        }
    }
}
