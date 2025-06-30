using Ataxx.Controller;
using Ataxx.Core;
using FluentScheduler;
using System.Collections.Concurrent;
using Tensorflow;
using Tensorflow.Keras.Engine;
using TFN = Tensorflow.NumPy;
using static Tensorflow.Binding;
public class EvaluationJob : IJob {
    private readonly ILogger<EvaluationJob> _logger;
    private readonly IConfiguration _configuration;
    private readonly ModelRegistryService _registry;

    public EvaluationJob(ILogger<EvaluationJob> logger, IConfiguration configuration, ModelRegistryService registry) {
        _logger = logger;
        _configuration = configuration;
        _registry = registry;
    }

    public void Execute() {
        _logger.LogInformation("Starting scheduled evaluation job at: {time}", DateTimeOffset.Now);
        try {
            var modelsPath = _configuration["ModelManagement:ModelsPath"];
            var bestModelName = _registry.GetBestModelName();
            if (bestModelName == "initial") {
                _logger.LogInformation("No best model set yet. Skipping evaluation.");
                return;
            }

            var latestCandidateDir = Directory.GetDirectories(modelsPath, "*_candidate")
                                              .OrderByDescending(d => new DirectoryInfo(d).CreationTime)
                                              .FirstOrDefault();
            if (latestCandidateDir == null) {
                _logger.LogInformation("No new candidate models found to evaluate.");
                return;
            }

            string candidateName = new DirectoryInfo(latestCandidateDir).Name.Replace("_candidate", "");
            string bestModelPath = Path.Combine(modelsPath, bestModelName);

            _logger.LogInformation("Found candidate '{candidate}' to evaluate against best '{best}'.", candidateName, bestModelName);

            bool candidateIsBetter = RunEvaluationMatch(bestModelPath, latestCandidateDir);

            if (candidateIsBetter) {
                var finalPath = Path.Combine(modelsPath, candidateName);
                if (Directory.Exists(finalPath)) Directory.Delete(finalPath, true);
                Directory.Move(latestCandidateDir, finalPath);
                _logger.LogInformation("Renamed '{candidateDir}' to '{finalDir}'.", latestCandidateDir, finalPath);

                _logger.LogInformation("Candidate '{candidate}' won evaluation. Promoting to best.", candidateName);
                _registry.PromoteModel(candidateName);
            } else {
                _logger.LogInformation("Candidate '{candidate}' did not outperform best model. Deleting candidate.", candidateName);
                Directory.Delete(latestCandidateDir, recursive: true);
            }
        } catch (Exception ex) {
            _logger.LogError(ex, "An error occurred during the evaluation job.");
        } finally {
            tf.reset_default_graph();
            GC.Collect();
            GC.WaitForPendingFinalizers();
            _logger.LogInformation("Evaluation job finished, TensorFlow graph reset.");
        }
    }

    private bool RunEvaluationMatch(string bestModelPath, string candidateModelPath) {
        var evalOptions = _configuration.GetSection("Evaluation").Get<EvaluationOptions>() ?? new EvaluationOptions();
        var results = new ConcurrentBag<double>();
        var logic = new Ataxx.Core.AtaxxLogic();

        _logger.LogInformation("Loading models for evaluation match...");
        var bestPlayerService = new PredictionService(bestModelPath);
        var candidatePlayerService = new PredictionService(candidateModelPath);
        var bestPlayerEngine = new MctsEngine(logic, bestPlayerService, evalOptions.Simulations);
        var candidatePlayerEngine = new MctsEngine(logic, candidatePlayerService, evalOptions.Simulations);

        _logger.LogInformation("Starting {gameCount}-game evaluation match, parallelized.", evalOptions.Games);

        Parallel.For(0, evalOptions.Games, i => {
            var candidatePlayerColor = (i % 2 == 0) ? Ataxx.Core.AtaxxLogic.PlayerColor.Red : Ataxx.Core.AtaxxLogic.PlayerColor.Blue;
            var bestPlayerColor = logic.SwitchPlayer(candidatePlayerColor);

            var board = new Ataxx.Core.BitboardState();
            board.RedPieces = (1UL << logic.GetBitIndex(0, 0)) | (1UL << logic.GetBitIndex(6, 6));
            board.BluePieces = (1UL << logic.GetBitIndex(0, 6)) | (1UL << logic.GetBitIndex(6, 0));

            var currentPlayer = Ataxx.Core.AtaxxLogic.PlayerColor.Red;

            while (!logic.IsGameOver(board)) {
                var engineToUse = (currentPlayer == candidatePlayerColor) ? candidatePlayerEngine : bestPlayerEngine;
                var validMoves = logic.GetAllValidMoves(board, currentPlayer);
                if (validMoves.Count == 0) {
                    currentPlayer = logic.SwitchPlayer(currentPlayer);
                    continue;
                }
                var move = engineToUse.FindBestMove(board, currentPlayer).bestMove;
                board = logic.MakeMove(board, move, currentPlayer, false);
                currentPlayer = logic.SwitchPlayer(currentPlayer);
            }

            int redCount = logic.PopCount(board.RedPieces);
            int blueCount = logic.PopCount(board.BluePieces);

            // CORRECTED: Fixed the typo from 'blue.Count' to 'blueCount'.
            if (redCount == blueCount) { results.Add(0.5); return; }

            if ((redCount > blueCount && candidatePlayerColor == Ataxx.Core.AtaxxLogic.PlayerColor.Red) ||
                (blueCount > redCount && candidatePlayerColor == Ataxx.Core.AtaxxLogic.PlayerColor.Blue)) { results.Add(1.0); } else { results.Add(0.0); }
        });

        int totalGames = results.Count(r => r != 0.5); // Exclude draws for win-rate calculation
        if (totalGames == 0) return false; // No decisive games
        double candidateWins = results.Count(r => r == 1.0);
        double winRate = candidateWins / totalGames;

        double z = 1.96;
        double n = totalGames;
        double phat = winRate;
        double term2 = z * Math.Sqrt((phat * (1 - phat) + z * z / (4 * n)) / n);
        double term1 = phat + z * z / (2 * n);
        double lowerBound = (term1 - term2) / (1 + z * z / n);

        _logger.LogInformation("Match complete. Win Rate: {rate:P1}. 95% CI lower bound: {lower:P1}", winRate, lowerBound);

        return lowerBound > 0.50;
    }
}
public class EvaluationOptions {
    public int Games { get; set; } = 100;
    public int Simulations { get; set; } = 100;
}