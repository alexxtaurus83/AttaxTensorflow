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
                _logger.LogInformation("No best model set. Attempting to seed with the first available candidate model.");
                var firstCandidateDir = Directory.GetDirectories(modelsPath, "*_candidate")
                                                 .OrderBy(d => new DirectoryInfo(d).CreationTime)
                                                 .FirstOrDefault();

                if (firstCandidateDir != null) {
                    string seedCandidateName = new DirectoryInfo(firstCandidateDir).Name.Replace("_candidate", "");
                    string finalPath = Path.Combine(modelsPath, seedCandidateName);

                    if (Directory.Exists(finalPath)) Directory.Delete(finalPath, true);
                    Directory.Move(firstCandidateDir, finalPath);
                    
                    _logger.LogInformation("Found first candidate '{candidate}'. Promoting it to be the initial best model.", seedCandidateName);
                    _registry.PromoteModel(seedCandidateName);
                } else {
                    _logger.LogInformation("No candidate models found to seed the registry. Skipping evaluation.");
                }
                return; // End the job after seeding, the next run will perform an evaluation.
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

        _logger.LogInformation("viLoading models for evaluation match...");
        var bestPlayerService = new PredictionService(bestModelPath);
        var candidatePlayerService = new PredictionService(candidateModelPath);
        var bestPlayerEngine = new MctsPlayer(new MctsEngine(logic, bestPlayerService, evalOptions.Simulations));
        var candidatePlayerEngine = new MctsPlayer(new MctsEngine(logic, candidatePlayerService, evalOptions.Simulations));

        _logger.LogInformation("Starting {gameCount}-game evaluation match, parallelized.", evalOptions.Games);

        Parallel.For(0, evalOptions.Games, i => {
            // Alternate who plays red for fairness
            var redPlayer = (i % 2 == 0) ? candidatePlayerEngine : bestPlayerEngine;
            var bluePlayer = (i % 2 == 0) ? bestPlayerEngine : candidatePlayerEngine;

            var match = new MatchPlayer(logic, redPlayer, bluePlayer);
            float result = match.Play(); // 1.0 for Red win, -1.0 for Blue win

            // Convert result to candidate's perspective (1.0 = win, 0.0 = loss, 0.5 = draw)
            if (result == 0.0f) {
                results.Add(0.5);
            } else if ((redPlayer == candidatePlayerEngine && result > 0) || (bluePlayer == candidatePlayerEngine && result < 0)) {
                results.Add(1.0);
            } else {
                results.Add(0.0);
            }
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
    public int Games { get; set; } = 1; //100
    public int Simulations { get; set; } = 100;
}