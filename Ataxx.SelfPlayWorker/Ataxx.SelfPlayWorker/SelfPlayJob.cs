using Ataxx.Core;
using FluentScheduler;
using Newtonsoft.Json;
using RestSharp;

namespace Ataxx.SelfPlayWorker {
    public class SelfPlayJob : IJob {
        private readonly Program.Options _options;
        private readonly AtaxxLogic _logic;
        private static readonly object _fileLock = new object();
        public SelfPlayJob(Program.Options options) {
            _options = options;
            _logic = new AtaxxLogic();
        }       

        public void Execute()
        {
            Console.WriteLine($"({DateTime.Now:T}) Starting new self-play session in '{_options.Mode}' mode (Backend: {_options.Backend})...");
            var sessionStopwatch = System.Diagnostics.Stopwatch.StartNew();
            
            PredictionService? predictionService = null;
            if (_options.Mode == PlayMode.Mcts)
            {
                try
                {
                    Console.WriteLine("Loading model for MCTS...");
                    var client = new RestClient(_options.ControllerUrl);
                    var request = new RestRequest("/api/model/latest");
                    var bestModelName = client.Get<BestModelResponse>(request)?.BestModelName;

                    if (string.IsNullOrEmpty(bestModelName) || bestModelName == "initial")
                    {
                        Console.WriteLine("No best model available from controller. Cannot run MCTS games.");
                        return;
                    }

                    string fullModelPath = Path.Combine(_options.ModelPath!, bestModelName);
                    predictionService = new PredictionService(fullModelPath);
                    Console.WriteLine($"Model '{bestModelName}' loaded successfully.");
                }
                catch (Exception ex)
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine($"\nFailed to load model. Is Controller/Share accessible? Error: {ex.Message}");
                    Console.ResetColor();
                    return;
                }
            }

            int completedGames = 0;
            int parallelismLevel = _options.Backend.ToLower() == "cpu" ? _options.ParallelGames : 1;
            if (_options.Backend.ToLower() != "cpu" && _options.ParallelGames > 1)
            {
                Console.WriteLine("Warning: --parallel-games is ignored when using GPU backend. Running sequentially.");
            }
            Console.WriteLine($"Running with parallelism level: {parallelismLevel}");

            if (parallelismLevel > 1) // CPU Parallel Batch Mode
            {
                for (int i = 0; i < _options.NumberOfGames; i += parallelismLevel)
                {
                    var batchSw = System.Diagnostics.Stopwatch.StartNew();
                    int batchSize = Math.Min(parallelismLevel, _options.NumberOfGames - i);
                    Console.WriteLine($"Starting batch of {batchSize} games (Total completed: {completedGames})...");
                    
                    Parallel.For(0, batchSize, j =>
                    {
                        var gameLog = RunSingleGame(predictionService);                        
                        if (gameLog != null)
                        {
                            lock (_fileLock)
                            {
                                int gameNumber = completedGames + j;
                                int filePart = (gameNumber / 5000) + 1;
                                string logFilePath = Path.Combine(_options.OutputPath, $"training_data_part_{filePart:D3}.jsonl");
                                string singleLineJson = JsonConvert.SerializeObject(gameLog, Formatting.None);
                                File.AppendAllText(logFilePath, singleLineJson + Environment.NewLine);
                            }
                        }
                    });
                    batchSw.Stop();
                    int currentCompleted = Interlocked.Add(ref completedGames, batchSize);
                    Console.WriteLine($"Batch of {batchSize} finished in {batchSw.Elapsed:g}. Total simulated: {currentCompleted} / {_options.NumberOfGames}");
                }
            }
            else // Sequential (one-by-one) Mode
            {
                for (int i = 0; i < _options.NumberOfGames; i++)
                {
                    Console.WriteLine($"Starting game {i + 1} / {_options.NumberOfGames}...");
                    var gameSw = System.Diagnostics.Stopwatch.StartNew();
                    var gameLog = RunSingleGame(predictionService);                    
                    gameSw.Stop();

                    if (gameLog != null)
                    {
                        lock (_fileLock)
                        {
                            int filePart = (completedGames / 5000) + 1;
                            string logFilePath = Path.Combine(_options.OutputPath, $"training_data_part_{filePart:D3}.jsonl");
                            string singleLineJson = JsonConvert.SerializeObject(gameLog, Formatting.None);
                            File.AppendAllText(logFilePath, singleLineJson + Environment.NewLine);
                            completedGames++;
                        }
                        Console.WriteLine($"Simulated game {completedGames} / {_options.NumberOfGames} in {gameSw.Elapsed:g}");
                    }
                }
            }
            Tensorflow.Binding.tf.reset_default_graph();
            sessionStopwatch.Stop();
            Console.WriteLine($"({DateTime.Now:T}) Finished session of {completedGames} games in {sessionStopwatch.Elapsed:g}.");
        }

        private TrainingGameLog? RunSingleGame(PredictionService? predictionService)
        {
            MctsEngine? mctsEngine = null;
            if (_options.Mode == PlayMode.Mcts)
            {
                if (predictionService == null)
                {
                    lock (Console.Out)
                    {
                        Console.WriteLine("\nError: MCTS mode requires a prediction service, but it was not provided.");
                    }
                    return null;
                }
                mctsEngine = new MctsEngine(_logic, predictionService, numSimulations: 100, batchSize: _options.MctsBatchSize);
                return RunAndSaveGame(mctsEngine);
            }
            else // Random Mode
            {
                return RunAndSaveGame(null);
            }
        }
        private TrainingGameLog? RunAndSaveGame(MctsEngine? mctsEngine) {
            var gameSimulator = new GameSimulator(_logic, _options, mctsEngine);
            TrainingGameLog gameLog = gameSimulator.PlayGame();

            if (gameLog != null && gameLog.Turns.Count > 0) {
                return gameLog;
            }
            return null;
        }
    }

    public class BestModelResponse { public string? BestModelName { get; set; } }
}