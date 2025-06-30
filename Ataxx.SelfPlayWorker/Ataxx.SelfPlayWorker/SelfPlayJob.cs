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

        public void Execute() {
            Console.WriteLine($"({DateTime.Now:T}) Starting new self-play session in '{_options.Mode}' mode (Parallel: {_options.Backend})...");
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();            
            if (_options.Backend.ToLower() == "cpu") {
                // PARALLEL EXECUTION (For your 24-core CPU machine)
                int completedGames = 0;
                Parallel.For(0, _options.NumberOfGames, i =>
                {
                    RunSingleGame();
                    int currentCount = Interlocked.Increment(ref completedGames);
                    lock (Console.Out) {
                        Console.Write($"\rProcessed {currentCount} / {_options.NumberOfGames} games... ");
                    }
                });
                Console.WriteLine();
            } else {
                // SEQUENTIAL EXECUTION (For your GPU machines)
                for (int i = 0; i < _options.NumberOfGames; i++) {
                    RunSingleGame();
                    Console.WriteLine($"Processed {i + 1} / {_options.NumberOfGames} games...");
                }
            }

            stopwatch.Stop();
            Console.WriteLine($"({DateTime.Now:T}) Finished session of {_options.NumberOfGames} games in {stopwatch.Elapsed:g}.");
        }

        private void RunSingleGame() {
            MctsEngine? mctsEngine = null;
            if (_options.Mode == PlayMode.Mcts) {
                try {
                    var client = new RestClient(_options.ControllerUrl);
                    var request = new RestRequest("/api/model/latest");
                    var bestModelName = client.Get<BestModelResponse>(request)?.BestModelName;

                    if (string.IsNullOrEmpty(bestModelName) || bestModelName == "initial") {
                        // In a parallel loop, we don't want to log this every time.
                        // A more advanced logger would handle this, but for now we just skip.
                        return;
                    }

                    string fullModelPath = Path.Combine(_options.ModelPath!, bestModelName);
                    var predictionService = new PredictionService(fullModelPath);
                    mctsEngine = new MctsEngine(_logic, predictionService, numSimulations: 100);
                    RunAndSaveGame(mctsEngine);                 
                    
                } catch (Exception ex) {
                    lock (Console.Out) {
                        Console.ForegroundColor = ConsoleColor.Yellow;
                        Console.WriteLine($"\nCould not run MCTS game. Is Controller/Share accessible? Error: {ex.Message}");
                        Console.ResetColor();
                    }
                }
            } else // Random Mode
              {
                RunAndSaveGame(null);
            }
        }
        private void RunAndSaveGame(MctsEngine? mctsEngine) {
            var gameSimulator = new GameSimulator(_logic, _options, mctsEngine);
            TrainingGameLog gameLog = gameSimulator.PlayGame();

            if (gameLog != null && gameLog.Turns.Count > 0) {
                string fileName = $"{Guid.NewGuid()}.json";
                Console.WriteLine($"({DateTime.Now:T}) Finished game, saving log {fileName}.");
                try {
                    // All workers will append to the same log file.
                    string logFilePath = Path.Combine(_options.OutputPath, "training_data.jsonl");

                    // Convert the entire game log to a single line of JSON.
                    string singleLineJson = JsonConvert.SerializeObject(gameLog, Formatting.None);

                    // Use a lock to prevent threads from writing at the same time.
                    lock (_fileLock) {
                        File.AppendAllText(logFilePath, singleLineJson + Environment.NewLine);
                    }
                } catch (Exception ex) {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine($"\nError writing log file '{fileName}': {ex.Message}");
                    Console.ResetColor();
                }
            }
        }
        private void SaveGameLog(TrainingGameLog log, string fileName) {
            string fullPath = Path.Combine(_options.OutputPath, fileName);
            string tempPath = fullPath + ".tmp";
            string fileContents = log.ToJson();

            // Use standard System.IO for direct file access
            try {
                // 1. Write to a temporary file first.
                File.WriteAllText(tempPath, fileContents);

                // 2. If the write was successful, rename it to the final name.
                // This move/rename operation is atomic on most file systems.
                File.Move(tempPath, fullPath);
            } catch {
                // If anything goes wrong, try to clean up the temp file.
                if (File.Exists(tempPath)) {
                    File.Delete(tempPath);
                }
                throw; // Re-throw the original exception
            }
        }
    }

    public class BestModelResponse { public string? BestModelName { get; set; } }
}