using Ataxx.SelfPlayWorker;
using CommandLine;
using FluentScheduler;
using System.Runtime.InteropServices;

// The PlayMode enum is at the top-level of the namespace for easy access from all classes.
public enum PlayMode { Random, Mcts }

public class Program {
    // Defines all command-line options for the worker.
    public class Options {
        [Option("controller-url", Default = "http://localhost:5299", HelpText = "URL of the Ataxx.Controller service.")]
        public string ControllerUrl { get; set; }

        [Option('g', "games", Default = 100, HelpText = "The total number of games to simulate per session.")] //40000
        public int NumberOfGames { get; set; }

        [Option('o', "output-path",  Default = @"F:\attax", HelpText = "The directory path to save game log files.")]
        public string OutputPath { get; set; } = string.Empty;

        [Option('p', "mode", Default = PlayMode.Mcts, HelpText = "The play mode for the worker: Random or Mcts.")] //Default = PlayMode.Mcts
        public PlayMode Mode { get; set; }

        [Option("model-path", Default = @"F:\attax\model", HelpText = "Required for Mcts mode. The path to the trained model folder (can be UNC).")]
        public string? ModelPath { get; set; }

        [Option("blocked-cells", Default = 6, HelpText = "Number of random blocked cells to place on the board.")]
        public int BlockedCells { get; set; }        

        [Option("backend", Default = "gpu", HelpText = "The processing backend to use: 'cpu' or 'gpu'.")]
        public string Backend { get; set; }

        [Option("parallel-games", Default = 1, HelpText = "Number of games to run in parallel (only used with CPU backend).")]
        public int ParallelGames { get; set; }

        [Option("mcts-batch-size", Default = 128, HelpText = "Batch size for MCTS predictions on GPU.")]
        public int MctsBatchSize { get; set; }

        [Option("run-once", Default = true, HelpText = "If true, the worker will run one session and exit. Otherwise, it runs continuously.")]
        public bool RunOnce { get; set; }
    }

    public static void Main(string[] args) {
        Parser.Default.ParseArguments<Options>(args)
               .WithParsed<Options>(o => {
                    var gpus = Tensorflow.Binding.tf.config.list_physical_devices("GPU");
                    if (gpus.Length > 0)
                    {
                        foreach (var device in gpus)
                            Tensorflow.Binding.tf.config.experimental.set_memory_growth(device, true);
                    }

                   if (o.Mode == PlayMode.Mcts && string.IsNullOrEmpty(o.ModelPath)) {
                       Console.ForegroundColor = ConsoleColor.Red;
                       Console.WriteLine("Error: --model-path is required when using Mcts mode.");
                       Console.ResetColor();
                       return;
                   }

                   if (o.RunOnce) {
                       Console.WriteLine($"Starting single self-play session for {o.NumberOfGames} games...");
                       var job = new SelfPlayJob(o);
                       job.Execute();
                       Console.WriteLine("Single session finished. Exiting.");
                   } else {
                       Console.WriteLine($"Initializing scheduler for '{o.Mode}' mode using '{o.Backend}' backend...");
                       JobManager.Initialize(new WorkerRegistry(o));
                       Console.WriteLine("Self-Play Worker service is running continuously. Press [Enter] to exit.");
                       Console.ReadLine();
                       JobManager.StopAndBlock();
                   }
               });
    }
}