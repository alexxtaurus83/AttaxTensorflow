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

        [Option('g', "games", Default = 30, HelpText = "The total number of games to simulate per session.")] //40000
        public int NumberOfGames { get; set; }

        [Option('o', "output-path",  Default = @"F:\attax", HelpText = "The directory path to save game log files.")]
        public string OutputPath { get; set; } = string.Empty;

        [Option('p', "mode", Default = PlayMode.Random, HelpText = "The play mode for the worker: Random or Mcts.")] //Default = PlayMode.Mcts
        public PlayMode Mode { get; set; }

        [Option("model-path", Default = @"F:\attax\model", HelpText = "Required for Mcts mode. The path to the trained model folder (can be UNC).")]
        public string? ModelPath { get; set; }

        [Option("blocked-cells", Default = 6, HelpText = "Number of random blocked cells to place on the board.")]
        public int BlockedCells { get; set; }        

        [Option("backend", Default = "cpu", HelpText = "The processing backend to use: 'cpu' or 'gpu'.")]
        public string Backend { get; set; }
    }

    public static void Main(string[] args) {
        Parser.Default.ParseArguments<Options>(args)
               .WithParsed<Options>(o => {
                   // --- DYNAMIC LIBRARY RESOLVER ---
                   // This block runs BEFORE any TensorFlow code is called.
                   // It tells the .NET runtime where to find the native tensorflow.dll
                   NativeLibrary.SetDllImportResolver(typeof(Tensorflow.Binding).Assembly, (libraryName, assembly, searchPath) =>
                   {
                       if (libraryName == "tensorflow") {
                           string arch = "win-x64"; // Assuming Windows 64-bit
                           string backendToLoad = o.Backend.ToLower() == "cpu" ? "cpu" : "gpu";

                           // Construct the full path to the chosen native library
                           string libPath = Path.Combine(AppContext.BaseDirectory, "runtimes", $"{arch}-{backendToLoad}", "tensorflow.dll");

                           if (File.Exists(libPath)) {
                               Console.WriteLine($"Dynamically loading TensorFlow backend from: {libPath}");
                               return NativeLibrary.Load(libPath);
                           }
                       }
                       return IntPtr.Zero; // Fallback to default loading mechanism
                   });

                   // --- The rest of the program logic runs now ---

                   if (o.Mode == PlayMode.Mcts && string.IsNullOrEmpty(o.ModelPath)) {
                       Console.ForegroundColor = ConsoleColor.Red;
                       Console.WriteLine("Error: --model-path is required when using Mcts mode.");
                       Console.ResetColor();
                       return;
                   }

                   Console.WriteLine($"Initializing scheduler for '{o.Mode}' mode using '{o.Backend}' backend...");

                   JobManager.Initialize(new WorkerRegistry(o));

                   Console.WriteLine("Self-Play Worker service is running. Press [Enter] to exit.");
                   Console.ReadLine();

                   JobManager.StopAndBlock();
               });
    }
}