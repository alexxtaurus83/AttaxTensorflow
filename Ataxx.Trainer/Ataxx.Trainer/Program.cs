using Ataxx.Trainer;
using CommandLine;
using FluentScheduler;
using static Tensorflow.Binding;
public class TrainerRegistry : Registry {
    public TrainerRegistry(Program.Options options) {
        Schedule(() => new TrainingJob(options)).NonReentrant().ToRunNow().AndEvery(10).Minutes();
    }
}

public class Program {
    public class Options {
        [Option('d', "data-path", Default = @"F:\attax", HelpText = "Path to the directory of game log files.")]
        public string DataPath { get; set; } = string.Empty;

        [Option('m', "model-output-path", Default = @"F:\attax\model", HelpText = "Directory to save the trained model.")]
        public string ModelOutputPath { get; set; } = string.Empty;

        [Option('e', "epochs", Default = 5, HelpText = "Number of training epochs.")]
        public int Epochs { get; set; }

        [Option('b', "batch-size", Default = 64, HelpText = "Batch size for training.")]
        public int BatchSize { get; set; }

        [Option('l', "learning-rate", Default = 0.001f, HelpText = "Learning rate for the Adam optimizer.")]
        public float LearningRate { get; set; }
                
        [Option('v', "val-split", Default = 0.1f, HelpText = "Fraction of data to use for validation (e.g., 0.1 for 10%).")]
        public float ValidationSplit { get; set; }
    }

    public static void Main(string[] args) {
        Environment.SetEnvironmentVariable("TF_XLA_FLAGS", "--tf_xla_auto_jit=1");
        Parser.Default.ParseArguments<Options>(args)
               .WithParsed<Options>(o => {
                   Console.WriteLine("Initializing training scheduler...");
                   JobManager.Initialize(new TrainerRegistry(o));
                   Console.WriteLine("Trainer service running. Press [enter] to exit.");
                   Console.ReadLine();
                   JobManager.StopAndBlock();
               });
    }
}