using FluentScheduler;

namespace Ataxx.Trainer {
    public class TrainingJob : IJob {
        private readonly Program.Options _options;

        public TrainingJob(Program.Options options) {
            _options = options;
        }

        public void Execute() {
            Console.WriteLine($"({DateTime.Now:T}) Starting training job...");
            try {
                var trainer = new ModelTrainer(_options);
                trainer.Train(); // This runs one full training session
            } catch (Exception ex) {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"({DateTime.Now:T}) Training job failed: {ex.Message}");
                Console.ResetColor();
            }
            Console.WriteLine($"({DateTime.Now:T}) Training job finished.");
        }
    }
}