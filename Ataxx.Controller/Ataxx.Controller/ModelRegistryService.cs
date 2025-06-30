using System.IO;

namespace Ataxx.Controller {
    public class ModelRegistryService {
        private readonly string _modelsPath;
        private readonly string _bestModelMarkerPath;
        private readonly string _historyFilePath;
        private static readonly object _fileLock = new object(); // Keep lock for process safety

        public ModelRegistryService(IConfiguration configuration) {
            _modelsPath = configuration["ModelManagement:ModelsPath"] ?? throw new InvalidOperationException("ModelsPath is not configured.");
            _bestModelMarkerPath = Path.Combine(_modelsPath, "best_model.txt");
            _historyFilePath = Path.Combine(_modelsPath, "history.log");

            // Use standard Directory.Exists
            if (!Directory.Exists(_modelsPath)) {
                Directory.CreateDirectory(_modelsPath);
            }
        }

        public string GetBestModelName() {
            lock (_fileLock) {
                // Use standard File.Exists and File.ReadAllText
                if (!File.Exists(_bestModelMarkerPath)) return "initial";
                return File.ReadAllText(_bestModelMarkerPath).Trim();
            }
        }

        public void PromoteModel(string modelName) {
            lock (_fileLock) {
                var finalDir = Path.Combine(_modelsPath, modelName);
                if (!Directory.Exists(finalDir)) {
                    throw new DirectoryNotFoundException($"Cannot promote model: Directory '{modelName}' does not exist at path '{_modelsPath}'.");
                }

                File.WriteAllText(_bestModelMarkerPath, modelName);

                string logMessage = $"{DateTime.UtcNow:O}: Promoted '{modelName}' to best.\n";
                File.AppendAllText(_historyFilePath, logMessage);
            }
        }
    }
}