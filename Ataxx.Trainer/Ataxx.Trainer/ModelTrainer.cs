using Ataxx.Core;
using Newtonsoft.Json;
using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Collections.Generic;
using System.Linq;
using Tensorflow;
using Tensorflow.Device;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Layers;
using Tensorflow.Keras.Losses;
using Tensorflow.Keras.Optimizers;
using Tensorflow.Operations.Initializers;
using static Tensorflow.Binding;
using Tensorflow.NumPy;

namespace Ataxx.Trainer {
    public class ModelTrainer {
        private readonly Program.Options _options;
        private readonly AtaxxLogic _logic;
        private const int SHUFFLE_BUFFER_SIZE = 10000;

        public ModelTrainer(Program.Options options) {
            _options = options;
            _logic = new AtaxxLogic();
            var gpus = tf.config.list_physical_devices("GPU");
            Console.WriteLine($"TensorFlow.NET v{tf.VERSION}, Found {gpus.Length} GPUs.");
            foreach (PhysicalDevice device in gpus) {
                try {
                    tf.config.experimental.set_memory_growth(device, true);
                    Console.WriteLine($"Successfully set memory growth for {device.DeviceName}");
                } catch (RuntimeError e) {
                    Console.WriteLine($"Error setting memory growth: {e.Message}");
                }
            }
        }

        private IEnumerable<(NDArray, NDArray, NDArray)> StreamSamplesFromFile(string logFilePath) {
            foreach (var line in File.ReadLines(logFilePath)) {
                if (string.IsNullOrWhiteSpace(line)) continue;

                TrainingGameLog gameLog = null;
                try {
                    gameLog = JsonConvert.DeserializeObject<TrainingGameLog>(line);
                } catch (JsonSerializationException) {
                    continue;
                }

                if (gameLog != null) {
                    foreach (var turn in gameLog.Turns) {
                        var inputTensor = DataPreprocessor.FenToInputTensor(_logic, turn.BoardFen);
                        var policyTensor = new NDArray(turn.MctsPolicy);
                        var valueTensor = new NDArray(new float[] { turn.GameOutcome });

                        yield return (inputTensor, policyTensor, valueTensor);
                    }
                }
            }
        }

        private int GetTotalSampleCount(string logFilePath) {
            int count = 0;
            foreach (var line in File.ReadLines(logFilePath)) {
                if (string.IsNullOrWhiteSpace(line)) continue;
                try {
                    var gameLog = JsonConvert.DeserializeObject<TrainingGameLog>(line);
                    if (gameLog != null) {
                        count += gameLog.Turns.Count;
                    }
                } catch (JsonSerializationException) {
                    continue;
                }
            }
            return count;
        }


        public void Train()
        {
            Tensorflow.Summaries.FileWriter writer = null;
            try
            {
                CudaInterop.PrintGpuMemory();

                var allFiles = Directory.GetFiles(_options.DataPath, "training_data_part_*.jsonl");
                if (allFiles.Length == 0)
                {
                    Console.WriteLine("No training data files found. Exiting job.");
                    return;
                }

                // Shuffle the file list and split for validation
                var rng = new Random(42);
                allFiles = allFiles.OrderBy(f => rng.Next()).ToArray();
                
                List<string> trainFiles;
                List<string> valFiles;

                if (allFiles.Length == 1)
                {
                    Console.WriteLine("Warning: Only one data file found. Using it for both training and validation.");
                    trainFiles = allFiles.ToList();
                    valFiles = allFiles.ToList();
                }
                else
                {
                    int valFileCount = (int)Math.Max(1, allFiles.Length * _options.ValidationSplit);
                    valFiles = allFiles.Take(valFileCount).ToList();
                    trainFiles = allFiles.Skip(valFileCount).ToList();
                }

                Console.WriteLine($"Found {allFiles.Length} data files. Using {trainFiles.Count} for training and {valFiles.Count} for validation.");

                var (model, optimizer) = BuildModel();
                model.summary();

                var policy_loss_func = new CategoricalCrossentropy();
                var value_loss_func = new MeanSquaredError();

                var logdir = Path.Combine("logs", DateTime.Now.ToString("yyyy-MM-dd-HH-mm-ss"));
                writer = tf.summary.FileWriter(logdir, tf.get_default_graph());

                long totalSamples = 0;
                for (int epoch = 0; epoch < _options.Epochs; epoch++)
                {
                    Console.WriteLine($"--- Starting Epoch {epoch + 1}/{_options.Epochs} ---");
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    float totalTrainLoss = 0, totalTrainPolicyLoss = 0, totalTrainValueLoss = 0;
                    int totalTrainBatches = 0;
                    if(epoch == 0) totalSamples = 0; // Reset on first epoch

                    // --- Training Phase ---
                    foreach (var (filePath, index) in trainFiles.Select((path, i) => (path, i)))
                    {
                        Console.WriteLine($"Processing training file {index + 1}/{trainFiles.Count}: {Path.GetFileName(filePath)}");
                        var (train_dataset, numSamples) = LoadDatasetFromFile(filePath);
                        Console.WriteLine($"dataset loaded");
                        totalSamples += numSamples;                        
                        foreach (var (batch_inputs, batch_labels) in train_dataset)
                        {
                            Console.WriteLine($"Processing totalTrainBatches {totalTrainBatches}");
                            var batch_policy_labels = (batch_labels as Tensors)[0];
                            var batch_value_labels = (batch_labels as Tensors)[1];

                            using var tape = tf.GradientTape();
                            var predictions = model.Apply(batch_inputs, training: true) as Tensors;
                            var policy_pred = predictions[0];
                            var value_pred = predictions[1];

                            var policy_loss = tf.reduce_mean(policy_loss_func.Call(batch_policy_labels, policy_pred));
                            var value_loss = tf.reduce_mean(value_loss_func.Call(batch_value_labels, value_pred));
                            var combined_loss = policy_loss + value_loss;

                            var gradients = tape.gradient(combined_loss, model.TrainableVariables);
                            var (clipped_gradients, _) = tf.clip_by_global_norm(gradients, clip_norm: 1.0f);
                            optimizer.apply_gradients(zip(clipped_gradients, model.TrainableVariables));

                            totalTrainLoss += combined_loss.numpy();
                            totalTrainPolicyLoss += policy_loss.numpy();
                            totalTrainValueLoss += value_loss.numpy();
                            totalTrainBatches++;
                        }
                    }

                    // --- Validation Phase ---
                    float totalValLoss = 0, totalValPolicyLoss = 0, totalValValueLoss = 0;
                    int totalValBatches = 0;
                    if (valFiles.Count > 0)
                    {
                        Console.WriteLine("Starting validation phase...");
                        foreach (var filePath in valFiles)
                        {                            
                            var (val_dataset, _) = LoadDatasetFromFile(filePath);
                            foreach (var (batch_inputs, batch_labels) in val_dataset)
                            {
                                Console.WriteLine($"Processing totalValBatches {totalValBatches}");
                                var batch_policy_labels = (batch_labels as Tensors)[0];
                                var batch_value_labels = (batch_labels as Tensors)[1];
                                var predictions = model.Apply(batch_inputs, training: false) as Tensors;
                                var policy_pred = predictions[0];
                                var value_pred = predictions[1];

                                var policy_loss = tf.reduce_mean(policy_loss_func.Call(batch_policy_labels, policy_pred));
                                var value_loss = tf.reduce_mean(value_loss_func.Call(batch_value_labels, value_pred));
                                var combined_loss = policy_loss + value_loss;

                                totalValLoss += combined_loss.numpy();
                                totalValPolicyLoss += policy_loss.numpy();
                                totalValValueLoss += value_loss.numpy();
                                totalValBatches++;
                            }
                        }
                    }

                    sw.Stop();
                    float avgTrainLoss = totalTrainBatches > 0 ? totalTrainLoss / totalTrainBatches : 0;
                    float avgTrainPolicyLoss = totalTrainBatches > 0 ? totalTrainPolicyLoss / totalTrainBatches : 0;
                    float avgTrainValueLoss = totalTrainBatches > 0 ? totalTrainValueLoss / totalTrainBatches : 0;

                    float avgValLoss = totalValBatches > 0 ? totalValLoss / totalValBatches : 0;
                    float avgValPolicyLoss = totalValBatches > 0 ? totalValPolicyLoss / totalValBatches : 0;
                    float avgValValueLoss = totalValBatches > 0 ? totalValValueLoss / totalValBatches : 0;

                    var trainSummary = new Tensorflow.Summary();
                    trainSummary.Value.Add(new Tensorflow.Summary.Types.Value { Tag = "Loss/train_total", SimpleValue = avgTrainLoss });
                    trainSummary.Value.Add(new Tensorflow.Summary.Types.Value { Tag = "Loss/train_policy", SimpleValue = avgTrainPolicyLoss });
                    trainSummary.Value.Add(new Tensorflow.Summary.Types.Value { Tag = "Loss/train_value", SimpleValue = avgTrainValueLoss });
                    writer.add_summary(trainSummary.ToString(), epoch);

                    var valSummary = new Tensorflow.Summary();
                    valSummary.Value.Add(new Tensorflow.Summary.Types.Value { Tag = "Loss/validation_total", SimpleValue = avgValLoss });
                    valSummary.Value.Add(new Tensorflow.Summary.Types.Value { Tag = "Loss/validation_policy", SimpleValue = avgValPolicyLoss });
                    valSummary.Value.Add(new Tensorflow.Summary.Types.Value { Tag = "Loss/validation_value", SimpleValue = avgValValueLoss });
                    writer.add_summary(valSummary.ToString(), epoch);

                    Console.WriteLine($"Epoch {epoch + 1}/{_options.Epochs} -> Train Loss: {avgTrainLoss:F4}, Val Loss: {avgValLoss:F4}, Time: {sw.Elapsed.TotalSeconds:F2}s");
                    CudaInterop.PrintGpuMemory();
                }

                SaveModel(model, (int)totalSamples);

                if (_options.DeleteDataAfterProcessing)
                {
                    Console.WriteLine("Deleting processed data files...");
                    foreach (var file in allFiles)
                    {
                        File.Delete(file);
                    }
                    Console.WriteLine($"Deleted {allFiles.Length} files.");
                }
                else
                {
                    string archiveDir = Path.Combine(_options.DataPath, "archive", $"processed_{DateTime.UtcNow:yyyyMMdd_HHmmss}");
                    Directory.CreateDirectory(archiveDir);
                    foreach (var file in allFiles)
                    {
                        File.Move(file, Path.Combine(archiveDir, Path.GetFileName(file)));
                    }
                    Console.WriteLine($"Archived {allFiles.Length} processed data files to {archiveDir}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                Console.WriteLine(ex.StackTrace);
            }
        }

        private (IModel model, OptimizerV2 optimizer) BuildModel() {
            var inputs = tf.keras.Input(shape: (7, 7, 4), name: "board_input");
            var conv1 = tf.keras.layers.Conv2D(
                    filters: 32,
                    kernel_size: (3, 3),
                    padding: "same",
                    activation: "relu"
                ).Apply(inputs);

            var conv2 = tf.keras.layers.Conv2D(
                    filters: 64,
                    kernel_size: (3, 3),
                    padding: "same",
                    activation: "relu"
                ).Apply(conv1);

            var conv3 = tf.keras.layers.Conv2D(
                    filters: 128,
                    kernel_size: (3, 3),
                    padding: "same",
                    activation: "relu"
                ).Apply(conv2);

            var flatten = new Flatten(new FlattenArgs()).Apply(conv3);

            var policy_dense = new Dense(new DenseArgs { Units = 256, Activation = tf.keras.activations.Relu }).Apply(flatten);
            var policy_head = new Dense(new DenseArgs { Units = AttaxConstants.MCTS.PolicyVectorSize, Activation = tf.keras.activations.Softmax, Name = "policy_head" }).Apply(policy_dense);

            var value_dense = new Dense(new DenseArgs { Units = 64, Activation = tf.keras.activations.Relu }).Apply(flatten);
            var value_head = new Dense(new DenseArgs { Units = 1, Activation = tf.keras.activations.Tanh, Name = "value_head" }).Apply(value_dense);

            var model = tf.keras.Model(inputs, new Tensors(policy_head, value_head));
            var optimizer = new Adam(_options.LearningRate);
            return (model, optimizer);
        }

        private (IDatasetV2, int) LoadDatasetFromFile(string filePath)
        {
            var all_samples = new List<(NDArray, NDArray, NDArray)>();
            var i = 0;
            foreach (var sample in StreamSamplesFromFile(filePath))
            {
                Console.WriteLine($"loading sample {i}");
                all_samples.Add(sample);
                i++;
            }
            Console.WriteLine($"{all_samples.Count} loaded");
            if (all_samples.Count == 0)
            {
                return (null, 0);
            }

            var input_tensors = all_samples.Select(s => s.Item1).ToArray();
            var policy_tensors = all_samples.Select(s => s.Item2).ToArray();
            var value_tensors = all_samples.Select(s => s.Item3).ToArray();

            var inputs = np.stack(input_tensors);
            var policies = np.stack(policy_tensors);
            var values = np.stack(value_tensors);

            var input_dataset = tf.data.Dataset.from_tensor_slices(tf.constant(inputs));
            var policy_dataset = tf.data.Dataset.from_tensor_slices(tf.constant(policies));
            var value_dataset = tf.data.Dataset.from_tensor_slices(tf.constant(values));
            var dataset = tf.data.Dataset.zip(input_dataset, policy_dataset, value_dataset);

            dataset = dataset.shuffle((int)all_samples.Count);
            dataset = dataset.batch(_options.BatchSize).prefetch(tf.data.AUTOTUNE);

            return (dataset, all_samples.Count);
        }

        private void SaveModel(IModel model, int numSamples) {
            Console.WriteLine("Training complete. Saving candidate model with metadata...");
            var candidateName = $"model_lr{_options.LearningRate}_bs{_options.BatchSize}_{DateTime.UtcNow:yyyyMMddHHmmss}_candidate";
            var savePath = Path.Combine(_options.ModelOutputPath, candidateName);
            model.save(savePath, save_format: "tf");

            var metadata = new {
                Options = _options,
                TrainingSamples = numSamples,
                TrainingDate = DateTime.UtcNow
            };
            File.WriteAllText(Path.Combine(savePath, "metadata.json"),
                JsonConvert.SerializeObject(metadata, Formatting.Indented));
            Console.WriteLine($"New candidate model saved to {savePath}");
        }
    }
}