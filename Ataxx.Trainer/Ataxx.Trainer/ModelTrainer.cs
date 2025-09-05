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


        public void Train() {
            try {
                CudaInterop.PrintGpuMemory();
                string logFilePath = Path.Combine(_options.DataPath, "training_data.jsonl");
                if (!File.Exists(logFilePath)) {
                    Console.WriteLine("No training data log file found. Exiting job.");
                    return;
                }

                var output_types = new TF_DataType[] { TF_DataType.TF_FLOAT, TF_DataType.TF_FLOAT, TF_DataType.TF_FLOAT };
                var output_shapes = new Shape[] { new Shape(7, 7, 4), new Shape(AttaxConstants.MCTS.PolicyVectorSize), new Shape(1) };

                Console.WriteLine("Loading all samples into memory (single pass)...");
                var all_samples = new List<(NDArray, NDArray, NDArray)>();
                var stopwatch = Stopwatch.StartNew();
                long lastUpdateTicks = 0;

                foreach (var sample in StreamSamplesFromFile(logFilePath))
                {
                    all_samples.Add(sample);

                    if (stopwatch.Elapsed.Ticks - lastUpdateTicks > TimeSpan.FromSeconds(5).Ticks)
                    {
                        lastUpdateTicks = stopwatch.Elapsed.Ticks;
                        Console.Write($"\rLoaded {all_samples.Count} samples...");
                    }
                }
                stopwatch.Stop();
                Console.WriteLine($"\rLoaded a total of {all_samples.Count} samples in {stopwatch.Elapsed.TotalSeconds:F2}s.");
                
                int numSamples = all_samples.Count;
                if (numSamples < (_options.BatchSize * 2)) {
                    Console.WriteLine($"Not enough valid data ({numSamples} samples) found to run a training session. Exiting job.");
                    return;
                }

                var input_tensors = all_samples.Select(s => s.Item1).ToArray();
                var policy_tensors = all_samples.Select(s => s.Item2).ToArray();
                var value_tensors = all_samples.Select(s => s.Item3).ToArray();

                var inputs = np.stack(input_tensors);
                var policies = np.stack(policy_tensors);
                var values = np.stack(value_tensors);
                Console.WriteLine("All samples loaded.");

                long ramUsedBytes = Process.GetCurrentProcess().PrivateMemorySize64;
                double ramUsedMb = ramUsedBytes / (1024.0 * 1024.0);
                Console.WriteLine($"RAM used by process after loading data: {ramUsedMb:F2} MB");

                Console.WriteLine("Creating TensorFlow dataset from tensors (moving data to GPU)...");
                var input_dataset = tf.data.Dataset.from_tensor_slices(tf.constant(inputs));
                var policy_dataset = tf.data.Dataset.from_tensor_slices(tf.constant(policies));
                var value_dataset = tf.data.Dataset.from_tensor_slices(tf.constant(values));
                var dataset = tf.data.Dataset.zip(new IDatasetV2[] { input_dataset, policy_dataset, value_dataset });
                Console.WriteLine("Dataset created successfully.");

                Console.WriteLine("Shuffling dataset...");
                dataset = dataset.shuffle(numSamples);
                Console.WriteLine("Dataset shuffled.");

                int valCount = (int)(numSamples * _options.ValidationSplit);
                int trainCount = numSamples - valCount;
                Console.WriteLine($"Loaded {trainCount} training samples and {valCount} validation samples.");

                var train_dataset = dataset.take(trainCount);
                var val_dataset = dataset.skip(trainCount);

                train_dataset = train_dataset.batch(_options.BatchSize).prefetch(tf.data.AUTOTUNE);
                val_dataset = val_dataset.batch(_options.BatchSize).prefetch(tf.data.AUTOTUNE);

                var (model, optimizer) = BuildModel();
                model.summary();

                var policy_loss_func = new CategoricalCrossentropy();
                var value_loss_func = new MeanSquaredError();

                var logdir = Path.Combine("logs", DateTime.Now.ToString("yyyy-MM-dd-HH-mm-ss"));
                var writer = tf.summary.FileWriter(logdir, tf.get_default_graph());

                Console.WriteLine("Starting training loop with large-file streaming pipeline...");
                for (int epoch = 0; epoch < _options.Epochs; epoch++) {
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    float totalTrainLoss = 0;
                    int trainBatches = 0;

                    foreach (var (batch_inputs, batch_labels) in train_dataset) {
                        var policy_and_value_labels = batch_labels as Tensors;
                        var batch_policy_labels = policy_and_value_labels[0];
                        var batch_value_labels = policy_and_value_labels[1];

                        using var tape = tf.GradientTape();
                        var predictions = model.Apply(batch_inputs, training: true) as Tensors;
                        var policy_pred = predictions[0];
                        var value_pred = predictions[1];

                        var policy_loss = policy_loss_func.Call(batch_policy_labels, policy_pred);
                        var value_loss = value_loss_func.Call(batch_value_labels, value_pred);
                        var combined_loss = tf.reduce_mean(policy_loss) + tf.reduce_mean(value_loss);

                        var gradients = tape.gradient(combined_loss, model.TrainableVariables);
                        optimizer.apply_gradients(zip(gradients, model.TrainableVariables));

                        totalTrainLoss += combined_loss.numpy();
                        trainBatches++;
                    }

                    float totalValLoss = 0;
                    int valBatches = 0;
                    foreach (var (batch_inputs, batch_labels) in val_dataset) {
                        var policy_and_value_labels = batch_labels as Tensors;
                        var batch_policy_labels = policy_and_value_labels[0];
                        var batch_value_labels = policy_and_value_labels[1];

                        var predictions = model.Apply(batch_inputs, training: false) as Tensors;
                        var policy_pred = predictions[0];
                        var value_pred = predictions[1];

                        var policy_loss = policy_loss_func.Call(batch_policy_labels, policy_pred);
                        var value_loss = value_loss_func.Call(batch_value_labels, value_pred);
                        var combined_loss = tf.reduce_mean(policy_loss) + tf.reduce_mean(value_loss);

                        totalValLoss += combined_loss.numpy();
                        valBatches++;
                    }

                    sw.Stop();
                    float avgTrainLoss = trainBatches > 0 ? totalTrainLoss / trainBatches : 0;
                    float avgValLoss = valBatches > 0 ? totalValLoss / valBatches : 0;

                    var trainSummary = new Summary();
                    trainSummary.Value.Add(new Summary.Types.Value { Tag = "Loss/train", SimpleValue = avgTrainLoss });
                    writer.add_summary(trainSummary.ToString(), epoch);

                    var valSummary = new Summary();
                    valSummary.Value.Add(new Summary.Types.Value { Tag = "Loss/validation", SimpleValue = avgValLoss });
                    writer.add_summary(valSummary.ToString(), epoch);

                    Console.WriteLine($"Epoch {epoch + 1}/{_options.Epochs} -> Train Loss: {avgTrainLoss:F4}, Val Loss: {avgValLoss:F4}, Time: {sw.ElapsedMilliseconds}ms");
                    CudaInterop.PrintGpuMemory();
                }
                SaveModel(model);

                string archiveDir = Path.Combine(_options.DataPath, "archive");
                Directory.CreateDirectory(archiveDir);
                string archiveFileName = $"data_{DateTime.UtcNow:yyyyMMddHHmmss}.jsonl";
                File.Move(logFilePath, Path.Combine(archiveDir, archiveFileName));
                Console.WriteLine($"Archived processed data to {archiveFileName}");

            } catch (Exception ex) {
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

        private void SaveModel(IModel model) {
            Console.WriteLine("Training complete. Saving candidate model with metadata...");
            var candidateName = $"model_lr{_options.LearningRate}_bs{_options.BatchSize}_{DateTime.UtcNow:yyyyMMddHHmmss}_candidate";
            var savePath = Path.Combine(_options.ModelOutputPath, candidateName);
            model.save(savePath, save_format: "tf");
            File.WriteAllText(Path.Combine(savePath, "metadata.json"), JsonConvert.SerializeObject(_options, Newtonsoft.Json.Formatting.Indented));
            Console.WriteLine($"New candidate model saved to {savePath}");
        }
    }
}