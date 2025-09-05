using Ataxx.Trainer;
using System;
using System.IO;
using System.Linq;
using Tensorflow;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;
using Tensorflow.NumPy;

namespace Ataxx.Core {
    // The class no longer needs to implement IDisposable
    public class PredictionService {
        private readonly IModel _model;
        private readonly AtaxxLogic _logic = new AtaxxLogic();

        public PredictionService(string modelPath) {           

            if (!Directory.Exists(modelPath)) {
                throw new DirectoryNotFoundException($"PredictionService: Model path not found at '{modelPath}'");
            }
            Console.WriteLine($"Loading TensorFlow model from local path: {modelPath}");
            _model = tf.keras.models.load_model(modelPath);
            Console.WriteLine("Model loaded successfully.");
            
        }

        public (float[] Policy, float Value) Predict(BitboardState boardState, AtaxxLogic.PlayerColor player) {
            // The prediction logic implicitly uses the default graph.
            var fen = _logic.GetStateAsFen(boardState, player);
            var ns_input = DataPreprocessor.FenToInputTensor(_logic, fen);

            var tf_input = ns_input.reshape(new int[] { 1, 7, 7, 4 });
            var outputs = _model.Apply(tf_input);
            var policyTensor = outputs[0];
            var valueTensor = outputs[1];

            var policy = policyTensor.numpy().ToArray<float>();
            var value = valueTensor.numpy().ToArray<float>()[0];

            return (policy, value);
        }        
    }
}