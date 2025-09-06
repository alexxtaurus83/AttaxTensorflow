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
            //Console.WriteLine($"Loading TensorFlow model from local path: {modelPath}");
            _model = tf.keras.models.load_model(modelPath);
            Console.WriteLine("Model loaded successfully.");
            
        }

        public (float[] Policy, float Value) Predict(BitboardState boardState, AtaxxLogic.PlayerColor player) {
            var inputs = new[] { DataPreprocessor.FenToInputTensor(_logic, _logic.GetStateAsFen(boardState, player)) };
            var (policies, values) = PredictBatch(inputs);
            return (policies[0], values[0]);
        }

        public (float[][] Policies, float[] Values) PredictBatch(NDArray[] batchInputs)
        {
            if (batchInputs == null || batchInputs.Length == 0)
            {
                return (new float[0][], new float[0]);
            }

            var tf_input = np.stack(batchInputs);
            var outputs = _model.Apply(tf_input);
            var policyTensor = outputs[0];
            var valueTensor = outputs[1];

            // ToArray<T> only works with flat arrays. We get the flat array and then reshape it manually.
            var flatPolicyArray = policyTensor.numpy().ToArray<float>();
            var values = valueTensor.numpy().ToArray<float>();

            int batchSize = batchInputs.Length;
            int policySize = AttaxConstants.MCTS.PolicyVectorSize;
            var policies = new float[batchSize][];

            for (int i = 0; i < batchSize; i++)
            {
                policies[i] = new float[policySize];
                Array.Copy(flatPolicyArray, i * policySize, policies[i], 0, policySize);
            }

            return (policies, values);
        }
    }
}