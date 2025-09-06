using Ataxx.Trainer;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Ataxx.Core {
    /// <summary>
    /// The main AI engine that uses Monte Carlo Tree Search guided by a neural network.
    /// </summary>
    public class MctsEngine {
        private readonly AtaxxLogic _logic;
        private readonly PredictionService _predictionService;
        private readonly int _numSimulations;

        private readonly int _batchSize;

        public MctsEngine(AtaxxLogic logic, PredictionService predictionService, int numSimulations = 400, int batchSize = 8) {
            _logic = logic;
            _predictionService = predictionService;
            _numSimulations = numSimulations;
            _batchSize = batchSize;
        }

        /// <summary>
        /// Finds the best move and the resulting MCTS policy for a given board state.
        /// </summary>
        /// <returns>A tuple containing the best move and the policy vector (based on node visit counts).</returns>
        public (AtaxxLogic.Move bestMove, float[] policy) FindBestMove(BitboardState rootState, AtaxxLogic.PlayerColor player) {
            var rootNode = new MctsNode(_logic, rootState, player);

            var validMoves = _logic.GetAllValidMoves(rootState, player);
            // If there is only one possible move, just take it without searching.
            if (validMoves.Count == 1) {
                var policy = new float[AttaxConstants.MCTS.PolicyVectorSize];
                policy[_logic.MoveToIndex(validMoves[0])] = 1.0f;
                return (validMoves[0], policy);
            }

            // If there are no moves, return a default move and an empty policy
            if (validMoves.Count == 0) {
                return (default(AtaxxLogic.Move), new float[AttaxConstants.MCTS.PolicyVectorSize]);
            }

            for (int i = 0; i < _numSimulations; i += _batchSize) {
                RunSimulationsInBatch(rootNode, Math.Min(_batchSize, _numSimulations - i));
            }

            // Create the policy vector based on the visit counts of the root's children.
            var mctsPolicy = new float[AttaxConstants.MCTS.PolicyVectorSize];
            var rootChildren = rootNode.GetChildren();
            if (rootNode.VisitCount > 0) {
                foreach (var child in rootChildren) {
                    mctsPolicy[child.Key] = (float)child.Value.VisitCount / rootNode.VisitCount;
                }
            }

            // The best move corresponds to the child node with the most visits.
            var bestMoveIndex = rootChildren.OrderByDescending(kvp => kvp.Value.VisitCount).First().Key;
            var bestMove = _logic.IndexToMove(bestMoveIndex);

            return (bestMove, mctsPolicy);
        }

        private void RunSimulationsInBatch(MctsNode rootNode, int numSimulationsToRun)
        {
            var leafNodes = new List<MctsNode>();
            var terminalNodes = new List<(MctsNode Node, double Value)>();

            // 1. SELECTION for the entire batch
            for (int i = 0; i < numSimulationsToRun; i++)
            {
                var node = rootNode;
                while (node.IsExpanded && !_logic.IsGameOver(node.State))
                {
                    node = node.SelectChild();
                }

                if (_logic.IsGameOver(node.State))
                {
                    int redCount = _logic.PopCount(node.State.RedPieces);
                    int blueCount = _logic.PopCount(node.State.BluePieces);
                    double value = (redCount > blueCount) ? 1.0 : (blueCount > redCount) ? -1.0 : 0.0;
                    terminalNodes.Add((node, value));
                }
                else
                {
                    leafNodes.Add(node);
                }
            }

            // 2. EXPANSION & PREDICTION for non-terminal leaves
            if (leafNodes.Count > 0)
            {
                var batchInputs = leafNodes.Select(n => DataPreprocessor.FenToInputTensor(_logic, _logic.GetStateAsFen(n.State, n.Player))).ToArray();
                var (policies, values) = _predictionService.PredictBatch(batchInputs);

                for (int i = 0; i < leafNodes.Count; i++)
                {
                    var node = leafNodes[i];
                    var policy = policies[i];
                    var value = values[i];

                    node.Expand(policy);
                    node.Backpropagate(value);
                }
            }

            // 3. BACKPROPAGATION for terminal nodes
            foreach (var (node, value) in terminalNodes)
            {
                node.Backpropagate(value);
            }
        }
    }
}