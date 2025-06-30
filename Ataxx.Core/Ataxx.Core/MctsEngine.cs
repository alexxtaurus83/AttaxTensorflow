using System;
using System.Linq;

namespace Ataxx.Core {
    /// <summary>
    /// The main AI engine that uses Monte Carlo Tree Search guided by a neural network.
    /// </summary>
    public class MctsEngine {
        private readonly AtaxxLogic _logic;
        private readonly PredictionService _predictionService;
        private readonly int _numSimulations;

        /// <summary>
        /// Initializes the MCTS engine.
        /// </summary>
        /// <param name="logic">An instance of the game's rules engine.</param>
        /// <param name="predictionService">The service that provides NN predictions.</param>
        /// <param name="numSimulations">The number of simulations to run for each move search.</param>
        public MctsEngine(AtaxxLogic logic, PredictionService predictionService, int numSimulations = 400) {
            _logic = logic;
            _predictionService = predictionService;
            _numSimulations = numSimulations;
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

            // Run the main simulation loop.
            for (int i = 0; i < _numSimulations; i++) {
                RunSimulation(rootNode);
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

        /// <summary>
        /// Runs a single iteration of the MCTS algorithm (Selection, Expansion, Simulation, Backpropagation).
        /// </summary>
        private void RunSimulation(MctsNode startNode) {
            var node = startNode;

            // 1. SELECTION: Traverse the tree using the UCB formula until a leaf node is found.
            while (node.IsExpanded && !_logic.IsGameOver(node.State)) {
                node = node.SelectChild();
            }

            double value;

            // If we've reached a terminal state (game over), the value is determined directly.
            if (_logic.IsGameOver(node.State)) {
                int redCount = _logic.PopCount(node.State.RedPieces);
                int blueCount = _logic.PopCount(node.State.BluePieces);
                value = (redCount > blueCount) ? 1.0 : (blueCount > redCount) ? -1.0 : 0.0;
            } else {
                // 2. EXPANSION & 3. SIMULATION:
                // Use the neural network to get the policy for child nodes and the value of the current leaf node.
                // The 'value' from the NN is our intelligent simulation result.
                var (policy, predictedValue) = _predictionService.Predict(node.State, startNode.Player);
                value = predictedValue;

                // Expand the leaf node with the new policy information.
                node.Expand(policy);
            }

            // 4. BACKPROPAGATION: Update the visit counts and total values back up the tree.
            node.Backpropagate(value);
        }
    }
}