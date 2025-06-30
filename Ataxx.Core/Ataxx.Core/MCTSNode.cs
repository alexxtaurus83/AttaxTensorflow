using OneOf.Types;
using System;
using System.Buffers.Text;
using System.Collections.Generic;
using System.Linq;

namespace Ataxx.Core {
    public class MctsNode {
        private readonly AtaxxLogic _logic;
        private readonly MctsNode? _parent;
        private readonly AtaxxLogic.PlayerColor _player; // The player who will make a move from this state.
        private readonly float _priorProbability; // The probability of selecting this node, given by the neural network's policy head.

        private readonly Dictionary<int, MctsNode> _children = new Dictionary<int, MctsNode>();
        private int _visitCount = 0;
        private double _totalValue = 0.0;

        public IReadOnlyDictionary<int, MctsNode> GetChildren() => _children;
        public AtaxxLogic.PlayerColor Player => _player;
        public BitboardState State { get; }

        public MctsNode(AtaxxLogic logic, BitboardState state, AtaxxLogic.PlayerColor player, MctsNode? parent = null, float priorProbability = 0) {
            _logic = logic;
            State = state;
            _parent = parent;
            _player = player;
            _priorProbability = priorProbability;
        }

        /// <summary>
        // Selects the best child node to explore next based on the UCB1 formula.
        // This balances exploration and exploitation.
        /// </summary>
        public MctsNode SelectChild() {
            // The UCB1 (Upper Confidence Bound 1) formula: Q(s,a) + c * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
            // Q(s,a) is the average value of the child node.
            // P(s,a) is the prior probability from the neural network.
            // N(s) is the visit count of the parent node.
            // N(s,a) is the visit count of the child node.
            // c is the exploration constant (Cpuct).

            return _children.Values.OrderByDescending(child => {
                double qValue = child._visitCount == 0 ? 0 : child._totalValue / child._visitCount;
                double uValue = AttaxConstants.MCTS.Cpuct * child._priorProbability * Math.Sqrt(_visitCount) / (1 + child._visitCount);
                return qValue + uValue;
            }).First();
        }

        /// <summary>
        // Expands the current node by creating child nodes for all valid moves.
        // The policy from the neural network is used to assign prior probabilities to these children.
        /// </summary>
        public void Expand(float[] policy) {
            var validMoves = _logic.GetAllValidMoves(State, _player);
            var nextPlayer = _logic.SwitchPlayer(_player);

            foreach (var move in validMoves) {
                int moveIndex = _logic.MoveToIndex(move);
                if (moveIndex != -1) {
                    var childState = _logic.MakeMove(State, move, _player, useOrthogonalCapture: false);
                    var childNode = new MctsNode(_logic, childState, nextPlayer, this, policy[moveIndex]);
                    _children[moveIndex] = childNode;
                }
            }
        }

        /// <summary>
        // Backpropagates the result of a simulation up the tree from this node to the root.
        /// </summary>
        public void Backpropagate(double value) {
            MctsNode? node = this;
            while (node != null) {
                node._visitCount++;
                // The value is from the perspective of the current player.
                // As we go up the tree, the player flips, so we must negate the value.
                node._totalValue += (node._player == _player) ? value : -value;
                node = node._parent;
            }
        }

        public bool IsLeafNode() => _children.Count == 0;
        public bool IsExpanded => _children.Count > 0;
        public int VisitCount => _visitCount;
    }
}