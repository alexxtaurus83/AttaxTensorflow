using Ataxx.Core;

namespace Ataxx.SelfPlayWorker {
    public class GameSimulator {
        private readonly AtaxxLogic _logic;
        private readonly Program.Options _options;
        private readonly Random _random;
        private readonly MctsEngine? _mctsEngine;

        // CORRECTED: The constructor now accepts an MctsEngine object from the SelfPlayJob.
        // This removes the responsibility of creating the engine from this class.
        public GameSimulator(AtaxxLogic logic, Program.Options options, MctsEngine? mctsEngine) {
            _logic = logic;
            _options = options;
            _random = new Random();
            _mctsEngine = mctsEngine;
        }

        public TrainingGameLog PlayGame() {
            var board = _logic.CreateInitialBoard(_options.BlockedCells);
            
            board.RedPieces = (1UL << _logic.GetBitIndex(0, 0)) | (1UL << _logic.GetBitIndex(6, 6));
            board.BluePieces = (1UL << _logic.GetBitIndex(0, 6)) | (1UL << _logic.GetBitIndex(6, 0));

            var currentPlayer = AtaxxLogic.PlayerColor.Red;
            var turnHistory = new List<TurnData>();

            while (!_logic.IsGameOver(board)) {
                var validMoves = _logic.GetAllValidMoves(board, currentPlayer);

                if (validMoves.Count == 0) {
                    // Record the state where the player was forced to pass.
                    turnHistory.Add(new TurnData {
                        BoardFen = _logic.GetStateAsFen(board, currentPlayer),
                        // The policy is a zero vector since there are no valid moves.
                        MctsPolicy = new float[AttaxConstants.MCTS.PolicyVectorSize]
                    });

                    currentPlayer = _logic.SwitchPlayer(currentPlayer);
                    continue;
                }

                AtaxxLogic.Move chosenMove;
                float[] policy;
                
                if (_mctsEngine != null) {
                    var mctsResult = _mctsEngine.FindBestMove(board, currentPlayer);
                    chosenMove = mctsResult.bestMove;
                    policy = mctsResult.policy;
                } else {
                    chosenMove = validMoves[_random.Next(validMoves.Count)];

                    policy = new float[AttaxConstants.MCTS.PolicyVectorSize];
                    float probability = 1.0f / validMoves.Count;
                    foreach (var move in validMoves) {
                        policy[_logic.MoveToIndex(move)] = probability;
                    }
                }

                turnHistory.Add(new TurnData {
                    BoardFen = _logic.GetStateAsFen(board, currentPlayer),
                    MctsPolicy = policy
                });

                board = _logic.MakeMove(board, chosenMove, currentPlayer, useOrthogonalCapture: false);
                currentPlayer = _logic.SwitchPlayer(currentPlayer);
            }

            return ProcessGameHistory(turnHistory, board);
        }

        private TrainingGameLog ProcessGameHistory(List<TurnData> turnHistory, BitboardState finalBoard) {
            var gameLog = new TrainingGameLog();

            int redCount = _logic.PopCount(finalBoard.RedPieces);
            int blueCount = _logic.PopCount(finalBoard.BluePieces);
            float condition;
            if (blueCount > redCount) {
                condition = -1.0f;
            } else {
                condition = 0.0f;
            }
            float gameOutcome = (redCount > blueCount) ? 1.0f : condition;

            const float epsilon = 1e-6f; // Define a small range for floating-point comparison
            foreach (var turn in turnHistory) {
                var turnPlayer = _logic.GetStateFromFen(turn.BoardFen).Item2;
                turn.GameOutcome = (turnPlayer == AtaxxLogic.PlayerColor.Red) ? gameOutcome : -gameOutcome;
                gameLog.Turns.Add(turn);
            }

            gameLog.DidRedWin = gameOutcome > 0;
            gameLog.DidBlueWin = gameOutcome < 0;
            gameLog.IsDraw = Math.Abs(gameOutcome) < epsilon; // Use range-based comparison for floating-point equality

            return gameLog;
        }
    }
}