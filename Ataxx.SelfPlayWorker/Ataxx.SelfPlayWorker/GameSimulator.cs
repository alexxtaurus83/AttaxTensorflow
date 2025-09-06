using Ataxx.Core;

namespace Ataxx.SelfPlayWorker
{
    // A player that makes random moves.
    public class RandomPlayer : IPlayer
    {
        private readonly AtaxxLogic _logic;
        private readonly Random _random = new Random();

        public RandomPlayer(AtaxxLogic logic)
        {
            _logic = logic;
        }

        public PlayerMove FindBestMove(BitboardState board, AtaxxLogic.PlayerColor currentPlayer)
        {
            var validMoves = _logic.GetAllValidMoves(board, currentPlayer);
            var chosenMove = validMoves[_random.Next(validMoves.Count)];

            // Create a uniform policy for training purposes
            var policy = new float[AttaxConstants.MCTS.PolicyVectorSize];
            float probability = 1.0f / validMoves.Count;
            foreach (var move in validMoves)
            {
                policy[_logic.MoveToIndex(move)] = probability;
            }

            return new PlayerMove { Move = chosenMove, Policy = policy };
        }
    }

    public class GameSimulator {
        private readonly AtaxxLogic _logic;
        private readonly Program.Options _options;
        private readonly MctsEngine? _mctsEngine;

        public GameSimulator(AtaxxLogic logic, Program.Options options, MctsEngine? mctsEngine) {
            _logic = logic;
            _options = options;
            _mctsEngine = mctsEngine;
        }

        public TrainingGameLog PlayGame() {
            IPlayer player;
            if (_mctsEngine != null)
            {
                player = new MctsPlayer(_mctsEngine);
            }
            else
            {
                player = new RandomPlayer(_logic);
            }

            // For self-play, both red and blue are the same player.
            var match = new MatchPlayer(_logic, player, player);
            var (gameOutcome, turnHistory) = match.PlayAndGetTrainingData(_options.BlockedCells);

            return ProcessGameHistory(turnHistory, gameOutcome);
        }

        private TrainingGameLog ProcessGameHistory(List<TurnData> turnHistory, float gameOutcome) {
            var gameLog = new TrainingGameLog();

            const float epsilon = 1e-6f;
            foreach (var turn in turnHistory) {
                var turnPlayer = _logic.GetStateFromFen(turn.BoardFen).Item2;
                // The outcome is from Red's perspective (1.0 = Red win).
                // If the current turn's player is Blue, we flip the outcome.
                turn.GameOutcome = (turnPlayer == AtaxxLogic.PlayerColor.Red) ? gameOutcome : -gameOutcome;
                gameLog.Turns.Add(turn);
            }

            gameLog.DidRedWin = gameOutcome > epsilon;
            gameLog.DidBlueWin = gameOutcome < -epsilon;
            gameLog.IsDraw = Math.Abs(gameOutcome) < epsilon;

            return gameLog;
        }
    }
}