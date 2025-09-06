using System;
using System.Collections.Generic;

namespace Ataxx.Core
{
    public struct PlayerMove
    {
        public AtaxxLogic.Move Move { get; set; }
        public float[] Policy { get; set; }
    }

    // Defines a common interface for any "player" that can choose a move.
    public interface IPlayer
    {
        PlayerMove FindBestMove(BitboardState board, AtaxxLogic.PlayerColor currentPlayer);
    }

    // A wrapper for MctsEngine to conform to the IPlayer interface.
    public class MctsPlayer : IPlayer
    {
        private readonly MctsEngine _mctsEngine;

        public MctsPlayer(MctsEngine mctsEngine)
        {
            _mctsEngine = mctsEngine;
        }

        public PlayerMove FindBestMove(BitboardState board, AtaxxLogic.PlayerColor currentPlayer)
        {
            var result = _mctsEngine.FindBestMove(board, currentPlayer);
            return new PlayerMove { Move = result.bestMove, Policy = result.policy };
        }
    }

    public class MatchPlayer
    {
        private readonly AtaxxLogic _logic;
        private readonly IPlayer _redPlayer;
        private readonly IPlayer _bluePlayer;

        public MatchPlayer(AtaxxLogic logic, IPlayer redPlayer, IPlayer bluePlayer)
        {
            _logic = logic;
            _redPlayer = redPlayer;
            _bluePlayer = bluePlayer;
        }

        public (float, List<TurnData>) PlayAndGetTrainingData(int blockedCells = 0)
        {
            var board = _logic.CreateInitialBoard(blockedCells);
            board.RedPieces = (1UL << _logic.GetBitIndex(0, 0)) | (1UL << _logic.GetBitIndex(6, 6));
            board.BluePieces = (1UL << _logic.GetBitIndex(0, 6)) | (1UL << _logic.GetBitIndex(6, 0));

            var currentPlayer = AtaxxLogic.PlayerColor.Red;
            var turnHistory = new List<TurnData>();

            while (!_logic.IsGameOver(board))
            {
                var engineToUse = (currentPlayer == AtaxxLogic.PlayerColor.Red) ? _redPlayer : _bluePlayer;
                var validMoves = _logic.GetAllValidMoves(board, currentPlayer);

                if (validMoves.Count == 0)
                {
                    turnHistory.Add(new TurnData
                    {
                        BoardFen = _logic.GetStateAsFen(board, currentPlayer),
                        MctsPolicy = new float[AttaxConstants.MCTS.PolicyVectorSize]
                    });
                    currentPlayer = _logic.SwitchPlayer(currentPlayer);
                    continue;
                }

                var playerMove = engineToUse.FindBestMove(board, currentPlayer);
                
                turnHistory.Add(new TurnData
                {
                    BoardFen = _logic.GetStateAsFen(board, currentPlayer),
                    MctsPolicy = playerMove.Policy
                });

                board = _logic.MakeMove(board, playerMove.Move, currentPlayer, false);
                currentPlayer = _logic.SwitchPlayer(currentPlayer);
            }

            int redCount = _logic.PopCount(board.RedPieces);
            int blueCount = _logic.PopCount(board.BluePieces);

            float result;
            if (redCount > blueCount) result = 1.0f;
            else if (blueCount > redCount) result = -1.0f;
            else result = 0.0f;

            return (result, turnHistory);
        }

        public float Play() => PlayAndGetTrainingData().Item1;
    }
}