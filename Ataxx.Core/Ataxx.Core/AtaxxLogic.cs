using System;
using System.Collections.Generic;
using System.Linq;
using static Ataxx.Core.AttaxConstants.Board;

namespace Ataxx.Core {
    /// <summary>
    /// Contains the core, stateless logic and rules for the game of Attax.
    /// This class does NOT contain any AI search algorithms.
    /// </summary>
    public class AtaxxLogic
    {
        #region Enums and Structs
        public enum PlayerColor { None, Red, Blue, Blocked }

        public struct Move
        {
            public int FromX, FromY, ToX, ToY;
            public Move(int fx, int fy, int tx, int ty)
            {
                FromX = fx; FromY = fy; ToX = tx; ToY = ty;
            }
            public override bool Equals(object obj) => obj is Move other && FromX == other.FromX && FromY == other.FromY && ToX == other.ToX && ToY == other.ToY;
            public override int GetHashCode() => HashCode.Combine(FromX, FromY, ToX, ToY);
            public static bool operator ==(Move left, Move right) => left.Equals(right);
            public static bool operator !=(Move left, Move right) => !(left == right);
        }
        #endregion

        #region Precomputed Move Data
        private readonly ulong[] _singleStepMoves;
        private readonly ulong[] _twoStepMoves;
        private readonly ulong[] _orthogonalStepMoves;
        #endregion

        public AtaxxLogic()
        {
            // Precompute move masks for performance
            _singleStepMoves = new ulong[TotalSquares];
            _twoStepMoves = new ulong[TotalSquares];
            _orthogonalStepMoves = new ulong[TotalSquares];

            for (int y = 0; y < BoardSize; y++)
            {
                for (int x = 0; x < BoardSize; x++)
                {
                    int fromIndex = y * BoardSize + x;
                    ulong singleStepMask = 0UL;
                    ulong twoStepMask = 0UL;
                    ulong orthoMask = 0UL;

                    for (int dy = -2; dy <= 2; dy++)
                    {
                        for (int dx = -2; dx <= 2; dx++)
                        {
                            if (dx == 0 && dy == 0) continue;

                            int toX = x + dx;
                            int toY = y + dy;

                            if (toX >= 0 && toX < BoardSize && toY >= 0 && toY < BoardSize)
                            {
                                int toIndex = toY * BoardSize + toX;
                                int distance = Math.Max(Math.Abs(dx), Math.Abs(dy));
                                if (distance == 1) singleStepMask |= (1UL << toIndex);
                                else if (distance == 2) twoStepMask |= (1UL << toIndex);
                            }
                        }
                    }
                    _singleStepMoves[fromIndex] = singleStepMask;
                    _twoStepMoves[fromIndex] = twoStepMask;

                    if (x > 0) orthoMask |= (1UL << (fromIndex - 1));
                    if (x < BoardSize - 1) orthoMask |= (1UL << (fromIndex + 1));
                    if (y > 0) orthoMask |= (1UL << (fromIndex - BoardSize));
                    if (y < BoardSize - 1) orthoMask |= (1UL << (fromIndex + BoardSize));
                    _orthogonalStepMoves[fromIndex] = orthoMask;
                }
            }
        }

        #region Core Game Rules

        public List<Move> GetAllValidMoves(BitboardState boardState, PlayerColor player)
        {
            var moves = new List<Move>();
            ulong playerPieces = (player == PlayerColor.Red) ? boardState.RedPieces : boardState.BluePieces;
            ulong emptySquares = boardState.EmptySquares();

            ulong remainingPieces = playerPieces;
            while (remainingPieces > 0)
            {
                int fromIndex = TrailingZeroCount(remainingPieces);
                int fromX = fromIndex % BoardSize;
                int fromY = fromIndex / BoardSize;

                ulong validCloneMoves = _singleStepMoves[fromIndex] & emptySquares;
                ulong remainingClones = validCloneMoves;
                while (remainingClones > 0)
                {
                    int toIndex = TrailingZeroCount(remainingClones);
                    moves.Add(new Move(fromX, fromY, toIndex % BoardSize, toIndex / BoardSize));
                    remainingClones &= remainingClones - 1;
                }

                ulong validJumpMoves = _twoStepMoves[fromIndex] & emptySquares;
                ulong remainingJumps = validJumpMoves;
                while (remainingJumps > 0)
                {
                    int toIndex = TrailingZeroCount(remainingJumps);
                    moves.Add(new Move(fromX, fromY, toIndex % BoardSize, toIndex / BoardSize));
                    remainingJumps &= remainingJumps - 1;
                }
                remainingPieces &= remainingPieces - 1;
            }
            return moves;
        }

        public BitboardState MakeMove(BitboardState boardState, Move move, PlayerColor player, bool useOrthogonalCapture)
        {
            var newState = boardState.Clone();
            int fromIndex = GetBitIndex(move.FromX, move.FromY);
            int toIndex = GetBitIndex(move.ToX, move.ToY);
            ulong fromMask = 1UL << fromIndex;
            ulong toMask = 1UL << toIndex;

            ref ulong playerPieces = ref (player == PlayerColor.Red ? ref newState.RedPieces : ref newState.BluePieces);
            ref ulong opponentPieces = ref (player == PlayerColor.Red ? ref newState.BluePieces : ref newState.RedPieces);

            bool isClone = IsCloneMove(move);
            if (!isClone)
            {
                playerPieces &= ~fromMask;
            }
            playerPieces |= toMask;

            ulong attackMask = useOrthogonalCapture ? _orthogonalStepMoves[toIndex] : _singleStepMoves[toIndex];
            ulong flippedPieces = opponentPieces & attackMask;

            if (flippedPieces > 0)
            {
                playerPieces |= flippedPieces;
                opponentPieces &= ~flippedPieces;
            }
            return newState;
        }
        
        public bool IsGameOver(BitboardState boardState)
        {
            int redCount = PopCount(boardState.RedPieces);
            int blueCount = PopCount(boardState.BluePieces);
            int emptyCount = PopCount(boardState.EmptySquares());

            if (redCount == 0 || blueCount == 0 || emptyCount == 0) return true;

            if (GetAllValidMoves(boardState, PlayerColor.Red).Count == 0 &&
                GetAllValidMoves(boardState, PlayerColor.Blue).Count == 0) return true;

            return false;
        }

        #endregion

        #region Helper Methods

        public bool IsCloneMove(Move move) => Math.Abs(move.ToX - move.FromX) <= 1 && Math.Abs(move.ToY - move.FromY) <= 1;
        public int GetBitIndex(int x, int y) => y * BoardSize + x;
        public PlayerColor SwitchPlayer(PlayerColor current) => current == PlayerColor.Red ? PlayerColor.Blue : PlayerColor.Red;
        public int PopCount(ulong value)
        {
            int count = 0;
            while (value > 0)
            {
                value &= (value - 1);
                count++;
            }
            return count;
        }
        private int TrailingZeroCount(ulong value)
        {
            if (value == 0) return 64;
            return PopCount((value & (0 - value)) - 1);
        }
        
        #endregion

        #region State & Policy Representation

        /// <summary>
        /// Converts a board state into a compact string for logging.
        /// Format: "RedPieces_ulong;BluePieces_ulong;BlockedSquares_ulong;CurrentPlayer_char"
        /// </summary>
        public string GetStateAsFen(BitboardState boardState, PlayerColor currentPlayer)
        {
            char playerChar = currentPlayer == PlayerColor.Red ? 'r' : 'b';
            return $"{boardState.RedPieces};{boardState.BluePieces};{boardState.BlockedSquares};{playerChar}";
        }

        /// <summary>
        /// Creates a BitboardState from a FEN string.
        /// </summary>
        public (BitboardState, PlayerColor) GetStateFromFen(string fen)
        {
            var parts = fen.Split(';');
            var boardState = new BitboardState
            {
                RedPieces = ulong.Parse(parts[0]),
                BluePieces = ulong.Parse(parts[1]),
                BlockedSquares = ulong.Parse(parts[2])
            };
            var player = parts[3][0] == 'r' ? PlayerColor.Red : PlayerColor.Blue;
            return (boardState, player);
        }

        #endregion

        #region Policy Vector Mapping
        
        private static readonly List<(int dx, int dy)> RelativeMoves = GenerateRelativeMoves();

        private static List<(int dx, int dy)> GenerateRelativeMoves() {
            var moves = new List<(int dx, int dy)>();
            for (int dy = -2; dy <= 2; dy++) {
                for (int dx = -2; dx <= 2; dx++) {
                    if (dx == 0 && dy == 0) continue; // Skip the center square
                    moves.Add((dx, dy));
                }
            }
            // The list will now correctly contain all 24 possible relative moves.
            return moves;
        }



        public int MoveToIndex(Move move) {
            int fromIndex = GetBitIndex(move.FromX, move.FromY);
            int dx = move.ToX - move.FromX;
            int dy = move.ToY - move.FromY;

            int relativeMoveIndex = -1;
            for (int i = 0; i < RelativeMoves.Count; i++) {
                if (RelativeMoves[i].dx == dx && RelativeMoves[i].dy == dy) {
                    relativeMoveIndex = i;
                    break;
                }
            }

            if (relativeMoveIndex == -1) {
                // Add logging for this invalid case as you suggested.
                // It should not happen with valid moves from GetAllValidMoves.
                return -1;
            }

            return fromIndex * 24 + relativeMoveIndex;
        }

        public Move IndexToMove(int policyIndex) {
            int fromSquareIndex = policyIndex / 24;
            int relativeMoveIndex = policyIndex % 24;

            int fromX = fromSquareIndex % BoardSize;
            int fromY = fromSquareIndex / BoardSize;

            var (dx, dy) = RelativeMoves[relativeMoveIndex];

            return new Move(fromX, fromY, fromX + dx, fromY + dy);
        }

        #endregion

        // Add this region inside the AtaxxLogic class
        #region Initial Board Setup

        public BitboardState CreateInitialBoard(int blockedCellCount = 0) {
            var board = new BitboardState();

            // Standard starting positions
            board.RedPieces = (1UL << GetBitIndex(0, 0)) | (1UL << GetBitIndex(6, 6));
            board.BluePieces = (1UL << GetBitIndex(0, 6)) | (1UL << GetBitIndex(6, 0));

            // Add random blocks if requested
            if (blockedCellCount > 0) {
                var random = new Random();
                ulong emptySquares = board.EmptySquares();
                int squaresToBlock = Math.Min(blockedCellCount, PopCount(emptySquares));

                var emptyIndices = new List<int>();
                for (int i = 0; i < TotalSquares; i++) {
                    if ((emptySquares & (1UL << i)) != 0) {
                        emptyIndices.Add(i);
                    }
                }

                for (int i = 0; i < squaresToBlock; i++) {
                    if (emptyIndices.Count == 0) break;
                    int randomIndex = random.Next(emptyIndices.Count);
                    int squareToBlock = emptyIndices[randomIndex];
                    emptyIndices.RemoveAt(randomIndex);
                    board.BlockedSquares |= (1UL << squareToBlock);
                }
            }
            return board;
        }

        #endregion
    }
}