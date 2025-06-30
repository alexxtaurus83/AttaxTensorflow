using Ataxx.Core;
using NumSharp;
using static Ataxx.Core.AtaxxLogic;

namespace Ataxx.Trainer {
    public static class DataPreprocessor {
        public static NDArray FenToInputTensor(AtaxxLogic logic, string fen) {
            var (boardState, currentPlayer) = logic.GetStateFromFen(fen);
            var tensor = np.zeros(new Shape(7, 7, 4), np.float32);

            for (int i = 0; i < AttaxConstants.Board.TotalSquares; i++) {
                int x = i % AttaxConstants.Board.BoardSize;
                int y = i / AttaxConstants.Board.BoardSize;
                ulong bit = 1UL << i;

                // CORRECTED: Applying normalization by mapping {0, 1} to {-0.5, 0.5}
                const float normFactor = 0.5f;

                if (currentPlayer == PlayerColor.Red) {
                    tensor[y, x, 0] = ((boardState.RedPieces & bit) != 0 ? 1 : 0) - normFactor;
                    tensor[y, x, 1] = ((boardState.BluePieces & bit) != 0 ? 1 : 0) - normFactor;
                } else {
                    tensor[y, x, 0] = ((boardState.BluePieces & bit) != 0 ? 1 : 0) - normFactor;
                    tensor[y, x, 1] = ((boardState.RedPieces & bit) != 0 ? 1 : 0) - normFactor;
                }

                tensor[y, x, 2] = ((boardState.BlockedSquares & bit) != 0 ? 1 : 0) - normFactor;
                tensor[y, x, 3] = (currentPlayer == PlayerColor.Red) ? 0.5f : -0.5f;
            }
            return tensor;
        }
    }
}