

namespace Ataxx.Core {
    public class AttaxConstants {

        public static class Board {
            public const int BoardSize = 7;
            public const int TotalSquares = BoardSize * BoardSize;
        }

        public static class MCTS  {
            // We will tune these values later in the project
            public const int NumSimulations = 100; // Number of MCTS simulations per move
            public const double Cpuct = 1.41;      // Exploration-exploitation trade-off constant
            public const int PolicyVectorSize = 1176; // 49 'from' squares * 24 'to' squares        
        }

    }
}