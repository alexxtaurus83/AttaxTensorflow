

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



        // We can keep these for reference or potential future use,
        // but they are not used by the new MCTS engine.        
        public static class OldHeuristicConst {
            public const int earlyGame = 14;
            public const int lateGame = 35;
            public const int flipWeight = 25;
            public const int riskAversionFactor = 13;
        }
    }
}