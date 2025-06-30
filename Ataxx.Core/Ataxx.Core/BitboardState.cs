

namespace Ataxx.Core {

    public class BitboardState {        

        /// <summary>
        /// A bitboard where each set bit (1) represents a square occupied by a Red piece.
        /// </summary>
        public ulong RedPieces;

        /// <summary>
        /// A bitboard where each set bit (1) represents a square occupied by a Blue piece.
        /// </summary>
        public ulong BluePieces;

        /// <summary>
        /// A bitboard where each set bit (1) represents a permanently blocked square.
        /// </summary>
        public ulong BlockedSquares;
        
        // A mask representing all valid squares on the 7x7 board (bits 0-48).
        private const ulong BoardMask = (1UL << 49) - 1;

        public BitboardState() {
            RedPieces = 0UL;
            BluePieces = 0UL;
            BlockedSquares = 0UL;
        }

        /// <summary>
        /// Gets a bitboard representing all squares occupied by player pieces (Red or Blue).
        /// </summary>
        public ulong AllPlayerPieces() {
            return RedPieces | BluePieces;
        }

        /// <summary>
        /// Gets a bitboard representing all truly empty, playable squares.
        /// </summary>
        public ulong EmptySquares() {
            // A playable empty square is one not occupied by Red, Blue, OR a Blocked cell.
            return ~(RedPieces | BluePieces | BlockedSquares) & BoardMask;
        }

        /// <summary>
        /// Creates an exact, deep copy of the current bitboard state.
        /// This is essential for the parallel search.
        /// </summary>
        public BitboardState Clone() {
            return new BitboardState {
                RedPieces = this.RedPieces,
                BluePieces = this.BluePieces,
                BlockedSquares = this.BlockedSquares                
            };
        }
    }
}
