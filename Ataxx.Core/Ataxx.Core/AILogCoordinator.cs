using Newtonsoft.Json;
using System.Collections.Generic;

namespace Ataxx.Core {
    /// <summary>
    /// Represents a single turn's data, which is one sample for the neural network.
    /// </summary>
    public class TurnData
    {
        // The board state represented as a FEN-like string.
        public string BoardFen { get; set; }

        // The MCTS-improved policy vector. The array index maps to a specific move.
        public float[] MctsPolicy { get; set; }

        // The final outcome of the game from this position (+1 for win, -1 for loss, 0 for draw).
        public float GameOutcome { get; set; }
    }

    /// <summary>
    /// Contains all the training data for a single, complete game.
    /// This object will be serialized to JSON and saved to the Samba share.
    /// </summary>
    public class TrainingGameLog
    {
        public List<TurnData> Turns { get; set; } = new List<TurnData>();
        public bool DidRedWin { get; set; }
        public bool DidBlueWin { get; set; }
        public bool IsDraw { get; set; }

        public string ToJson()
        {
            return JsonConvert.SerializeObject(this);
        }

        public static TrainingGameLog FromJson(string json)
        {
            return JsonConvert.DeserializeObject<TrainingGameLog>(json);
        }
    }
}