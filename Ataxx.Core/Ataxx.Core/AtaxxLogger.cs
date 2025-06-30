
using Serilog;
using System.IO;
# if UNITY_2017_1_OR_NEWER
using UnityEngine;
#endif
namespace Ataxx.Core {
    public class AtaxxLogger {
        public void InitAtaxxLogger() {
            Serilog.Log.Logger = new LoggerConfiguration()
              .MinimumLevel.Information()
              .WriteTo.File(Path.Combine("debug.log"))
#if UNITY_2017_1_OR_NEWER
              .WriteTo.File(Path.Combine(Application.persistentDataPath, "serilogs", "debug.log"))
#endif
              .CreateLogger();
        }
        public void WriteTolog(string logline, bool useSerilog = true) {
            if (useSerilog) { Log.Information(logline); } else {
#if UNITY_2017_1_OR_NEWER
                Debug.Log(logline);
#endif
            }
        }
    }
}
