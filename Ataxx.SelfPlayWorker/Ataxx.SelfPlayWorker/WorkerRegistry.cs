using FluentScheduler;

namespace Ataxx.SelfPlayWorker {
    public class WorkerRegistry : Registry {
        public WorkerRegistry(Program.Options options) {
            // This schedule will run the SelfPlayJob immediately,
            // and then schedule the next run immediately after the previous one completes.
            // This creates a continuous loop.
            Schedule(() => new SelfPlayJob(options)).NonReentrant().ToRunNow().AndEvery(1).Seconds();
        }
    }
}