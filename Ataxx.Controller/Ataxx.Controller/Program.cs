
using FluentScheduler;

namespace Ataxx.Controller
{
    public class Program
    {
        public static void Main(string[] args)
        {
            var builder = WebApplication.CreateBuilder(args);

            // --- Kestrel Configuration ---
            // Explicitly configure the Kestrel web server to listen on a specific HTTP port.
            // This makes the controller accessible from other machines on your network.
            builder.WebHost.ConfigureKestrel(serverOptions =>
            {
                // Listen on port 5299 on all available network interfaces (localhost, LAN IP, etc.)
                // You can change this port number if you wish.
                serverOptions.Listen(System.Net.IPAddress.Any, 5299);
            });


            // --- Service Configuration ---

            // This line remains the same. It registers our service for dependency injection.
            builder.Services.AddSingleton<ModelRegistryService>();
            builder.Services.AddTransient<EvaluationJob>();

            // This line tells the application to discover and use our new Controller classes.
            builder.Services.AddControllers();

            builder.Services.AddEndpointsApiExplorer();
            builder.Services.AddSwaggerGen();

            var app = builder.Build();

            // --- Middleware Pipeline Configuration ---

            app.UseSwagger();
            app.UseSwaggerUI();

            app.UseAuthorization();

            // This line is the key. It automatically maps the routes from our controller classes.
            app.MapControllers();
            app.MapGet("/health", () => Results.Ok(new { Status = "Healthy", Timestamp = DateTime.UtcNow }));

            var serviceProvider = app.Services;

            var registry = new Registry();
            // Schedule the job to run 30 seconds after startup, and then every 15 minutes.
            registry.Schedule(() => serviceProvider.GetRequiredService<EvaluationJob>()).NonReentrant().ToRunOnceAt(DateTime.Now.AddSeconds(30)).AndEvery(15).Minutes();

            // Start the scheduler
            JobManager.Initialize(registry);


            app.Run();
        }
    }
}
