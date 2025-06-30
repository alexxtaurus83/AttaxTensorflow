using Microsoft.AspNetCore.Mvc;

namespace Ataxx.Controller.Controllers {
    [ApiController]
    [Route("api/[controller]")] // This sets the base route to "api/model"
    public class ModelController : ControllerBase {
        private readonly ModelRegistryService _registry;
        private readonly ILogger<ModelController> _logger;

        // The ModelRegistryService is automatically injected by the framework.
        public ModelController(ModelRegistryService registry, ILogger<ModelController> logger) {
            _registry = registry;
            _logger = logger;
        }

        // Handles GET requests to /api/model/latest
        [HttpGet("latest")]
        public IActionResult GetLatestModel() {
            _logger.LogInformation("Request received for latest model.");
            var modelName = _registry.GetBestModelName();
            return Ok(new { bestModelName = modelName });
        }

        // Handles POST requests to /api/model/promote/{modelName}
        [HttpPost("promote/{modelName}")]
        public IActionResult PromoteModel(string modelName) {
            try {
                _logger.LogInformation("Request received to promote model: {ModelName}", modelName);
                _registry.PromoteModel(modelName);
                _logger.LogInformation("Successfully promoted model: {ModelName}", modelName);
                return Ok($"Model '{modelName}' has been promoted to best.");
            } catch (DirectoryNotFoundException ex) {
                _logger.LogWarning(ex, "Failed to promote model. Directory not found for {ModelName}", modelName);
                return NotFound(ex.Message);
            } catch (Exception ex) {
                _logger.LogError(ex, "An unexpected error occurred while promoting model {ModelName}", modelName);
                return Problem(ex.Message);
            }
        }
    }
}