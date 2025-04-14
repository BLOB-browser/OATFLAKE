# Import the LLM manager
from scripts.utils.llm_manager import get_openrouter_client

# Instead of creating a new OpenRouterClient instance:
# client = OpenRouterClient()

# Use this instead:
client = get_openrouter_client()
