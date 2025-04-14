from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from .routes import stats, knowledge  # Import the knowledge router
from .routes import ngrok  # Import the ngrok router
import logging

logger = logging.getLogger(__name__)

app = FastAPI()

# CORS settings
origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Debug routes before including routers
logger.info("=== Routes before including routers ===")
for route in app.routes:
    logger.info(f"Route: {route.path} - Methods: {route.methods}")

# Include routers with explicit logging
logger.info("Including stats router with prefix: " + stats.router.prefix)
app.include_router(stats.router)

logger.info("Including knowledge router with prefix: " + knowledge.router.prefix)
app.include_router(knowledge.router)

# Include the ngrok router
logger.info("Including ngrok router with prefix: " + ngrok.router.prefix)
app.include_router(ngrok.router)

# Import and include the new Ollama router
from .routes import ollama
logger.info("Including ollama router with prefix: " + ollama.router.prefix)
app.include_router(ollama.router)

# Import the goals router with better logging
from api.routes.goals import router as goals_router
logger.info("Including goals router with prefix: " + goals_router.prefix)
app.include_router(goals_router)

# Add explicit direct route for API stability
from api.routes.goals import list_goals, list_topics, vote_on_goal
logger.info("Adding direct mappings for goals endpoints")
app.add_api_route("/api/goals", list_goals, methods=["GET"])
app.add_api_route("/api/goals/", list_goals, methods=["GET"])  # Add trailing slash version too
app.add_api_route("/api/goals/topics", list_topics, methods=["GET"])
app.add_api_route("/api/goals/topics/", list_topics, methods=["GET"])  # With trailing slash

# Add direct mapping from /api/data/stats/knowledge to the actual function
# This ensures we have a stable URL that won't break if routes are moved
from .routes.stats import get_knowledge_stats
logger.info("Adding direct mapping for /api/data/stats/knowledge")
app.add_api_route("/api/data/stats/knowledge", get_knowledge_stats, methods=["GET"])

# Debug routes after including routers - more detailed
logger.info("=== Routes after including routers ===")
for route in app.routes:
    logger.info(f"Route: {route.path} - Methods: {route.methods}")

@app.get("/")
async def root():
    return {"message": "Welcome to the API"}

# Add a diagnostic endpoint for debugging routes
@app.get("/api/debug/routes")
async def list_routes():
    """List all registered routes for debugging"""
    routes = []
    for route in app.routes:
        routes.append({
            "path": route.path,
            "methods": list(route.methods) if route.methods else [],
            "name": route.name
        })
    return {"routes": routes}