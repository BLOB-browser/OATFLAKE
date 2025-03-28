from .auth import router as auth_router
from .data import router as data_router
from .system import router as system_router
from .training import router as training_router
from .slack import router as slack_router
from .questions import router as questions_router
from .markdown import router as markdown_router

# Export all routers
__all__ = [
    'auth_router',
    'data_router',
    'system_router',
    'training_router',
    'slack_router',
    'questions_router',
    'markdown_router'
]