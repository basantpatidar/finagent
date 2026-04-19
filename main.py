import uvicorn
from finagent.api.app import app
from finagent.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "finagent.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_debug,
        log_level="info",
    )
