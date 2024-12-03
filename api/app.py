import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from finagent.api.models.schemas import HealthResponse
from finagent.api.routers.research_router import router as research_router
from finagent.config import settings

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FinAgent API starting up. model=%s", settings.claude_model)
    yield
    logger.info("FinAgent API shutting down.")


# ── App factory ────────────────────────────────────────────────────────────────
def create_app() -> FastAPI:
    app = FastAPI(
        title="FinAgent — Agentic Financial Research API",
        description=(
            "An LLM-powered financial research agent that autonomously fetches market data, "
            "SEC filings, and news to produce structured investment research briefs.\n\n"
            "Built with LangGraph + Claude + FastAPI."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS — allow frontend (FinDash) to call this API
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(research_router)

    @app.get("/health", response_model=HealthResponse, tags=["health"])
    async def health():
        return HealthResponse(model=settings.claude_model)

    return app


app = create_app()
