import json
import logging
from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from finagent.agent.runner import run_research, stream_research
from finagent.api.models.schemas import ResearchRequest, ResearchResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/research", tags=["research"])


@router.post(
    "/",
    response_model=ResearchResponse,
    summary="Run full research pipeline (synchronous)",
    description=(
        "Runs the complete FinAgent pipeline for a ticker symbol. "
        "Calls 9 financial data tools, then synthesises a structured research brief. "
        "Typical response time: 30-60 seconds. Use /stream for real-time progress."
    ),
)
async def research(request: ResearchRequest) -> ResearchResponse:
    """
    POST /api/research/
    Synchronous research endpoint — waits for full pipeline completion.
    """
    logger.info("[ResearchRouter] POST /api/research ticker=%s", request.ticker)

    try:
        final_state = run_research(ticker=request.ticker, question=request.question)

        # Count tool-related messages
        tool_calls = sum(
            len(getattr(m, "tool_calls", []))
            for m in final_state.get("messages", [])
        )

        return ResearchResponse(
            ticker=request.ticker,
            research_brief=final_state.get("research_brief", ""),
            errors=final_state.get("errors", []),
            tool_calls_made=tool_calls,
        )

    except Exception as e:
        logger.error("[ResearchRouter] Error for ticker=%s: %s", request.ticker, e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/stream/{ticker}",
    summary="Stream research pipeline via SSE",
    description=(
        "Streams real-time progress of the FinAgent pipeline as Server-Sent Events. "
        "Events: tool_start, tool_end, text_chunk, complete, error. "
        "Connect with EventSource in the browser or curl --no-buffer."
    ),
)
async def stream(ticker: str, question: str | None = None) -> EventSourceResponse:
    """
    GET /api/research/stream/{ticker}
    Streaming endpoint — emits SSE events as each tool completes.

    Example:
        curl -N http://localhost:8000/api/research/stream/AAPL
    """
    ticker = ticker.upper().strip()
    logger.info("[ResearchRouter] GET /api/research/stream/%s", ticker)

    async def event_generator():
        async for event in stream_research(ticker=ticker, question=question):
            yield {
                "event": event["type"],
                "data": json.dumps(event["data"]),
            }

    return EventSourceResponse(event_generator())
