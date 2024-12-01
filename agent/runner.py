import logging
from typing import AsyncIterator
from langchain_core.messages import HumanMessage

from finagent.agent.graph.agent_graph import finagent_graph
from finagent.agent.state import AgentState

logger = logging.getLogger(__name__)


def _initial_state(ticker: str, question: str | None = None) -> AgentState:
    """Build the initial AgentState for a new research run."""
    prompt = question or f"Please produce a comprehensive research brief for {ticker.upper()}."
    return AgentState(
        messages=[HumanMessage(content=prompt)],
        ticker=ticker.upper(),
        price_data={},
        company_info={},
        financials={},
        news_articles=[],
        sec_filings=[],
        portfolio_metrics={},
        research_brief="",
        errors=[],
        iteration_count=0,
    )


def run_research(ticker: str, question: str | None = None) -> AgentState:
    """
    Run the full FinAgent research pipeline synchronously.

    Args:
        ticker: Stock ticker symbol e.g. 'AAPL'
        question: Optional custom research question

    Returns:
        Final AgentState with research_brief populated.
    """
    logger.info("[Runner] Starting research run for ticker=%s", ticker)
    initial = _initial_state(ticker, question)
    final_state = finagent_graph.invoke(initial)
    logger.info(
        "[Runner] Research complete. brief_length=%d errors=%d",
        len(final_state.get("research_brief", "")),
        len(final_state.get("errors", [])),
    )
    return final_state


async def stream_research(
    ticker: str, question: str | None = None
) -> AsyncIterator[dict]:
    """
    Stream the FinAgent research pipeline as server-sent events.

    Yields dicts with:
        - type: 'tool_start' | 'tool_end' | 'text_chunk' | 'complete' | 'error'
        - data: relevant payload for each event type

    Args:
        ticker: Stock ticker symbol
        question: Optional custom research question
    """
    logger.info("[Runner] Starting streaming research run for ticker=%s", ticker)
    initial = _initial_state(ticker, question)

    try:
        async for event in finagent_graph.astream(initial, stream_mode="updates"):
            for node_name, node_output in event.items():

                if node_name == "tools":
                    # Emit an event for each tool that ran
                    messages = node_output.get("messages", [])
                    for msg in messages:
                        yield {
                            "type": "tool_end",
                            "data": {
                                "tool_call_id": getattr(msg, "tool_call_id", ""),
                                "content_preview": str(msg.content)[:100] + "...",
                            },
                        }

                elif node_name == "llm":
                    messages = node_output.get("messages", [])
                    for msg in messages:
                        # Stream tool call intentions
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                yield {
                                    "type": "tool_start",
                                    "data": {
                                        "tool": tc["name"],
                                        "args": tc["args"],
                                    },
                                }
                        # Stream text chunks if present
                        elif hasattr(msg, "content") and isinstance(msg.content, str):
                            yield {
                                "type": "text_chunk",
                                "data": {"text": msg.content},
                            }

                elif node_name == "synthesis":
                    brief = node_output.get("research_brief", "")
                    yield {
                        "type": "complete",
                        "data": {
                            "research_brief": brief,
                            "ticker": ticker.upper(),
                        },
                    }

    except Exception as e:
        logger.error("[Runner] Streaming error for ticker=%s: %s", ticker, e, exc_info=True)
        yield {
            "type": "error",
            "data": {"message": str(e), "ticker": ticker},
        }
