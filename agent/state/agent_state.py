from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """
    Shared state that flows through every node in the LangGraph.

    LangGraph uses TypedDict for state. The `add_messages` reducer
    appends to the messages list rather than overwriting it — this
    preserves the full conversation and tool call history for the LLM.
    """

    # Full message history: HumanMessage, AIMessage, ToolMessage
    messages: Annotated[list[BaseMessage], add_messages]

    # The ticker symbol being researched (e.g. "AAPL")
    ticker: str

    # Structured outputs from each tool — populated as agent runs
    price_data: dict[str, Any]
    company_info: dict[str, Any]
    financials: dict[str, Any]
    news_articles: list[dict[str, Any]]
    sec_filings: list[dict[str, Any]]
    portfolio_metrics: dict[str, Any]

    # Final synthesised research brief — populated in the final node
    research_brief: str

    # Error tracking — non-fatal tool errors collected here
    errors: list[str]

    # Iteration guard — prevents runaway loops
    iteration_count: int
