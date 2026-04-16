import logging
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage

from finagent.agent.state import AgentState
from finagent.config import settings
from finagent.tools import ALL_TOOLS

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are FinAgent, an expert financial research assistant backed by real-time
market data, SEC filings, and financial news.

Your job is to produce a comprehensive, structured research brief for a given stock ticker.

## Research Methodology

For every research request, you MUST call tools in this order:
1. `get_stock_price` — current price and daily movement
2. `get_company_fundamentals` — P/E, margins, analyst targets
3. `get_price_history` — 6-month chart data and moving averages
4. `get_recent_news` — latest headlines (last 5 articles)
5. `get_earnings_calendar` — upcoming earnings and recent surprises
6. `get_sec_filings` — recent 10-K / 10-Q filings
7. `get_sec_facts` — XBRL financial data (revenue, net income)
8. `get_portfolio_metrics` — Sharpe ratio, beta, max drawdown
9. `get_peer_comparison` — sector valuation context

Only after calling ALL tools above should you produce a final research brief.

## Research Brief Format

Structure your final brief with these exact sections:

**EXECUTIVE SUMMARY**
One paragraph: what does the company do, current price, key investment thesis.

**PRICE & TECHNICALS**
Current price, 52-week range, moving averages, trend direction, volatility.

**FUNDAMENTALS**
P/E vs sector, margins, revenue/earnings growth, debt levels, analyst consensus.

**RECENT DEVELOPMENTS**
Top 3 news headlines and their investment relevance. Upcoming earnings.

**SEC FILINGS ANALYSIS**
Revenue, net income, operating cash flow from latest filings. YoY changes.

**RISK METRICS**
Sharpe ratio, beta, max drawdown, correlation with SPY. Risk-adjusted return.

**INVESTMENT CONSIDERATIONS**
2-3 bull case points. 2-3 bear case / risk points. Analyst target vs current price.

**DISCLAIMER**
This is AI-generated research for informational purposes only. Not financial advice.

## Rules
- Always base every claim on actual tool output — never hallucinate numbers
- If a tool returns an error, note it and continue with available data
- Be concise but precise — numbers should include units (%, $, x for multiples)
- Flag any data that seems anomalous
"""


def build_llm_with_tools() -> ChatAnthropic:
    """Initialise Claude with all tools bound."""
    llm = ChatAnthropic(
        model=settings.claude_model,
        anthropic_api_key=settings.anthropic_api_key,
        temperature=settings.agent_temperature,
        max_tokens=4096,
    )
    return llm.bind_tools(ALL_TOOLS)


def llm_node(state: AgentState) -> dict:
    """
    Core LLM node — invokes Claude with the current message history
    and bound tools. Claude decides which tools to call next or
    produces the final research brief.
    """
    logger.info(
        "[LLMNode] Invoking Claude. iteration=%d messages=%d",
        state["iteration_count"],
        len(state["messages"]),
    )

    llm = build_llm_with_tools()

    # Prepend system message on first call only
    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

    response = llm.invoke(messages)

    logger.info(
        "[LLMNode] Response: stop_reason=%s tool_calls=%d",
        getattr(response, "response_metadata", {}).get("stop_reason", "unknown"),
        len(response.tool_calls) if hasattr(response, "tool_calls") else 0,
    )

    return {
        "messages": [response],
        "iteration_count": state["iteration_count"] + 1,
    }
