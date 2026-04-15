from finagent.tools.market.market_tools import (
    get_stock_price,
    get_price_history,
    get_company_fundamentals,
)
from finagent.tools.news.news_tools import (
    get_recent_news,
    get_earnings_calendar,
)
from finagent.tools.sec.sec_tools import (
    get_sec_filings,
    get_sec_facts,
)
from finagent.tools.portfolio.portfolio_tools import (
    get_portfolio_metrics,
    get_peer_comparison,
)

# All tools available to the agent
ALL_TOOLS = [
    get_stock_price,
    get_price_history,
    get_company_fundamentals,
    get_recent_news,
    get_earnings_calendar,
    get_sec_filings,
    get_sec_facts,
    get_portfolio_metrics,
    get_peer_comparison,
]

TOOL_MAP = {tool.name: tool for tool in ALL_TOOLS}

__all__ = ["ALL_TOOLS", "TOOL_MAP"]
