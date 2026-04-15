import logging
from datetime import datetime
from typing import Any

import yfinance as yf
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def get_recent_news(ticker: str, max_articles: int = 5) -> dict[str, Any]:
    """
    Fetch recent news articles and headlines for a stock ticker.
    Includes title, publisher, summary, and publish timestamp.

    Args:
        ticker: Stock ticker symbol, e.g. 'AAPL'
        max_articles: Maximum number of articles to return (default 5)

    Returns:
        Dict containing a list of news articles with metadata.
    """
    try:
        logger.info("[NewsTool] Fetching news for ticker=%s max=%d", ticker, max_articles)
        stock = yf.Ticker(ticker.upper())
        raw_news = stock.news or []

        articles = []
        for item in raw_news[:max_articles]:
            # yfinance news structure
            content = item.get("content", {})
            articles.append({
                "title": content.get("title") or item.get("title", ""),
                "publisher": (content.get("provider") or {}).get("displayName", ""),
                "summary": (content.get("summary") or "")[:300],
                "url": (content.get("canonicalUrl") or {}).get("url", ""),
                "published_at": content.get("pubDate", ""),
            })

        return {
            "ticker": ticker.upper(),
            "article_count": len(articles),
            "articles": articles,
            "fetched_at": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error("[NewsTool] Error fetching news for %s: %s", ticker, e)
        return {"error": str(e), "ticker": ticker, "articles": []}


@tool
def get_earnings_calendar(ticker: str) -> dict[str, Any]:
    """
    Fetch upcoming and recent earnings dates and estimates for a ticker.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dict with next earnings date, EPS estimates, and earnings history.
    """
    try:
        logger.info("[NewsTool] Fetching earnings calendar for ticker=%s", ticker)
        stock = yf.Ticker(ticker.upper())
        info = stock.info

        # Earnings history
        hist = stock.earnings_history
        earnings_history = []
        if hist is not None and not hist.empty:
            for _, row in hist.head(4).iterrows():
                earnings_history.append({
                    "date": str(row.name) if hasattr(row, "name") else "",
                    "eps_actual": row.get("epsActual"),
                    "eps_estimate": row.get("epsEstimate"),
                    "surprise_pct": row.get("surprisePercent"),
                })

        return {
            "ticker": ticker.upper(),
            "next_earnings_date": info.get("earningsTimestamp"),
            "eps_estimate_current_year": info.get("epsCurrentYear"),
            "eps_estimate_next_year": info.get("epsNextYear"),
            "earnings_quarterly_growth": info.get("earningsQuarterlyGrowth"),
            "recent_earnings": earnings_history,
        }

    except Exception as e:
        logger.error("[NewsTool] Error fetching earnings for %s: %s", ticker, e)
        return {"error": str(e), "ticker": ticker}
