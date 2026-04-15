import logging
from datetime import datetime, timedelta
from typing import Any

import yfinance as yf
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def get_stock_price(ticker: str) -> dict[str, Any]:
    """
    Fetch current stock price, daily change, volume, and 52-week range
    for a given ticker symbol.

    Args:
        ticker: Stock ticker symbol, e.g. 'AAPL', 'MSFT', 'GOOGL'

    Returns:
        Dict with current price, change, volume, market cap, and 52-week range.
    """
    try:
        logger.info("[MarketTool] Fetching price for ticker=%s", ticker)
        stock = yf.Ticker(ticker.upper())
        info = stock.info

        # Fast quote data
        hist = stock.history(period="2d")
        if hist.empty:
            return {"error": f"No price data found for ticker '{ticker}'"}

        current = float(hist["Close"].iloc[-1])
        prev_close = float(hist["Close"].iloc[-2]) if len(hist) > 1 else current
        change = current - prev_close
        change_pct = (change / prev_close) * 100 if prev_close else 0

        return {
            "ticker": ticker.upper(),
            "current_price": round(current, 2),
            "previous_close": round(prev_close, 2),
            "change": round(change, 2),
            "change_percent": round(change_pct, 2),
            "volume": info.get("volume"),
            "avg_volume": info.get("averageVolume"),
            "market_cap": info.get("marketCap"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
            "currency": info.get("currency", "USD"),
            "as_of": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error("[MarketTool] Error fetching price for %s: %s", ticker, e)
        return {"error": str(e), "ticker": ticker}


@tool
def get_price_history(ticker: str, period: str = "6mo") -> dict[str, Any]:
    """
    Fetch historical OHLCV data and compute key moving averages.

    Args:
        ticker: Stock ticker symbol
        period: One of '1mo', '3mo', '6mo', '1y', '2y', '5y'

    Returns:
        Dict with OHLCV summary, moving averages, and volatility metrics.
    """
    try:
        logger.info("[MarketTool] Fetching history for ticker=%s period=%s", ticker, period)
        stock = yf.Ticker(ticker.upper())
        hist = stock.history(period=period)

        if hist.empty:
            return {"error": f"No historical data for '{ticker}'"}

        close = hist["Close"]

        # Moving averages
        ma20 = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else None
        ma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
        ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None

        # Volatility: annualised std dev of daily returns
        daily_returns = close.pct_change().dropna()
        volatility = float(daily_returns.std() * (252 ** 0.5) * 100) if len(daily_returns) > 1 else None

        # Period performance
        period_return = float(((close.iloc[-1] / close.iloc[0]) - 1) * 100)

        return {
            "ticker": ticker.upper(),
            "period": period,
            "start_date": hist.index[0].strftime("%Y-%m-%d"),
            "end_date": hist.index[-1].strftime("%Y-%m-%d"),
            "start_price": round(float(close.iloc[0]), 2),
            "end_price": round(float(close.iloc[-1]), 2),
            "period_return_pct": round(period_return, 2),
            "high": round(float(hist["High"].max()), 2),
            "low": round(float(hist["Low"].min()), 2),
            "avg_daily_volume": round(float(hist["Volume"].mean())),
            "ma_20": round(ma20, 2) if ma20 else None,
            "ma_50": round(ma50, 2) if ma50 else None,
            "ma_200": round(ma200, 2) if ma200 else None,
            "annualised_volatility_pct": round(volatility, 2) if volatility else None,
        }

    except Exception as e:
        logger.error("[MarketTool] Error fetching history for %s: %s", ticker, e)
        return {"error": str(e), "ticker": ticker}


@tool
def get_company_fundamentals(ticker: str) -> dict[str, Any]:
    """
    Fetch company overview and key fundamental metrics: P/E, EPS,
    revenue, profit margins, debt ratios, and analyst targets.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dict with company info and fundamental financial metrics.
    """
    try:
        logger.info("[MarketTool] Fetching fundamentals for ticker=%s", ticker)
        stock = yf.Ticker(ticker.upper())
        info = stock.info

        return {
            "ticker": ticker.upper(),
            "company_name": info.get("longName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "country": info.get("country"),
            "employees": info.get("fullTimeEmployees"),
            "description": (info.get("longBusinessSummary") or "")[:500],
            # Valuation
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "price_to_book": info.get("priceToBook"),
            "price_to_sales": info.get("priceToSalesTrailing12Months"),
            "ev_to_ebitda": info.get("enterpriseToEbitda"),
            # Earnings
            "eps_trailing": info.get("trailingEps"),
            "eps_forward": info.get("forwardEps"),
            "earnings_growth": info.get("earningsGrowth"),
            "revenue_growth": info.get("revenueGrowth"),
            # Margins
            "gross_margins": info.get("grossMargins"),
            "operating_margins": info.get("operatingMargins"),
            "profit_margins": info.get("profitMargins"),
            # Balance sheet
            "debt_to_equity": info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
            "return_on_equity": info.get("returnOnEquity"),
            "return_on_assets": info.get("returnOnAssets"),
            # Dividends
            "dividend_yield": info.get("dividendYield"),
            "payout_ratio": info.get("payoutRatio"),
            # Analyst consensus
            "analyst_target_price": info.get("targetMeanPrice"),
            "analyst_recommendation": info.get("recommendationKey"),
            "analyst_count": info.get("numberOfAnalystOpinions"),
            # Beta
            "beta": info.get("beta"),
        }

    except Exception as e:
        logger.error("[MarketTool] Error fetching fundamentals for %s: %s", ticker, e)
        return {"error": str(e), "ticker": ticker}
