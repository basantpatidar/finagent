import logging
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

RISK_FREE_RATE_ANNUAL = 0.05  # ~5% approximation (update from Fed data in production)


@tool
def get_portfolio_metrics(ticker: str, benchmark: str = "SPY") -> dict[str, Any]:
    """
    Compute key portfolio risk metrics for a stock: Sharpe ratio,
    beta against a benchmark, alpha, max drawdown, and correlation.

    Args:
        ticker: Stock ticker to analyse, e.g. 'AAPL'
        benchmark: Benchmark ticker to compare against (default 'SPY')

    Returns:
        Dict with Sharpe ratio, beta, alpha, max drawdown, and correlation.
    """
    try:
        logger.info("[PortfolioTool] Computing metrics for ticker=%s vs benchmark=%s", ticker, benchmark)

        tickers = [ticker.upper(), benchmark.upper()]
        raw = yf.download(tickers, period="1y", auto_adjust=True, progress=False)

        if raw.empty:
            return {"error": "No price data returned", "ticker": ticker}

        # Handle both single and multi-ticker DataFrames
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        else:
            prices = raw[["Close"]].rename(columns={"Close": ticker.upper()})

        if ticker.upper() not in prices.columns or benchmark.upper() not in prices.columns:
            return {"error": f"Missing data for {ticker} or {benchmark}", "ticker": ticker}

        prices = prices.dropna()
        returns = prices.pct_change().dropna()

        stock_ret = returns[ticker.upper()]
        bench_ret = returns[benchmark.upper()]

        # ── Sharpe Ratio ──────────────────────────────────────────────────────
        daily_rf = RISK_FREE_RATE_ANNUAL / 252
        excess = stock_ret - daily_rf
        sharpe = float((excess.mean() / excess.std()) * np.sqrt(252)) if excess.std() > 0 else None

        # ── Beta & Alpha ──────────────────────────────────────────────────────
        cov_matrix = np.cov(stock_ret, bench_ret)
        beta = float(cov_matrix[0, 1] / cov_matrix[1, 1]) if cov_matrix[1, 1] > 0 else None
        annualised_stock = float(stock_ret.mean() * 252)
        annualised_bench = float(bench_ret.mean() * 252)
        alpha = float(annualised_stock - (RISK_FREE_RATE_ANNUAL + (beta or 0) * (annualised_bench - RISK_FREE_RATE_ANNUAL)))

        # ── Max Drawdown ──────────────────────────────────────────────────────
        cumulative = (1 + stock_ret).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = float(drawdown.min() * 100)

        # ── Correlation ───────────────────────────────────────────────────────
        correlation = float(stock_ret.corr(bench_ret))

        # ── Annualised Return & Volatility ────────────────────────────────────
        annualised_return = round(annualised_stock * 100, 2)
        annualised_vol = round(float(stock_ret.std() * np.sqrt(252) * 100), 2)

        return {
            "ticker": ticker.upper(),
            "benchmark": benchmark.upper(),
            "period": "1y",
            "sharpe_ratio": round(sharpe, 3) if sharpe else None,
            "beta": round(beta, 3) if beta else None,
            "alpha_pct": round(alpha * 100, 2),
            "max_drawdown_pct": round(max_drawdown, 2),
            "correlation_with_benchmark": round(correlation, 3),
            "annualised_return_pct": annualised_return,
            "annualised_volatility_pct": annualised_vol,
            "risk_free_rate_used": RISK_FREE_RATE_ANNUAL,
        }

    except Exception as e:
        logger.error("[PortfolioTool] Error computing metrics for %s: %s", ticker, e)
        return {"error": str(e), "ticker": ticker}


@tool
def get_peer_comparison(ticker: str) -> dict[str, Any]:
    """
    Compare a stock's key valuation multiples against its sector peers.
    Fetches the stock's sector, then compares P/E, P/B, and margins.

    Args:
        ticker: Stock ticker to compare, e.g. 'AAPL'

    Returns:
        Dict with the stock's multiples and sector context for comparison.
    """
    try:
        logger.info("[PortfolioTool] Fetching peer comparison for ticker=%s", ticker)

        stock = yf.Ticker(ticker.upper())
        info = stock.info

        sector = info.get("sector", "Unknown")
        industry = info.get("industry", "Unknown")

        return {
            "ticker": ticker.upper(),
            "company_name": info.get("longName"),
            "sector": sector,
            "industry": industry,
            "valuation_multiples": {
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "price_to_book": info.get("priceToBook"),
                "price_to_sales": info.get("priceToSalesTrailing12Months"),
                "ev_to_ebitda": info.get("enterpriseToEbitda"),
            },
            "profitability": {
                "gross_margin": info.get("grossMargins"),
                "operating_margin": info.get("operatingMargins"),
                "net_margin": info.get("profitMargins"),
                "roe": info.get("returnOnEquity"),
                "roa": info.get("returnOnAssets"),
            },
            "growth": {
                "revenue_growth_yoy": info.get("revenueGrowth"),
                "earnings_growth_yoy": info.get("earningsGrowth"),
            },
            "note": (
                f"Compare these multiples to sector peers in {sector} / {industry}. "
                "A low P/E vs sector average may indicate undervaluation or headwinds."
            ),
        }

    except Exception as e:
        logger.error("[PortfolioTool] Error in peer comparison for %s: %s", ticker, e)
        return {"error": str(e), "ticker": ticker}
