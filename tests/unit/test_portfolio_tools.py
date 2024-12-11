import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np


class TestGetPortfolioMetrics:

    @patch("finagent.tools.portfolio.portfolio_tools.yf.download")
    def test_computes_sharpe_beta_drawdown(self, mock_download):
        from finagent.tools.portfolio.portfolio_tools import get_portfolio_metrics

        # Build synthetic 1-year daily price data
        dates = pd.date_range("2023-01-01", periods=252, freq="B")
        np.random.seed(42)

        aapl_prices = 150 * np.cumprod(1 + np.random.normal(0.001, 0.015, 252))
        spy_prices  = 400 * np.cumprod(1 + np.random.normal(0.0005, 0.01, 252))

        prices_df = pd.DataFrame(
            {"AAPL": aapl_prices, "SPY": spy_prices},
            index=dates,
        )
        mock_download.return_value = pd.concat(
            {"Close": prices_df}, axis=1
        )

        result = get_portfolio_metrics.invoke({"ticker": "AAPL", "benchmark": "SPY"})

        assert result["ticker"] == "AAPL"
        assert result["benchmark"] == "SPY"
        assert "sharpe_ratio" in result
        assert "beta" in result
        assert "max_drawdown_pct" in result
        assert "correlation_with_benchmark" in result
        assert result["max_drawdown_pct"] <= 0  # drawdown is always non-positive

    @patch("finagent.tools.portfolio.portfolio_tools.yf.download")
    def test_returns_error_for_empty_data(self, mock_download):
        from finagent.tools.portfolio.portfolio_tools import get_portfolio_metrics

        mock_download.return_value = pd.DataFrame()

        result = get_portfolio_metrics.invoke({"ticker": "FAKE", "benchmark": "SPY"})

        assert "error" in result

    @patch("finagent.tools.portfolio.portfolio_tools.yf.download")
    def test_handles_exception_gracefully(self, mock_download):
        from finagent.tools.portfolio.portfolio_tools import get_portfolio_metrics

        mock_download.side_effect = Exception("yfinance rate limit")

        result = get_portfolio_metrics.invoke({"ticker": "AAPL", "benchmark": "SPY"})

        assert "error" in result
        assert result["ticker"] == "AAPL"


class TestGetPeerComparison:

    @patch("finagent.tools.portfolio.portfolio_tools.yf.Ticker")
    def test_returns_valuation_and_profitability(self, mock_ticker_cls):
        from finagent.tools.portfolio.portfolio_tools import get_peer_comparison

        mock_ticker = MagicMock()
        mock_ticker_cls.return_value = mock_ticker
        mock_ticker.info = {
            "longName": "Apple Inc.",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "trailingPE": 28.5,
            "forwardPE": 25.0,
            "priceToBook": 45.0,
            "grossMargins": 0.46,
            "operatingMargins": 0.30,
            "profitMargins": 0.26,
            "returnOnEquity": 1.47,
            "revenueGrowth": 0.06,
            "earningsGrowth": 0.11,
        }

        result = get_peer_comparison.invoke({"ticker": "AAPL"})

        assert result["ticker"] == "AAPL"
        assert result["sector"] == "Technology"
        assert result["valuation_multiples"]["pe_ratio"] == 28.5
        assert result["profitability"]["gross_margin"] == 0.46
        assert "note" in result
