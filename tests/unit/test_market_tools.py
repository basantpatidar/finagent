import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from datetime import datetime


class TestGetStockPrice:

    @patch("finagent.tools.market.market_tools.yf.Ticker")
    def test_returns_price_data_for_valid_ticker(self, mock_ticker_cls):
        from finagent.tools.market.market_tools import get_stock_price

        mock_ticker = MagicMock()
        mock_ticker_cls.return_value = mock_ticker
        mock_ticker.info = {
            "volume": 55_000_000,
            "averageVolume": 60_000_000,
            "marketCap": 3_000_000_000_000,
            "fiftyTwoWeekHigh": 230.0,
            "fiftyTwoWeekLow": 165.0,
            "currency": "USD",
        }
        mock_ticker.history.return_value = pd.DataFrame(
            {"Close": [190.0, 195.0]},
            index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
        )

        result = get_stock_price.invoke({"ticker": "AAPL"})

        assert result["ticker"] == "AAPL"
        assert result["current_price"] == 195.0
        assert result["previous_close"] == 190.0
        assert result["change"] == pytest.approx(5.0)
        assert result["change_percent"] == pytest.approx(2.63, rel=0.01)
        assert "as_of" in result

    @patch("finagent.tools.market.market_tools.yf.Ticker")
    def test_returns_error_for_empty_history(self, mock_ticker_cls):
        from finagent.tools.market.market_tools import get_stock_price

        mock_ticker = MagicMock()
        mock_ticker_cls.return_value = mock_ticker
        mock_ticker.info = {}
        mock_ticker.history.return_value = pd.DataFrame()

        result = get_stock_price.invoke({"ticker": "INVALID"})

        assert "error" in result

    @patch("finagent.tools.market.market_tools.yf.Ticker")
    def test_ticker_uppercased(self, mock_ticker_cls):
        from finagent.tools.market.market_tools import get_stock_price

        mock_ticker = MagicMock()
        mock_ticker_cls.return_value = mock_ticker
        mock_ticker.info = {"currency": "USD"}
        mock_ticker.history.return_value = pd.DataFrame(
            {"Close": [100.0, 101.0]},
            index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
        )

        result = get_stock_price.invoke({"ticker": "aapl"})
        assert result["ticker"] == "AAPL"

    @patch("finagent.tools.market.market_tools.yf.Ticker")
    def test_handles_yfinance_exception_gracefully(self, mock_ticker_cls):
        from finagent.tools.market.market_tools import get_stock_price

        mock_ticker_cls.side_effect = Exception("network timeout")

        result = get_stock_price.invoke({"ticker": "AAPL"})
        assert "error" in result


class TestGetCompanyFundamentals:

    @patch("finagent.tools.market.market_tools.yf.Ticker")
    def test_returns_fundamentals(self, mock_ticker_cls):
        from finagent.tools.market.market_tools import get_company_fundamentals

        mock_ticker = MagicMock()
        mock_ticker_cls.return_value = mock_ticker
        mock_ticker.info = {
            "longName": "Apple Inc.",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "trailingPE": 28.5,
            "forwardPE": 25.0,
            "profitMargins": 0.26,
            "debtToEquity": 150.0,
            "returnOnEquity": 1.47,
            "recommendationKey": "buy",
            "numberOfAnalystOpinions": 42,
            "targetMeanPrice": 220.0,
            "beta": 1.2,
        }

        result = get_company_fundamentals.invoke({"ticker": "AAPL"})

        assert result["company_name"] == "Apple Inc."
        assert result["sector"] == "Technology"
        assert result["pe_ratio"] == 28.5
        assert result["analyst_recommendation"] == "buy"
        assert result["beta"] == 1.2

    @patch("finagent.tools.market.market_tools.yf.Ticker")
    def test_handles_missing_fields_gracefully(self, mock_ticker_cls):
        from finagent.tools.market.market_tools import get_company_fundamentals

        mock_ticker = MagicMock()
        mock_ticker_cls.return_value = mock_ticker
        mock_ticker.info = {}  # empty info

        result = get_company_fundamentals.invoke({"ticker": "AAPL"})

        # Should return dict with None values, not raise
        assert result["ticker"] == "AAPL"
        assert result["pe_ratio"] is None
