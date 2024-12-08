import pytest
from unittest.mock import patch, MagicMock
import json


class TestGetSecFilings:

    @patch("finagent.tools.sec.sec_tools.requests.get")
    def test_returns_filings_for_known_ticker(self, mock_get):
        from finagent.tools.sec.sec_tools import get_sec_filings

        # Mock company_tickers.json response
        tickers_response = MagicMock()
        tickers_response.json.return_value = {
            "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}
        }
        tickers_response.raise_for_status = MagicMock()

        # Mock submissions response
        submissions_response = MagicMock()
        submissions_response.json.return_value = {
            "name": "Apple Inc.",
            "filings": {
                "recent": {
                    "form": ["10-K", "10-Q", "8-K"],
                    "filingDate": ["2024-11-01", "2024-08-01", "2024-07-01"],
                    "accessionNumber": ["0000320193-24-000123", "0000320193-24-000456", "0000320193-24-000789"],
                    "primaryDocument": ["aapl-20240928.htm", "aapl-20240629.htm", "aapl-20240601.htm"],
                }
            },
        }
        submissions_response.raise_for_status = MagicMock()

        mock_get.side_effect = [tickers_response, submissions_response]

        result = get_sec_filings.invoke({"ticker": "AAPL", "max_filings": 2})

        assert result["ticker"] == "AAPL"
        assert result["company_name"] == "Apple Inc."
        assert len(result["filings"]) == 2
        assert result["filings"][0]["form_type"] == "10-K"
        assert result["filings"][1]["form_type"] == "10-Q"

    @patch("finagent.tools.sec.sec_tools.requests.get")
    def test_returns_error_when_cik_not_found(self, mock_get):
        from finagent.tools.sec.sec_tools import get_sec_filings

        tickers_response = MagicMock()
        tickers_response.json.return_value = {}
        tickers_response.raise_for_status = MagicMock()

        mock_get.return_value = tickers_response

        result = get_sec_filings.invoke({"ticker": "FAKEXYZ"})

        assert "error" in result
        assert result["filings"] == []

    @patch("finagent.tools.sec.sec_tools.requests.get")
    def test_handles_network_error_gracefully(self, mock_get):
        from finagent.tools.sec.sec_tools import get_sec_filings

        mock_get.side_effect = Exception("Connection refused")

        result = get_sec_filings.invoke({"ticker": "AAPL"})

        assert "error" in result


class TestGetSecFacts:

    @patch("finagent.tools.sec.sec_tools.requests.get")
    def test_extracts_latest_annual_revenue(self, mock_get):
        from finagent.tools.sec.sec_tools import get_sec_facts

        tickers_response = MagicMock()
        tickers_response.json.return_value = {
            "0": {"cik_str": 320193, "ticker": "AAPL"}
        }
        tickers_response.raise_for_status = MagicMock()

        facts_response = MagicMock()
        facts_response.json.return_value = {
            "facts": {
                "us-gaap": {
                    "Revenues": {
                        "units": {
                            "USD": [
                                {"val": 383_000_000_000, "end": "2023-09-30", "form": "10-K", "filed": "2023-11-03"},
                                {"val": 394_000_000_000, "end": "2024-09-28", "form": "10-K", "filed": "2024-11-01"},
                                {"val": 94_000_000_000,  "end": "2024-06-29", "form": "10-Q", "filed": "2024-08-02"},
                            ]
                        }
                    },
                    "NetIncomeLoss": {
                        "units": {
                            "USD": [
                                {"val": 97_000_000_000, "end": "2024-09-28", "form": "10-K", "filed": "2024-11-01"},
                            ]
                        }
                    },
                }
            }
        }
        facts_response.raise_for_status = MagicMock()

        mock_get.side_effect = [tickers_response, facts_response]

        result = get_sec_facts.invoke({"ticker": "AAPL"})

        assert result["ticker"] == "AAPL"
        assert result["revenue"]["value"] == 394_000_000_000
        assert result["revenue"]["period_end"] == "2024-09-28"
        assert result["net_income"]["value"] == 97_000_000_000
