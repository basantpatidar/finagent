import json
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from httpx import AsyncClient, ASGITransport


@pytest.fixture
def app():
    from finagent.api.app import create_app
    return create_app()


@pytest.fixture
async def client(app):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


class TestHealthEndpoint:

    async def test_health_returns_ok(self, client):
        response = await client.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert "model" in body


class TestResearchEndpoint:

    async def test_valid_ticker_returns_research_brief(self, client):
        mock_state = {
            "research_brief": "## EXECUTIVE SUMMARY\nApple Inc is a technology company...",
            "errors": [],
            "messages": [],
        }

        with patch("finagent.api.routers.research_router.run_research", return_value=mock_state):
            response = await client.post(
                "/api/research/",
                json={"ticker": "AAPL"},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["ticker"] == "AAPL"
        assert "EXECUTIVE SUMMARY" in body["research_brief"]
        assert body["errors"] == []

    async def test_ticker_is_uppercased(self, client):
        mock_state = {
            "research_brief": "Brief for MSFT",
            "errors": [],
            "messages": [],
        }

        with patch("finagent.api.routers.research_router.run_research", return_value=mock_state) as mock_run:
            response = await client.post(
                "/api/research/",
                json={"ticker": "msft"},
            )

        assert response.status_code == 200
        # Ticker should be normalised to uppercase
        mock_run.assert_called_once_with(ticker="MSFT", question=None)

    async def test_invalid_ticker_returns_422(self, client):
        response = await client.post(
            "/api/research/",
            json={"ticker": "INVALID123!!"},
        )

        assert response.status_code == 422

    async def test_empty_ticker_returns_422(self, client):
        response = await client.post(
            "/api/research/",
            json={"ticker": ""},
        )

        assert response.status_code == 422

    async def test_custom_question_passed_to_runner(self, client):
        mock_state = {
            "research_brief": "Focused analysis...",
            "errors": [],
            "messages": [],
        }

        with patch("finagent.api.routers.research_router.run_research", return_value=mock_state) as mock_run:
            response = await client.post(
                "/api/research/",
                json={"ticker": "AAPL", "question": "What is the dividend yield?"},
            )

        assert response.status_code == 200
        mock_run.assert_called_once_with(
            ticker="AAPL", question="What is the dividend yield?"
        )

    async def test_agent_error_returns_500(self, client):
        with patch(
            "finagent.api.routers.research_router.run_research",
            side_effect=RuntimeError("LLM API error"),
        ):
            response = await client.post(
                "/api/research/",
                json={"ticker": "AAPL"},
            )

        assert response.status_code == 500

    async def test_errors_in_state_included_in_response(self, client):
        mock_state = {
            "research_brief": "Partial brief despite tool errors...",
            "errors": ["SECTool failed: timeout", "NewsTool failed: rate limit"],
            "messages": [],
        }

        with patch("finagent.api.routers.research_router.run_research", return_value=mock_state):
            response = await client.post(
                "/api/research/",
                json={"ticker": "AAPL"},
            )

        assert response.status_code == 200
        body = response.json()
        assert len(body["errors"]) == 2


class TestStreamEndpoint:

    async def test_stream_returns_sse_events(self, client):

        async def mock_stream(ticker, question=None):
            yield {"type": "tool_start", "data": {"tool": "get_stock_price", "args": {"ticker": "AAPL"}}}
            yield {"type": "tool_end",   "data": {"tool_call_id": "call_001", "content_preview": "price data..."}}
            yield {"type": "complete",   "data": {"research_brief": "## Research for AAPL", "ticker": "AAPL"}}

        with patch("finagent.api.routers.research_router.stream_research", side_effect=mock_stream):
            response = await client.get("/api/research/stream/AAPL")

        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

    async def test_stream_ticker_uppercased(self, client):

        async def mock_stream(ticker, question=None):
            yield {"type": "complete", "data": {"research_brief": "Brief", "ticker": ticker}}

        with patch("finagent.api.routers.research_router.stream_research", side_effect=mock_stream) as mock_fn:
            await client.get("/api/research/stream/aapl")

        mock_fn.assert_called_once_with(ticker="AAPL", question=None)
