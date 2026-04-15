# FinAgent — Agentic Financial Research Assistant

An LLM-powered autonomous research agent that takes a stock ticker, calls 9 real financial data tools, reasons across market data, SEC filings, and news — then streams a structured investment research brief in real time.

Built with **LangGraph + Claude + FastAPI**.

---

## Architecture

```
Client
  │
  ▼
FastAPI (port 8000)
  │
  ├── POST /api/research/          Sync: waits for full brief (~30-60s)
  └── GET  /api/research/stream/{ticker}  SSE: real-time tool progress
  │
  ▼
LangGraph ReAct Agent
  │
  ┌─────────────────────────────────────────────────┐
  │                                                 │
  │   START → llm_node ──── has tool_calls? ──YES──► tool_node ─┐
  │                │                                             │
  │               NO                                            │
  │                │                                            │
  │          synthesis_node ◄───────────────────────────────────┘
  │                │
  │               END
  │                                                 │
  └─────────────────────────────────────────────────┘
  │
  ▼
9 Financial Tools
  ├── Market:    get_stock_price, get_price_history, get_company_fundamentals
  ├── News:      get_recent_news, get_earnings_calendar
  ├── SEC:       get_sec_filings (EDGAR), get_sec_facts (XBRL)
  └── Portfolio: get_portfolio_metrics (Sharpe/beta), get_peer_comparison
```

## SSE Event Stream

When using the streaming endpoint, events are emitted as each step completes:

```
event: tool_start
data: {"tool": "get_stock_price", "args": {"ticker": "AAPL"}}

event: tool_end
data: {"tool_call_id": "call_001", "content_preview": "{\"current_price\": 195.0 ..."}

event: tool_start
data: {"tool": "get_sec_filings", "args": {"ticker": "AAPL"}}

...9 tools total...

event: complete
data: {"research_brief": "## EXECUTIVE SUMMARY\n...", "ticker": "AAPL"}
```

## Research Brief Structure

Every brief follows this format:

```
EXECUTIVE SUMMARY
PRICE & TECHNICALS
FUNDAMENTALS
RECENT DEVELOPMENTS
SEC FILINGS ANALYSIS
RISK METRICS
INVESTMENT CONSIDERATIONS
DISCLAIMER
```

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/basant/finagent
cd finagent

cp .env.example .env
# Edit .env and add: ANTHROPIC_API_KEY=sk-ant-...

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Run the API
python main.py
# → API docs: http://localhost:8000/docs

# 3a. Synchronous research (waits for full brief)
curl -X POST http://localhost:8000/api/research/ \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL"}' | python -m json.tool

# 3b. Streaming research (real-time progress)
curl -N http://localhost:8000/api/research/stream/AAPL

# 3c. Custom research question
curl -X POST http://localhost:8000/api/research/ \
  -H "Content-Type: application/json" \
  -d '{"ticker": "NVDA", "question": "Analyse NVDA data centre growth and AI chip demand"}'
```

## Docker

```bash
docker-compose up --build
# API available at http://localhost:8000
```

## Running Tests

```bash
pytest tests/ -v

# Unit tests only (fast, no network)
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# With coverage
pytest tests/ --cov=finagent --cov-report=term-missing
```

## API Reference

### `POST /api/research/`

Run a full research pipeline synchronously.

**Request:**
```json
{
  "ticker": "AAPL",
  "question": null
}
```

**Response:**
```json
{
  "ticker": "AAPL",
  "research_brief": "## EXECUTIVE SUMMARY\n...",
  "errors": [],
  "tool_calls_made": 9
}
```

### `GET /api/research/stream/{ticker}`

Stream research progress as Server-Sent Events.

```bash
curl -N "http://localhost:8000/api/research/stream/AAPL?question=Focus+on+AI+revenue"
```

### `GET /health`

```json
{
  "status": "ok",
  "model": "claude-3-sonnet-20240229",
  "version": "1.0.0"
}
```

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Claude (claude-3-sonnet-20240229) via Anthropic API |
| Agent Framework | LangGraph 0.2 + LangChain |
| API | FastAPI + Uvicorn |
| Streaming | SSE Starlette |
| Market Data | yfinance |
| SEC Data | EDGAR REST API + XBRL |
| Portfolio Math | NumPy + pandas |
| Config | pydantic-settings + python-dotenv |
| Testing | pytest + pytest-asyncio + httpx |

## Design Decisions

See `ADR.md` for detailed reasoning. Key choices:

- **LangGraph over raw loops** — explicit graph boundaries make state inspection and
  debugging far easier; essential for production agentic systems
- **Tools never raise** — return `{"error": "..."}` dicts so the agent can continue
  with partial data if one source (e.g. SEC EDGAR) times out
- **SSE over WebSockets** — one-directional streaming from server to client is all
  we need; SSE is simpler, HTTP-native, and works through proxies
- **Structured state fields** — `price_data`, `sec_filings` etc. in `AgentState`
  alongside raw `messages`, so the API can return typed structured data
