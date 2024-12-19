# FinAgent — Complete Developer Guide

> **Goal of this document:** Give you everything you need to rebuild this project from scratch,
> understand every design decision, and extend it confidently. Written as if you are starting
> with a blank directory.

---

## Table of Contents

1. [What the System Does](#1-what-the-system-does)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Tech Stack and Why](#3-tech-stack-and-why)
4. [Project Layout Explained](#4-project-layout-explained)
5. [Step 0 — Environment Setup](#5-step-0--environment-setup)
6. [Step 1 — Configuration (config.py)](#6-step-1--configuration-configpy)
7. [Step 2 — The 9 Financial Data Tools](#7-step-2--the-9-financial-data-tools)
8. [Step 3 — Agent State (Shared Memory)](#8-step-3--agent-state-shared-memory)
9. [Step 4 — LangGraph Nodes](#9-step-4--langgraph-nodes)
10. [Step 5 — The Agent Graph (Routing Logic)](#10-step-5--the-agent-graph-routing-logic)
11. [Step 6 — The Runner (Entry Points)](#11-step-6--the-runner-entry-points)
12. [Step 7 — FastAPI Layer](#12-step-7--fastapi-layer)
13. [Step 8 — Server-Sent Events (SSE Streaming)](#13-step-8--server-sent-events-sse-streaming)
14. [Step 9 — Docker and Containerisation](#14-step-9--docker-and-containerisation)
15. [Step 10 — Testing Strategy](#15-step-10--testing-strategy)
16. [What Is NOT Built Yet (Roadmap)](#16-what-is-not-built-yet-roadmap)
17. [How a Full Request Flows End-to-End](#17-how-a-full-request-flows-end-to-end)
18. [Planned Infrastructure: Postgres, Kafka, Redis](#18-planned-infrastructure-postgres-kafka-redis)
19. [Error Handling Philosophy](#19-error-handling-philosophy)
20. [Key Design Decisions (ADRs)](#20-key-design-decisions-adrs)

---

## 1. What the System Does

FinAgent is a **LLM-powered financial research assistant**. Given a stock ticker symbol (e.g. `AAPL`),
it autonomously:

1. Calls **9 financial data tools** in parallel-ish reasoning loops
2. Gathers: current price, 6-month history, fundamentals, news, earnings, SEC filings,
   XBRL financial facts, portfolio risk metrics, and peer comparison
3. Synthesises everything into a **structured investment research brief** — like what a junior
   equity analyst would produce

The output is streamed in real-time so the UI shows live tool progress, not a spinner.

**This is not a chatbot.** It is an **agentic system**: an LLM decides which tools to call,
in what order, and when it has enough data to produce the final report. The developer does not
hardcode this logic.

---

## 2. System Architecture Overview

```
┌───────────────────────────────────────────────────────────────────────┐
│  Client (browser / curl / FinDash React app)                          │
│    POST /api/research/           → waits, returns JSON                │
│    GET  /api/research/stream/AAPL → SSE stream of progress events     │
└───────────────────────┬───────────────────────────────────────────────┘
                        │ HTTP
                        ▼
┌───────────────────────────────────────────────────────────────────────┐
│  FastAPI (port 8000)                                                   │
│  api/app.py          → CORS, lifespan, health check                   │
│  api/routers/        → POST /api/research, GET /stream/{ticker}       │
│  api/models/schemas  → Pydantic request/response validation           │
└───────────────────────┬───────────────────────────────────────────────┘
                        │ calls runner.py
                        ▼
┌───────────────────────────────────────────────────────────────────────┐
│  agent/runner.py                                                       │
│  run_research()   → sync:  finagent_graph.invoke(initial_state)       │
│  stream_research() → async: finagent_graph.astream(initial_state)     │
└───────────────────────┬───────────────────────────────────────────────┘
                        │ runs the graph
                        ▼
┌───────────────────────────────────────────────────────────────────────┐
│  LangGraph StateGraph (agent/graph/agent_graph.py)                     │
│                                                                        │
│   START ──► llm_node ──► should_continue() ──► tool_node ──┐         │
│                │                  │                          │         │
│                │                  └──► synthesis_node ──► END│         │
│                │                  └──► END (max iterations)  │         │
│                └──────────────────────────────────────────────┘         │
│                              (loop)                                    │
└───────────────────────────────────────────────────────────────────────┘
         │                          │
         ▼                          ▼
  Claude Sonnet (LLM)         9 Financial Tools
  Decides what to call        market / news / SEC / portfolio
```

There are **three node types** in the graph:

| Node | What it does |
|---|---|
| `llm_node` | Calls Claude. Claude sees full message history + bound tools. Returns either tool call requests or final text. |
| `tool_node` | Executes whichever tools Claude requested. Stores results in typed state fields. Never raises — returns error dicts. |
| `synthesis_node` | Extracts the final text brief from the last AIMessage and stores it in `state["research_brief"]`. |

---

## 3. Tech Stack and Why

| Library | Version | Why This Choice |
|---|---|---|
| **Python** | 3.12 | `typing` improvements, `TypedDict` ergonomics, `match` statements |
| **LangGraph** | 0.2.28 | Explicit graph topology vs raw while loops. Each node is a testable pure function. Supports `astream` for SSE. |
| **LangChain Anthropic** | 0.2.4 | `ChatAnthropic.bind_tools()` converts Python functions into Claude tool schemas automatically |
| **Anthropic SDK** | 0.37.1 | Underlying Claude API client |
| **FastAPI** | 0.115.4 | Async, Pydantic v2 native, OpenAPI docs auto-generated |
| **uvicorn** | 0.32.0 | ASGI server. `[standard]` installs uvloop + httptools for performance |
| **sse-starlette** | 2.1.3 | `EventSourceResponse` — wraps an async generator into an HTTP SSE stream |
| **yfinance** | 0.2.44 | Free Yahoo Finance API. Used for price, history, fundamentals, news, earnings |
| **requests** | 2.32.3 | Sync HTTP for SEC EDGAR API calls (EDGAR does not support async well) |
| **pandas** | 2.2.3 | Time-series manipulation for price history, rolling averages |
| **numpy** | 2.1.3 | Sharpe ratio, beta, covariance matrix, max drawdown math |
| **pydantic** | 2.9.2 | Request/response validation. `field_validator` for ticker sanitisation |
| **pydantic-settings** | 2.6.1 | `BaseSettings` reads `.env` file and env vars into a typed config object |
| **pytest** | 8.3.3 | Test runner |
| **pytest-asyncio** | 0.24.0 | Makes `async def` test functions work natively |
| **pytest-mock** | 3.14.0 | `mocker` fixture — ergonomic mock.patch without decorators |
| **httpx** | 0.27.2 | `AsyncClient(transport=ASGITransport(app))` for integration tests without a live server |

---

## 4. Project Layout Explained

```
finagent/                       ← Python package root (__init__.py present)
├── main.py                     ← Uvicorn entrypoint (run this to start the server)
├── config.py                   ← All settings from .env via pydantic-settings
├── requirements.txt            ← Pinned dependencies
├── Dockerfile                  ← Multi-stage build (builder + runtime)
├── docker-compose.yml          ← Single-service compose for the API container
├── pytest.ini                  ← asyncio_mode=auto, testpaths=tests
├── .env.example                ← Template — copy to .env and fill in keys
│
├── agent/                      ← Everything the LLM agent needs
│   ├── state/
│   │   └── agent_state.py      ← AgentState TypedDict (shared memory between nodes)
│   ├── nodes/
│   │   ├── llm_node.py         ← Invokes Claude, prepends system prompt once
│   │   ├── tool_node.py        ← Executes tools, stores results in state
│   │   └── synthesis_node.py   ← Extracts final brief from last AIMessage
│   ├── graph/
│   │   └── agent_graph.py      ← StateGraph topology + should_continue router
│   └── runner.py               ← run_research() (sync) and stream_research() (async)
│
├── tools/                      ← 9 financial data tools (LangChain @tool decorated)
│   ├── __init__.py             ← ALL_TOOLS list + TOOL_MAP dict
│   ├── market/market_tools.py  ← yfinance: price, history, fundamentals
│   ├── news/news_tools.py      ← yfinance: articles, earnings calendar
│   ├── sec/sec_tools.py        ← SEC EDGAR API: filings, XBRL facts
│   └── portfolio/portfolio_tools.py ← NumPy: Sharpe, beta, drawdown, peers
│
├── api/                        ← FastAPI web layer
│   ├── app.py                  ← create_app() factory, CORS, lifespan, /health
│   ├── models/schemas.py       ← ResearchRequest, ResearchResponse, StreamEvent
│   └── routers/research_router.py ← POST /api/research + GET /stream/{ticker}
│
└── tests/
    ├── unit/                   ← Mocked tests — never hit real APIs
    │   ├── test_market_tools.py
    │   ├── test_sec_tools.py
    │   ├── test_portfolio_tools.py
    │   ├── test_agent_nodes.py
    │   └── test_schemas.py
    └── integration/            ← httpx AsyncClient tests against the real FastAPI app
        └── test_api.py
```

---

## 5. Step 0 — Environment Setup

### Prerequisites

- Python 3.12 (`python --version` to check)
- An Anthropic API key (get one at console.anthropic.com)
- Docker Desktop (optional, for container mode)

### Local Setup

```bash
# Clone / create the project directory
cd finagent/

# Create virtual environment
python -m venv .venv

# Activate it
source .venv/bin/activate          # macOS/Linux
.venv\Scripts\activate             # Windows

# Install all dependencies
pip install -r requirements.txt

# Set up your environment file
cp .env.example .env
# Open .env and set: ANTHROPIC_API_KEY=sk-ant-...

# Run the server
python main.py
# Server starts at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

### .env Variables

```env
# Required
ANTHROPIC_API_KEY=sk-ant-...        # Your Anthropic key

# Agent tuning
CLAUDE_MODEL=claude-3-sonnet-20240229      # Which Claude model to use
AGENT_MAX_ITERATIONS=10             # Hard stop: max LLM calls per research run
AGENT_TEMPERATURE=0.1               # Low temp = more deterministic tool-calling

# API server
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=false                     # True enables hot reload in uvicorn

# Tool limits
YFINANCE_TIMEOUT=10                 # Seconds before yfinance call fails
NEWS_MAX_ARTICLES=5                 # Articles per news fetch
SEC_MAX_FILINGS=3                   # SEC filings to retrieve
```

**Rule:** Never use `os.getenv()` directly in source files. All config comes from `settings.*`.

---

## 6. Step 1 — Configuration (config.py)

`config.py` is the single source of truth for all tunables:

```python
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    anthropic_api_key: str = Field(..., env="ANTHROPIC_API_KEY")  # ... = required
    claude_model: str = Field("claude-3-sonnet-20240229", env="CLAUDE_MODEL")
    agent_max_iterations: int = Field(10, env="AGENT_MAX_ITERATIONS")
    agent_temperature: float = Field(0.1, env="AGENT_TEMPERATURE")
    # ... all other settings

    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()  # Module-level singleton — import this everywhere
```

**How it works:**
- `pydantic-settings` reads from environment variables first, then `.env` file
- `Field(..., env="X")` means required — startup fails fast if missing
- `Field("default", env="X")` means optional with a default
- Access anywhere: `from finagent.config import settings` then `settings.claude_model`

---

## 7. Step 2 — The 9 Financial Data Tools

Tools are the "hands" of the agent. Each tool is a Python function decorated with `@tool`
from LangChain. This decoration does two things:

1. Generates a **JSON schema** from the function's type hints and docstring
2. Makes it invocable by the LLM via Claude's tool_use feature

### Tool Contract (Critical Rules)

- **Never raise exceptions.** Catch everything and return `{"error": "...", "ticker": "..."}`.
- **Always log entry** at INFO level: `logger.info("[ToolName] action for ticker=%s", ticker)`
- **Return dicts**, never raw strings or primitives

### Market Tools (`tools/market/market_tools.py`)

**`get_stock_price(ticker: str)`**
- Uses `yf.Ticker(ticker).history(period="2d")` for current + previous close
- Computes `change = current - prev_close` and `change_percent`
- Returns: current_price, change, change_percent, volume, market_cap, 52-week high/low

**`get_price_history(ticker: str, period: str = "6mo")`**
- Fetches OHLCV data for the period
- Computes: MA-20, MA-50, MA-200 using `close.rolling(N).mean().iloc[-1]`
- Computes annualised volatility: `daily_returns.std() * sqrt(252) * 100`
- Returns: start/end price, period return %, moving averages, volatility

**`get_company_fundamentals(ticker: str)`**
- Reads `yf.Ticker(ticker).info` — a large dict of ~100 fields from Yahoo Finance
- Extracts: P/E, forward P/E, P/B, P/S, EV/EBITDA, margins, debt/equity, ROE, ROA,
  dividend yield, analyst target price, recommendation, beta

### News Tools (`tools/news/news_tools.py`)

**`get_recent_news(ticker: str, max_articles: int = 5)`**
- Uses `yf.Ticker(ticker).news` which returns a list of article dicts
- Normalises the nested `content` structure into flat article objects
- Returns: title, publisher, summary (≤300 chars), URL, published_at

**`get_earnings_calendar(ticker: str)`**
- Uses `yf.Ticker(ticker).earnings_history` — a DataFrame of past earnings
- Extracts: EPS actual vs estimate, surprise %, next earnings date from `info`

### SEC Tools (`tools/sec/sec_tools.py`)

The SEC EDGAR API is **free and public** — no API key needed. You must send a
`User-Agent` header identifying your application (EDGAR requirement).

**`_get_cik_for_ticker(ticker)` (internal helper)**
- Fetches `https://data.sec.gov/files/company_tickers.json` — a mapping of all
  tickers to their CIK (Central Index Key) numbers
- Searches for the ticker and zero-pads the CIK to 10 digits (`zfill(10)`)

**`get_sec_filings(ticker: str, max_filings: int = 3)`**
- Fetches `https://data.sec.gov/submissions/CIK{cik}.json` — the company's
  submission history
- Filters for `form_type in {"10-K", "10-Q"}` (annual and quarterly reports)
- Returns: form type, filing date, accession number, direct EDGAR document URL

**`get_sec_facts(ticker: str)`**
- Fetches `https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json`
- XBRL = eXtensible Business Reporting Language — structured financial data
- Parses `facts.us-gaap.{concept}.units.USD` to extract the most recent 10-K value
- Concepts: `Revenues`, `NetIncomeLoss`, `Assets`, `Liabilities`,
  `NetCashProvidedByUsedInOperatingActivities`, `ResearchAndDevelopmentExpense`

### Portfolio Tools (`tools/portfolio/portfolio_tools.py`)

**`get_portfolio_metrics(ticker: str, benchmark: str = "SPY")`**

Downloads 1 year of price history for both ticker and SPY using `yf.download()`.

Computes:
- **Sharpe Ratio** = `(mean(excess_return) / std(excess_return)) * sqrt(252)`
  where `excess_return = daily_return - (risk_free_rate / 252)`
- **Beta** = `cov(stock, benchmark) / var(benchmark)` — from the 2×2 covariance matrix
- **Alpha** = `annualised_stock_return - (risk_free + beta * (bench_return - risk_free))`
- **Max Drawdown** = peak-to-trough decline as a percentage using cumulative returns
- **Correlation** = Pearson correlation of daily returns vs benchmark

The risk-free rate is hardcoded at 5% (US 10-year Treasury approximation). In production
this would be fetched from the Fed API.

**`get_peer_comparison(ticker: str)`**
- Fetches sector and industry from `yf.Ticker(ticker).info`
- Returns the stock's valuation multiples (P/E, P/B, P/S, EV/EBITDA) and
  profitability metrics (margins, ROE, ROA) with context about the sector

### How Tools Are Registered

```python
# tools/__init__.py
ALL_TOOLS = [
    get_stock_price, get_price_history, get_company_fundamentals,
    get_recent_news, get_earnings_calendar,
    get_sec_filings, get_sec_facts,
    get_portfolio_metrics, get_peer_comparison,
]

TOOL_MAP = {tool.name: tool for tool in ALL_TOOLS}
# tool.name comes from the function name when @tool is applied
```

`ALL_TOOLS` is passed to `llm.bind_tools(ALL_TOOLS)` — LangChain converts each function's
signature and docstring into a tool schema that Claude understands. `TOOL_MAP` is used
in `tool_node` to look up and invoke tools by name.

---

## 8. Step 3 — Agent State (Shared Memory)

`agent/state/agent_state.py` defines the **shared state object** that flows through every
node in the graph. Think of it as the agent's working memory for a single research run.

```python
class AgentState(TypedDict):
    # Full message history (auto-appended by add_messages reducer)
    messages: Annotated[list[BaseMessage], add_messages]

    # The ticker being researched
    ticker: str

    # Structured outputs populated by tool_node as tools run
    price_data: dict[str, Any]         # from get_stock_price, get_price_history
    company_info: dict[str, Any]       # from get_company_fundamentals, get_peer_comparison
    financials: dict[str, Any]         # from get_sec_facts
    news_articles: list[dict[str, Any]] # from get_recent_news, get_earnings_calendar
    sec_filings: list[dict[str, Any]]  # from get_sec_filings
    portfolio_metrics: dict[str, Any]  # from get_portfolio_metrics

    # Final output
    research_brief: str                # populated by synthesis_node

    # Error tracking
    errors: list[str]                  # non-fatal errors from tools or nodes

    # Loop guard
    iteration_count: int               # incremented by llm_node each call
```

### The `add_messages` Reducer

This is a critical LangGraph concept. Without it, each node returning `{"messages": [new_msg]}`
would **overwrite** the entire history. With `Annotated[list[BaseMessage], add_messages]`,
LangGraph **appends** instead.

This means after a full run, `state["messages"]` contains the entire conversation thread:
```
HumanMessage("Research AAPL")
SystemMessage(system_prompt)
AIMessage(tool_calls=[get_stock_price, get_price_history, ...])
ToolMessage(tool_call_id=..., content="{current_price: 195}")
ToolMessage(tool_call_id=..., content="{period_return: 12.3}")
...more ToolMessages...
AIMessage("## EXECUTIVE SUMMARY\nApple Inc. is...")
```

Claude sees this full thread on every `llm_node` call, which is how it knows which
tools have already been called.

### Why Structured Fields Alongside Messages?

The `messages` list is for the LLM. The structured fields (`price_data`, `sec_filings`, etc.)
are for the **API response**. The `ResearchResponse` Pydantic model can return structured
data directly without parsing the research brief text. This lets the frontend render a
Recharts sparkline from `price_data` without scraping text.

---

## 9. Step 4 — LangGraph Nodes

Each node is a **pure function** that takes `AgentState` and returns a dict of state updates.
LangGraph merges the returned dict back into the state.

### llm_node (`agent/nodes/llm_node.py`)

```python
def llm_node(state: AgentState) -> dict:
    llm = ChatAnthropic(model=settings.claude_model, ...).bind_tools(ALL_TOOLS)

    # System prompt injected only on first iteration
    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

    response = llm.invoke(messages)

    return {
        "messages": [response],           # add_messages appends this
        "iteration_count": state["iteration_count"] + 1,
    }
```

The **system prompt** tells Claude:
- Who it is (FinAgent, expert analyst)
- In what order to call the 9 tools
- What sections to include in the final brief
- That it must never hallucinate numbers

The system prompt is injected **only once** by checking if a `SystemMessage` already exists
in the history. On subsequent iterations (after tool results come back), the check prevents
duplicate system messages.

### tool_node (`agent/nodes/tool_node.py`)

```python
def tool_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]  # The AIMessage with tool_calls
    tool_messages = []
    state_updates = { ... copy of current state fields ... }

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        try:
            result = TOOL_MAP[tool_name].invoke(tool_args)
            _store_result(tool_name, result, state_updates)  # typed state field
            tool_messages.append(ToolMessage(content=json.dumps(result), tool_call_id=tool_id))
        except Exception as e:
            # Never re-raise — store error, continue with next tool
            state_updates["errors"].append(f"Tool '{tool_name}' raised: {e}")
            tool_messages.append(ToolMessage(content=json.dumps({"error": str(e)}), tool_call_id=tool_id))

    return {"messages": tool_messages, **state_updates}
```

`_store_result()` routes each tool's output to the correct typed state field:
- `get_stock_price` → `price_data`
- `get_company_fundamentals` → `company_info`
- `get_sec_facts` → `financials`
- `get_recent_news` → `news_articles`
- etc.

### synthesis_node (`agent/nodes/synthesis_node.py`)

This runs once, after the LLM produces its final text (no more tool calls):

```python
def synthesis_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]

    if isinstance(last_message, AIMessage) and isinstance(last_message.content, str):
        brief = last_message.content
    elif isinstance(last_message, AIMessage) and isinstance(last_message.content, list):
        # Claude sometimes returns content as a list of blocks
        text_blocks = [b.get("text", "") for b in last_message.content if b.get("type") == "text"]
        brief = "\n".join(text_blocks)
    else:
        brief = "Research complete. See message history."

    return {"research_brief": brief}
```

---

## 10. Step 5 — The Agent Graph (Routing Logic)

`agent/graph/agent_graph.py` wires the three nodes together:

```python
graph = StateGraph(AgentState)

graph.add_node("llm", llm_node)
graph.add_node("tools", tool_node)
graph.add_node("synthesis", synthesis_node)

graph.set_entry_point("llm")  # always starts at LLM

graph.add_conditional_edges(
    "llm",
    should_continue,           # routing function
    {
        "tools": "tools",
        "synthesis": "synthesis",
        "end": END,
    }
)

graph.add_edge("tools", "llm")       # after tools, always return to LLM
graph.add_edge("synthesis", END)     # after synthesis, done

compiled = graph.compile()
```

### The Router (`should_continue`)

```python
def should_continue(state: AgentState) -> str:
    if state["iteration_count"] >= settings.agent_max_iterations:
        return "end"                    # hard stop — prevents runaway loops

    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"                  # Claude wants to call tools

    return "synthesis"                  # Claude produced final text
```

### Execution Flow for a Typical Run

```
Iteration 1:
  llm_node → Claude sees HumanMessage("Research AAPL") + system prompt
  Claude responds with AIMessage(tool_calls=[get_stock_price, get_company_fundamentals, ...])
  should_continue → "tools"
  tool_node → runs all 9 tools, adds 9 ToolMessages to state

Iteration 2:
  llm_node → Claude sees full history including all tool results
  Claude produces final brief: AIMessage(content="## EXECUTIVE SUMMARY...")
  should_continue → "synthesis"
  synthesis_node → extracts brief text, stores in state["research_brief"]
  → END
```

Claude typically calls all 9 tools in **one batch** (a single AIMessage with 9 tool_calls),
making this a 2-iteration loop. The `agent_max_iterations=10` guard exists for edge cases
where Claude decides to call tools in multiple rounds.

---

## 11. Step 6 — The Runner (Entry Points)

`agent/runner.py` provides two entry points used by the API layer.

### Sync Research

```python
def run_research(ticker: str, question: str | None = None) -> AgentState:
    initial = _initial_state(ticker, question)
    final_state = finagent_graph.invoke(initial)  # blocks until complete
    return final_state
```

`_initial_state()` builds the starting `AgentState`:
- `messages = [HumanMessage("Research AAPL")]`
- All other fields empty/zero

### Async Streaming Research

```python
async def stream_research(ticker, question=None) -> AsyncIterator[dict]:
    initial = _initial_state(ticker, question)

    async for event in finagent_graph.astream(initial, stream_mode="updates"):
        for node_name, node_output in event.items():

            if node_name == "tools":
                for msg in node_output["messages"]:
                    yield {"type": "tool_end", "data": {...}}

            elif node_name == "llm":
                for msg in node_output["messages"]:
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            yield {"type": "tool_start", "data": {"tool": tc["name"], ...}}
                    elif msg.content:
                        yield {"type": "text_chunk", "data": {"text": msg.content}}

            elif node_name == "synthesis":
                yield {"type": "complete", "data": {"research_brief": ..., "ticker": ...}}
```

`stream_mode="updates"` makes LangGraph emit one dict per node completion (not per token).
This gives the client **tool-level progress events**, which is more useful than character-by-
character streaming for this use case.

### SSE Event Types

| Event Type | When It Fires | Payload |
|---|---|---|
| `tool_start` | LLM has decided to call a tool | `{tool: "get_stock_price", args: {ticker: "AAPL"}}` |
| `tool_end` | A tool has completed | `{tool_call_id: "...", content_preview: "..."}` |
| `text_chunk` | LLM produced non-tool text | `{text: "..."}` |
| `complete` | synthesis_node finished | `{research_brief: "...", ticker: "AAPL"}` |
| `error` | Unhandled exception | `{message: "...", ticker: "AAPL"}` |

---

## 12. Step 7 — FastAPI Layer

### App Factory (`api/app.py`)

```python
def create_app() -> FastAPI:
    app = FastAPI(title="FinAgent...", lifespan=lifespan)

    app.add_middleware(CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(research_router)

    @app.get("/health")
    async def health():
        return HealthResponse(model=settings.claude_model)

    return app

app = create_app()  # module-level — imported by uvicorn and by tests
```

CORS origins include `3000` (Create React App default) and `5173` (Vite default) to support
the planned React frontend without a proxy.

### Research Router (`api/routers/research_router.py`)

**POST `/api/research/`** — synchronous

```python
@router.post("/", response_model=ResearchResponse)
async def research(request: ResearchRequest) -> ResearchResponse:
    final_state = run_research(ticker=request.ticker, question=request.question)
    return ResearchResponse(
        ticker=request.ticker,
        research_brief=final_state.get("research_brief", ""),
        errors=final_state.get("errors", []),
        tool_calls_made=sum(len(getattr(m, "tool_calls", [])) for m in final_state["messages"]),
    )
```

**GET `/api/research/stream/{ticker}`** — SSE stream

```python
@router.get("/stream/{ticker}")
async def stream(ticker: str) -> EventSourceResponse:
    ticker = ticker.upper().strip()

    async def event_generator():
        async for event in stream_research(ticker=ticker):
            yield {"event": event["type"], "data": json.dumps(event["data"])}

    return EventSourceResponse(event_generator())
```

`EventSourceResponse` from `sse-starlette` wraps the async generator and sends proper
SSE headers (`Content-Type: text/event-stream`, `Cache-Control: no-cache`).

### Pydantic Schemas (`api/models/schemas.py`)

```python
class ResearchRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10)
    question: str | None = Field(None, max_length=500)

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        v = v.upper().strip()
        if not re.match(r"^[A-Z]{1,10}$", v):
            raise ValueError("Ticker must be 1-10 uppercase letters only")
        return v
```

The regex `^[A-Z]{1,10}$` prevents injection, handles all real NYSE/NASDAQ tickers, and
rejects strings like `INVALID123!!`. Validation happens at the Pydantic layer before
the agent is invoked.

---

## 13. Step 8 — Server-Sent Events (SSE Streaming)

SSE is a one-directional protocol: server pushes events to the client over a persistent
HTTP connection. The browser's native `EventSource` API handles reconnection automatically.

### Server Side (FastAPI + sse-starlette)

```python
return EventSourceResponse(event_generator())
# event_generator() is an async generator that yields dicts:
# {"event": "tool_start", "data": '{"tool": "get_stock_price"}'}
```

Wire format over the HTTP connection:
```
event: tool_start
data: {"tool": "get_stock_price", "args": {"ticker": "AAPL"}}

event: tool_end
data: {"tool_call_id": "call_001", "content_preview": "current_price: 195..."}

event: complete
data: {"research_brief": "## EXECUTIVE SUMMARY\n...", "ticker": "AAPL"}
```

### Client Side (Browser)

```javascript
const source = new EventSource("http://localhost:8000/api/research/stream/AAPL");

source.addEventListener("tool_start", (e) => {
    const data = JSON.parse(e.data);
    console.log(`Calling tool: ${data.tool}`);
});

source.addEventListener("complete", (e) => {
    const data = JSON.parse(e.data);
    console.log(data.research_brief);
    source.close();
});

source.addEventListener("error", (e) => {
    console.error("Stream error", e);
    source.close();
});
```

### Testing SSE Without a Browser

```bash
# curl with -N disables buffering so events arrive in real time
curl -N http://localhost:8000/api/research/stream/AAPL
```

---

## 14. Step 9 — Docker and Containerisation

### Multi-Stage Dockerfile

The Dockerfile uses two stages to keep the final image small:

```dockerfile
# Stage 1: Builder — has build tools, compiles C extensions (numpy, pandas)
FROM python:3.12-slim AS builder
WORKDIR /app
RUN apt-get install -y build-essential curl
COPY requirements.txt .
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt
# --prefix=/install puts packages into /install, not the system Python

# Stage 2: Runtime — clean image, no build tools
FROM python:3.12-slim
WORKDIR /app

# Non-root user for security
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

# Copy compiled packages from builder
COPY --from=builder /install /usr/local

# Copy source code (owned by appuser)
COPY --chown=appuser:appgroup . .

USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["python", "-m", "uvicorn", "finagent.api.app:app", \
            "--host", "0.0.0.0", "--port", "8000"]
```

Why multi-stage?
- `build-essential` (~200MB) is only needed to compile numpy/pandas C extensions
- The runtime image has only the compiled `.so` files and Python bytecode
- Final image is ~400MB vs ~800MB single-stage

### docker-compose.yml (Current — API Only)

```yaml
version: '3.8'

services:
  finagent-api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: finagent-api
    ports:
      - "8000:8000"
    environment:
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}   # read from host .env
      CLAUDE_MODEL: ${CLAUDE_MODEL:-claude-3-sonnet-20240229}
      AGENT_MAX_ITERATIONS: ${AGENT_MAX_ITERATIONS:-10}
      API_DEBUG: ${API_DEBUG:-false}
    env_file:
      - .env
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: on-failure:3
```

### Build and Run

```bash
# Build and start
docker-compose up --build

# Rebuild after code change
docker-compose up --build --force-recreate

# View logs
docker-compose logs -f finagent-api

# Stop
docker-compose down
```

---

## 15. Step 10 — Testing Strategy

### Unit Tests

All external calls are mocked. Tests never hit real APIs, never need `ANTHROPIC_API_KEY`.

**Pattern for tool tests:**
```python
@patch("finagent.tools.market.market_tools.yf.Ticker")
def test_returns_price_data_for_valid_ticker(self, mock_ticker_cls):
    mock_ticker = MagicMock()
    mock_ticker_cls.return_value = mock_ticker
    mock_ticker.info = {"volume": 55_000_000, ...}
    mock_ticker.history.return_value = pd.DataFrame(
        {"Close": [190.0, 195.0]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )

    result = get_stock_price.invoke({"ticker": "AAPL"})  # .invoke() because it's a @tool

    assert result["ticker"] == "AAPL"
    assert result["current_price"] == 195.0
```

Note: `@tool` decorated functions must be called with `.invoke(args_dict)`, not directly.

**Pattern for node tests (pytest-mock):**
```python
def test_returns_tool_message_for_valid_tool(self, mocker):
    mocker.patch(
        "finagent.agent.nodes.tool_node.TOOL_MAP",
        {"get_stock_price": MagicMock(invoke=lambda args: {"ticker": "AAPL", "current_price": 195.0})}
    )
    state = self._make_state("get_stock_price", {"ticker": "AAPL"})
    result = tool_node(state)
    assert "195.0" in result["messages"][0].content
```

### Integration Tests

Uses `httpx.AsyncClient` with `ASGITransport` — **no live server needed**:

```python
@pytest.fixture
async def client(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac

async def test_valid_ticker_returns_research_brief(self, client):
    mock_state = {"research_brief": "## EXECUTIVE SUMMARY...", "errors": [], "messages": []}

    with patch("finagent.api.routers.research_router.run_research", return_value=mock_state):
        response = await client.post("/api/research/", json={"ticker": "AAPL"})

    assert response.status_code == 200
    assert "EXECUTIVE SUMMARY" in response.json()["research_brief"]
```

### Running Tests

```bash
# All tests
pytest tests/ -v

# Only unit tests
pytest tests/unit/ -v

# Only integration tests
pytest tests/integration/ -v

# Specific test file
pytest tests/unit/test_market_tools.py -v

# With coverage
pytest tests/ --cov=finagent --cov-report=term-missing
```

### pytest.ini Configuration

```ini
[pytest]
asyncio_mode = auto          # all async tests run without @pytest.mark.asyncio
testpaths = tests
python_files = test_*.py
log_cli = true               # shows logger output in terminal during tests
log_cli_level = INFO
```

### Test Naming Convention

Every test: `test_<what>_<condition>_<expected>`

Examples:
- `test_returns_price_data_for_valid_ticker`
- `test_returns_error_for_empty_history`
- `test_handles_yfinance_exception_gracefully`
- `test_routes_to_end_when_max_iterations_reached`

---

## 16. What Is NOT Built Yet (Roadmap)

The following sections document the planned infrastructure additions. The current system
has no database, no message queue, and no cache. This is intentional — the MVP is
API-only.

### 1. React Frontend (High Priority)

Stack: React 18 + TypeScript + Vite + Recharts + Tailwind CSS

Key components:
- `TickerInput` — validates and submits the ticker
- `ProgressFeed` — listens to SSE stream, renders tool events as they arrive
- `ResearchBrief` — renders the final markdown-formatted brief
- `PriceChart` — Recharts `LineChart` fed by `price_data.history` from the API response

SSE connection pattern:
```typescript
const source = new EventSource(`/api/research/stream/${ticker}`);
source.addEventListener("tool_start", handler);
source.addEventListener("complete", handler);
```

### 2. Caching Layer (Redis)

Research runs are expensive (~30-60s, ~$0.10). Results are cacheable by `ticker + date`.

Planned design:
- **Cache key:** `research:{ticker}:{YYYY-MM-DD}`
- **TTL:** 6 hours during market hours, 24 hours overnight
- **Storage:** Redis in production, in-memory `dict` in development
- **Headers:** `X-Cache: HIT/MISS`, `Cache-Control: max-age=21600`
- Add a `cache.py` module, call `cache.get()` before running the agent and
  `cache.set()` after

### 3. Persistent Research History (PostgreSQL)

Planned schema:
```sql
CREATE TABLE research_runs (
    id          BIGSERIAL PRIMARY KEY,
    ticker      VARCHAR(10)  NOT NULL,
    question    TEXT,
    brief       TEXT,
    errors      JSONB,
    model_used  VARCHAR(50),
    tool_calls  INTEGER,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_research_runs_ticker ON research_runs(ticker);
CREATE INDEX idx_research_runs_created_at ON research_runs(created_at DESC);
```

Planned implementation:
- SQLAlchemy async (`sqlalchemy[asyncio]` + `asyncpg`) for all DB calls
- Alembic for migrations
- New endpoints:
  - `GET /api/research/history/{ticker}` — last N runs for a ticker
  - `GET /api/research/history` — recent runs across all tickers

### 4. Kafka Event Streaming (Future)

**Why Kafka?** For a production research platform serving thousands of users,
you would not run the agent synchronously inside the HTTP request. Instead:

```
HTTP Request → FastAPI → Kafka topic: "research-requests"
                                ↓
                    Consumer Group: "agent-workers"
                        ↓
                    Run agent asynchronously
                        ↓
                    Kafka topic: "research-results"
                        ↓
                    Persist to PostgreSQL
                        ↓
                    Notify client via WebSocket or polling
```

**Producer** (in the API):
```python
from aiokafka import AIOKafkaProducer

producer = AIOKafkaProducer(bootstrap_servers="kafka:9092")
await producer.send("research-requests", value=json.dumps({
    "ticker": "AAPL",
    "request_id": str(uuid4()),
    "timestamp": datetime.utcnow().isoformat(),
}).encode())
```

**Consumer** (separate worker service):
```python
from aiokafka import AIOKafkaConsumer

consumer = AIOKafkaConsumer("research-requests", bootstrap_servers="kafka:9092",
                              group_id="agent-workers")
async for msg in consumer:
    payload = json.loads(msg.value)
    result = run_research(payload["ticker"])
    await producer.send("research-results", value=json.dumps(result).encode())
```

**docker-compose additions for Kafka:**
```yaml
services:
  zookeeper:
    image: confluentinc/cp-zookeeper:7.6.0
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181

  kafka:
    image: confluentinc/cp-kafka:7.6.0
    depends_on: [zookeeper]
    ports:
      - "9092:9092"
    environment:
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
```

**When to add Kafka:**
- When research requests need to be queued (burst traffic)
- When you want to decouple the API from the agent (scale them independently)
- When you need audit trail of every request/response
- When the agent runs on a separate machine/pod from the API

**For the current portfolio project:** Kafka is overkill. The SSE streaming approach
gives real-time feedback without a message queue. The comment above is for interview
conversations about scaling.

### 5. Full docker-compose with PostgreSQL + Redis

```yaml
version: '3.8'

services:
  finagent-api:
    build: .
    ports: ["8000:8000"]
    depends_on:
      postgres: {condition: service_healthy}
      redis: {condition: service_healthy}
    environment:
      DATABASE_URL: postgresql+asyncpg://finagent:finagent@postgres:5432/finagent
      REDIS_URL: redis://redis:6379/0
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
    env_file: [.env]

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: finagent
      POSTGRES_PASSWORD: finagent
      POSTGRES_DB: finagent
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports: ["5432:5432"]
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U finagent"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  postgres_data:
```

### 6. Rate Limiting (slowapi)

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@router.post("/")
@limiter.limit("5/minute")
async def research(request: Request, body: ResearchRequest):
    ...
```

---

## 17. How a Full Request Flows End-to-End

Here is the complete lifecycle of `POST /api/research/ {"ticker": "AAPL"}`:

```
1. HTTP POST arrives at FastAPI

2. ResearchRequest Pydantic model validates:
   - ticker is 1-10 uppercase letters only (regex)
   - question is optional, ≤500 chars

3. research() route handler calls run_research(ticker="AAPL")

4. runner.py builds initial AgentState:
   messages = [HumanMessage("Please produce a comprehensive research brief for AAPL")]
   ticker = "AAPL"
   price_data = {}   (all empty)
   iteration_count = 0

5. finagent_graph.invoke(initial_state) starts execution

6. ITERATION 1 — llm_node:
   - No SystemMessage in history yet → prepend SYSTEM_PROMPT
   - Call Claude with [SystemMessage, HumanMessage]
   - Claude responds: AIMessage(tool_calls=[
       {name: "get_stock_price",          args: {ticker: "AAPL"}, id: "call_001"},
       {name: "get_price_history",        args: {ticker: "AAPL"}, id: "call_002"},
       {name: "get_company_fundamentals", args: {ticker: "AAPL"}, id: "call_003"},
       {name: "get_recent_news",          args: {ticker: "AAPL"}, id: "call_004"},
       {name: "get_earnings_calendar",    args: {ticker: "AAPL"}, id: "call_005"},
       {name: "get_sec_filings",          args: {ticker: "AAPL"}, id: "call_006"},
       {name: "get_sec_facts",            args: {ticker: "AAPL"}, id: "call_007"},
       {name: "get_portfolio_metrics",    args: {ticker: "AAPL"}, id: "call_008"},
       {name: "get_peer_comparison",      args: {ticker: "AAPL"}, id: "call_009"},
     ])
   - iteration_count becomes 1
   - should_continue sees tool_calls → routes to "tools"

7. tool_node:
   - Iterates over all 9 tool_calls
   - Calls each tool via TOOL_MAP[name].invoke(args)
   - Stores results in state fields (price_data, company_info, etc.)
   - Appends 9 ToolMessages to state
   - Routes back to "llm"

8. ITERATION 2 — llm_node:
   - Now has SystemMessage + HumanMessage + AIMessage(tool_calls) + 9×ToolMessages
   - Claude reads all tool results and synthesises the brief
   - Returns: AIMessage(content="## EXECUTIVE SUMMARY\nApple Inc...")
   - iteration_count becomes 2
   - should_continue sees no tool_calls → routes to "synthesis"

9. synthesis_node:
   - Extracts content from last AIMessage
   - state["research_brief"] = "## EXECUTIVE SUMMARY\nApple Inc..."

10. Graph reaches END → finagent_graph.invoke() returns final_state

11. ResearchResponse built from final_state:
    {
      "ticker": "AAPL",
      "research_brief": "## EXECUTIVE SUMMARY\n...",
      "errors": [],
      "tool_calls_made": 9
    }

12. JSON response returned to client. Total time: ~30-60 seconds.
```

---

## 18. Planned Infrastructure: Postgres, Kafka, Redis

This section is a detailed expansion of the infrastructure not yet implemented but
architecturally designed.

### PostgreSQL — Research History

**Why PostgreSQL over SQLite:**
- SQLite is single-writer. For a web app with concurrent requests, this creates write contention.
- PostgreSQL supports async drivers (`asyncpg`), full JSONB for storing `errors` arrays,
  and row-level locking.

**ORM choice: SQLAlchemy async**
```python
# db/models.py
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Text, Integer, DateTime, func
from sqlalchemy.dialects.postgresql import JSONB

class Base(DeclarativeBase):
    pass

class ResearchRun(Base):
    __tablename__ = "research_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    ticker: Mapped[str] = mapped_column(String(10), index=True)
    question: Mapped[str | None] = mapped_column(Text)
    brief: Mapped[str | None] = mapped_column(Text)
    errors: Mapped[dict] = mapped_column(JSONB, default=list)
    model_used: Mapped[str] = mapped_column(String(50))
    tool_calls: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                   server_default=func.now())
```

**Async engine setup:**
```python
# db/engine.py
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

engine = create_async_engine(settings.database_url, pool_size=10, max_overflow=20)
AsyncSession = async_sessionmaker(engine, expire_on_commit=False)
```

**Alembic migrations:**
```bash
alembic init alembic
# Edit alembic/env.py to import Base.metadata
alembic revision --autogenerate -m "create research_runs table"
alembic upgrade head
```

**How it integrates with the research flow:**
```python
# In research_router.py (after Postgres is added)
async def research(request: ResearchRequest, db: AsyncSession = Depends(get_db)):
    final_state = run_research(ticker=request.ticker)

    run = ResearchRun(
        ticker=request.ticker,
        brief=final_state["research_brief"],
        errors=final_state["errors"],
        model_used=settings.claude_model,
    )
    db.add(run)
    await db.commit()

    return ResearchResponse(...)
```

### Redis — Caching

**Why Redis over in-memory dict:**
- In-memory cache dies when the process restarts or when running multiple API replicas
- Redis persists across restarts and is shared across all API replicas behind a load balancer

**Implementation plan:**
```python
# cache/redis_cache.py
import redis.asyncio as aioredis
import json

redis = aioredis.from_url(settings.redis_url, decode_responses=True)

CACHE_KEY = "research:{ticker}:{date}"
TTL_MARKET_HOURS = 6 * 3600    # 6 hours
TTL_AFTER_HOURS = 24 * 3600    # 24 hours

async def get_cached(ticker: str) -> dict | None:
    key = CACHE_KEY.format(ticker=ticker, date=date.today().isoformat())
    raw = await redis.get(key)
    return json.loads(raw) if raw else None

async def set_cached(ticker: str, data: dict) -> None:
    key = CACHE_KEY.format(ticker=ticker, date=date.today().isoformat())
    ttl = TTL_MARKET_HOURS if _is_market_hours() else TTL_AFTER_HOURS
    await redis.set(key, json.dumps(data), ex=ttl)
```

**Integration point** — in `research_router.py`:
```python
async def research(request: ResearchRequest):
    cached = await get_cached(request.ticker)
    if cached:
        return ResearchResponse(**cached)  # X-Cache: HIT

    final_state = run_research(ticker=request.ticker)
    await set_cached(request.ticker, final_state)
    return ResearchResponse(...)            # X-Cache: MISS
```

### Kafka — Async Job Queue

**Why Kafka over HTTP:**
- Decouples API availability from agent worker availability
- If the agent crashes mid-run, the message stays in Kafka for retry
- Can scale workers independently from the API

**Topic design:**
- `finagent.research.requests` — incoming ticker requests
- `finagent.research.results` — completed research briefs
- `finagent.research.errors` — failed runs for alerting

**Partition strategy:** Partition by ticker symbol so that requests for the same ticker
always go to the same consumer (locality for cache warming).

**Offset management:** Use `earliest` for consumer recovery (reprocess failed messages)
and `latest` for new consumers joining mid-stream.

**When to add Kafka (signal check):**
- Research requests are queueing (>5 concurrent runs saturate the API)
- You need to retry failed runs automatically
- You need an audit trail in Kafka Streams or KSQLDB
- You're adding a "research-subscription" feature (alert when AAPL run completes)

---

## 19. Error Handling Philosophy

**Three layers of error handling:**

**Layer 1 — Tools (never raise):**
```python
@tool
def get_stock_price(ticker: str) -> dict:
    try:
        ...
    except Exception as e:
        return {"error": str(e), "ticker": ticker}  # structured error dict
```
Claude receives the error as a `ToolMessage` and reasons around it:
_"SEC data unavailable. I'll note this in the risk section and continue with market data."_

**Layer 2 — Nodes (catch, log, store in errors):**
```python
def tool_node(state: AgentState) -> dict:
    for tool_call in last_message.tool_calls:
        try:
            result = TOOL_MAP[tool_name].invoke(tool_args)
        except Exception as e:
            state_updates["errors"].append(f"Tool '{tool_name}' raised: {e}")
            # continue to next tool — never re-raise
```

**Layer 3 — API (catch, return 500):**
```python
@router.post("/")
async def research(request: ResearchRequest):
    try:
        return ResearchResponse(...)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**Result:** A research run where the SEC EDGAR API is down returns a 200 with a partial
brief that notes "SEC data unavailable." Not a 500. This is more useful to the user.

---

## 20. Key Design Decisions (ADRs)

These are documented in `ADR.md`. Summary:

| Decision | Choice | Reason |
|---|---|---|
| Agentic framework | LangGraph StateGraph | Inspectable graph, testable nodes, `astream` support |
| Message accumulation | `add_messages` reducer | Claude needs full history to avoid re-calling tools |
| Tool error policy | Return dict, never raise | Partial data > total failure for financial research |
| Streaming protocol | SSE over WebSockets | Server-push only, HTTP-native, no proxy config needed |
| State design | Typed fields + messages | Clean domain model for API serialisation, not just raw messages |
| LLM model | claude-3-sonnet-20240229 | Cost/quality balance; configurable via `CLAUDE_MODEL` env var |
| System prompt injection | Once, first iteration only | Prevents duplicate system messages on re-entry to llm_node |
| Ticker validation | `^[A-Z]{1,10}$` regex in Pydantic | Injection prevention, handles all real exchange tickers |

---

## Quick Reference — Common Commands

```bash
# Start server
python main.py

# Sync research
curl -X POST http://localhost:8000/api/research/ \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL"}'

# Stream research
curl -N http://localhost:8000/api/research/stream/AAPL

# Health check
curl http://localhost:8000/health

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=finagent --cov-report=term-missing

# Docker
docker-compose up --build
docker-compose down

# Interactive API docs
open http://localhost:8000/docs
```
