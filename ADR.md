# Architecture Decision Records — FinAgent

## ADR-001: LangGraph over Custom ReAct Loop

**Status:** Accepted

**Context:** Building an agentic loop (LLM → tools → LLM → ... → final response) can
be done with a raw while loop. However, as tool count and complexity grow, raw loops
become hard to debug, test, and extend.

**Decision:** Use LangGraph's `StateGraph`. Each step (LLM call, tool execution,
synthesis) is an explicit named node. Routing logic is a pure function (`should_continue`).
State flows as a typed `TypedDict`.

**Consequences:** The graph is inspectable (can visualise with `graph.get_graph().draw_mermaid()`),
each node is independently unit-testable, and adding new nodes (e.g. a fact-check node)
requires only adding a node + edge without touching existing logic.

---

## ADR-002: `add_messages` Reducer for Full History

**Status:** Accepted

**Context:** Claude needs the full message history (HumanMessage → AIMessage with tool_calls
→ ToolMessages → AIMessage → ...) to understand what tools have already been called and
what their results were.

**Decision:** Use LangGraph's `add_messages` annotation on `AgentState.messages`. This
appends each new message rather than overwriting, automatically maintaining the complete
conversation thread.

**Consequences:** The context window grows with each tool call. With 9 tools and JSON
tool results, a full run is approximately 8-15k tokens. Well within Claude's 200k limit.

---

## ADR-003: Tools Return Error Dicts, Never Raise

**Status:** Accepted

**Context:** Financial data APIs are unreliable. SEC EDGAR times out. yfinance rate-limits.
News feeds return empty. If any tool raises, the LangGraph node catches it, but the
agent's reasoning is disrupted.

**Decision:** Every tool catches all exceptions internally and returns a structured dict:
`{"error": "...", "ticker": "..."}`. The LLM receives the error as a ToolMessage and
can reason around it ("SEC data unavailable, continuing with market data only...").

**Consequences:** Research runs never hard-fail. The brief may note missing data sources,
which is more useful to the user than a 500 error.

---

## ADR-004: SSE over WebSockets for Streaming

**Status:** Accepted

**Context:** We need to stream tool progress events from server to client in real time.
Both SSE and WebSockets could work.

**Decision:** Use Server-Sent Events (SSE) via `sse-starlette`. SSE is:
- One-directional (server → client), which is all we need
- HTTP-native, works through proxies and load balancers without configuration
- Built-in reconnection and `EventSource` browser API
- Simpler to implement and test than WebSockets

**Consequences:** Cannot receive messages from the client mid-stream. If interactive
"redirect the research" capability is needed in future, we would upgrade to WebSockets.

---

## ADR-005: Structured State Fields Alongside Raw Messages

**Status:** Accepted

**Context:** The API response needs to return structured financial data (not just the
text brief) so the frontend (FinDash) can render charts, tables, and metrics from
typed fields rather than parsing the research text.

**Decision:** `AgentState` holds both `messages` (for LLM history) and typed fields
(`price_data`, `company_info`, `sec_filings`, `portfolio_metrics`). The `tool_node`
populates both simultaneously.

**Consequences:** Slight redundancy (data exists in both messages and state fields)
but gives us a clean domain model that the API layer can serialise directly. This
mirrors the pattern from T. Rowe Price systems where raw API payloads were always
mapped to domain objects.

---

## ADR-006: Single `claude-3-sonnet-20240229` Model

**Status:** Accepted

**Context:** Claude Opus would produce higher-quality analysis but at ~5x higher cost
per run. Claude Haiku is faster but lacks the reasoning depth for multi-tool financial
analysis.

**Decision:** Use `claude-3-sonnet-20240229` as the default. It balances quality, speed, and
cost for a portfolio project. The model is configurable via `CLAUDE_MODEL` env var.

**Consequences:** Each research run costs approximately $0.08-0.15 at current pricing.
Acceptable for a portfolio demo. A Redis-backed caching layer keyed on `ticker + date`
would reduce costs significantly for repeated queries on the same ticker.
