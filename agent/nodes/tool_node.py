import json
import logging
from langchain_core.messages import ToolMessage

from finagent.agent.state import AgentState
from finagent.tools import TOOL_MAP

logger = logging.getLogger(__name__)


def tool_node(state: AgentState) -> dict:
    """
    Tool executor node — finds the last AIMessage with tool_calls,
    runs each requested tool, and returns ToolMessages with results.

    Results are also stored in structured state fields so downstream
    nodes and the API can access them directly without parsing messages.
    """
    last_message = state["messages"][-1]

    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        logger.warning("[ToolNode] Called but no tool_calls found on last message")
        return {"messages": []}

    tool_messages = []
    state_updates: dict = {
        "price_data": state.get("price_data", {}),
        "company_info": state.get("company_info", {}),
        "financials": state.get("financials", {}),
        "news_articles": state.get("news_articles", []),
        "sec_filings": state.get("sec_filings", []),
        "portfolio_metrics": state.get("portfolio_metrics", {}),
        "errors": state.get("errors", []),
    }

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        logger.info("[ToolNode] Executing tool=%s args=%s", tool_name, tool_args)

        if tool_name not in TOOL_MAP:
            error_msg = f"Unknown tool '{tool_name}'"
            logger.error("[ToolNode] %s", error_msg)
            tool_messages.append(
                ToolMessage(content=json.dumps({"error": error_msg}), tool_call_id=tool_id)
            )
            state_updates["errors"].append(error_msg)
            continue

        try:
            result = TOOL_MAP[tool_name].invoke(tool_args)

            # Store in structured state fields for easy access
            _store_result(tool_name, result, state_updates)

            tool_messages.append(
                ToolMessage(
                    content=json.dumps(result, default=str),
                    tool_call_id=tool_id,
                )
            )
            logger.info("[ToolNode] Tool=%s completed successfully", tool_name)

        except Exception as e:
            error_msg = f"Tool '{tool_name}' raised exception: {e}"
            logger.error("[ToolNode] %s", error_msg, exc_info=True)
            tool_messages.append(
                ToolMessage(
                    content=json.dumps({"error": error_msg}),
                    tool_call_id=tool_id,
                )
            )
            state_updates["errors"].append(error_msg)

    return {"messages": tool_messages, **state_updates}


def _store_result(tool_name: str, result: dict, state: dict) -> None:
    """Route tool output into the correct structured state field."""
    if tool_name in ("get_stock_price", "get_price_history"):
        state["price_data"].update(result)

    elif tool_name in ("get_company_fundamentals", "get_peer_comparison"):
        state["company_info"].update(result)

    elif tool_name in ("get_sec_facts",):
        state["financials"].update(result)

    elif tool_name in ("get_recent_news", "get_earnings_calendar"):
        if "articles" in result:
            state["news_articles"].extend(result.get("articles", []))
        else:
            # earnings calendar — merge into news context
            state["news_articles"].append(result)

    elif tool_name in ("get_sec_filings",):
        state["sec_filings"].extend(result.get("filings", []))

    elif tool_name in ("get_portfolio_metrics",):
        state["portfolio_metrics"].update(result)
