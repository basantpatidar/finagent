import pytest
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


class TestShouldContinueRouter:

    def _make_state(self, messages, iteration_count=0):
        return {
            "messages": messages,
            "ticker": "AAPL",
            "price_data": {},
            "company_info": {},
            "financials": {},
            "news_articles": [],
            "sec_filings": [],
            "portfolio_metrics": {},
            "research_brief": "",
            "errors": [],
            "iteration_count": iteration_count,
        }

    def test_routes_to_tools_when_tool_calls_present(self):
        from finagent.agent.graph.agent_graph import should_continue

        ai_msg = AIMessage(content="", tool_calls=[
            {"name": "get_stock_price", "args": {"ticker": "AAPL"}, "id": "call_001"}
        ])
        state = self._make_state([HumanMessage(content="Research AAPL"), ai_msg])

        assert should_continue(state) == "tools"

    def test_routes_to_synthesis_when_no_tool_calls(self):
        from finagent.agent.graph.agent_graph import should_continue

        ai_msg = AIMessage(content="Here is the research brief for AAPL...")
        state = self._make_state([HumanMessage(content="Research AAPL"), ai_msg])

        assert should_continue(state) == "synthesis"

    def test_routes_to_end_when_max_iterations_reached(self):
        from finagent.agent.graph.agent_graph import should_continue

        ai_msg = AIMessage(content="", tool_calls=[
            {"name": "get_stock_price", "args": {"ticker": "AAPL"}, "id": "call_001"}
        ])
        state = self._make_state(
            [HumanMessage(content="Research AAPL"), ai_msg],
            iteration_count=10  # at max
        )

        assert should_continue(state) == "end"

    def test_routes_to_end_before_tool_calls_when_over_max(self):
        from finagent.agent.graph.agent_graph import should_continue

        ai_msg = AIMessage(content="", tool_calls=[
            {"name": "get_stock_price", "args": {"ticker": "AAPL"}, "id": "call_001"}
        ])
        state = self._make_state(
            [HumanMessage(content="Research AAPL"), ai_msg],
            iteration_count=99
        )

        assert should_continue(state) == "end"


class TestToolNode:

    def _make_state(self, tool_name, tool_args, tool_call_id="call_001"):
        from langchain_core.messages import AIMessage
        ai_msg = AIMessage(
            content="",
            tool_calls=[{"name": tool_name, "args": tool_args, "id": tool_call_id}]
        )
        return {
            "messages": [ai_msg],
            "ticker": "AAPL",
            "price_data": {},
            "company_info": {},
            "financials": {},
            "news_articles": [],
            "sec_filings": [],
            "portfolio_metrics": {},
            "research_brief": "",
            "errors": [],
            "iteration_count": 1,
        }

    def test_returns_tool_message_for_valid_tool(self, mocker):
        from finagent.agent.nodes.tool_node import tool_node

        mocker.patch(
            "finagent.agent.nodes.tool_node.TOOL_MAP",
            {
                "get_stock_price": MagicMock(
                    invoke=lambda args: {"ticker": "AAPL", "current_price": 195.0}
                )
            }
        )

        state = self._make_state("get_stock_price", {"ticker": "AAPL"})
        result = tool_node(state)

        assert len(result["messages"]) == 1
        assert result["messages"][0].tool_call_id == "call_001"
        assert "195.0" in result["messages"][0].content

    def test_returns_error_message_for_unknown_tool(self):
        from finagent.agent.nodes.tool_node import tool_node

        state = self._make_state("nonexistent_tool", {"ticker": "AAPL"})
        result = tool_node(state)

        assert len(result["messages"]) == 1
        assert "error" in result["messages"][0].content.lower()
        assert len(result["errors"]) == 1

    def test_handles_tool_exception_without_crashing(self, mocker):
        from finagent.agent.nodes.tool_node import tool_node

        mocker.patch(
            "finagent.agent.nodes.tool_node.TOOL_MAP",
            {
                "get_stock_price": MagicMock(
                    invoke=MagicMock(side_effect=RuntimeError("API timeout"))
                )
            }
        )

        state = self._make_state("get_stock_price", {"ticker": "AAPL"})
        result = tool_node(state)

        # Should not raise — returns error ToolMessage instead
        assert len(result["messages"]) == 1
        assert len(result["errors"]) == 1
        assert "API timeout" in result["errors"][0]


class TestSynthesisNode:

    def _make_state(self, last_message):
        return {
            "messages": [last_message],
            "ticker": "AAPL",
            "price_data": {},
            "company_info": {},
            "financials": {},
            "news_articles": [],
            "sec_filings": [],
            "portfolio_metrics": {},
            "research_brief": "",
            "errors": [],
            "iteration_count": 5,
        }

    def test_extracts_brief_from_string_content(self):
        from finagent.agent.nodes.synthesis_node import synthesis_node

        ai_msg = AIMessage(content="## EXECUTIVE SUMMARY\nApple is a great company...")
        state = self._make_state(ai_msg)

        result = synthesis_node(state)

        assert result["research_brief"] == "## EXECUTIVE SUMMARY\nApple is a great company..."

    def test_extracts_brief_from_list_content(self):
        from finagent.agent.nodes.synthesis_node import synthesis_node

        ai_msg = AIMessage(content=[
            {"type": "text", "text": "## EXECUTIVE SUMMARY\nApple Inc..."},
            {"type": "text", "text": "\n**PRICE & TECHNICALS**\nPrice: $195"},
        ])
        state = self._make_state(ai_msg)

        result = synthesis_node(state)

        assert "EXECUTIVE SUMMARY" in result["research_brief"]
        assert "PRICE & TECHNICALS" in result["research_brief"]
