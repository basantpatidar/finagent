import logging
from langgraph.graph import StateGraph, END

from finagent.agent.state import AgentState
from finagent.agent.nodes.llm_node import llm_node
from finagent.agent.nodes.tool_node import tool_node
from finagent.agent.nodes.synthesis_node import synthesis_node
from finagent.config import settings

logger = logging.getLogger(__name__)


def should_continue(state: AgentState) -> str:
    """
    Routing function — decides what happens after the LLM node runs.

    Returns:
        'tools'     — LLM requested tool calls, execute them
        'synthesis' — LLM produced final text, extract the brief
        'end'       — Max iterations reached, bail out safely
    """
    # Guard: max iterations
    if state["iteration_count"] >= settings.agent_max_iterations:
        logger.warning(
            "[Router] Max iterations (%d) reached. Forcing end.",
            settings.agent_max_iterations,
        )
        return "end"

    last_message = state["messages"][-1]

    # If last message has tool_calls → go to tool executor
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        logger.info(
            "[Router] → tools (%d tool calls requested)",
            len(last_message.tool_calls),
        )
        return "tools"

    # No tool calls → LLM produced final response → synthesise
    logger.info("[Router] → synthesis (no tool calls, final response)")
    return "synthesis"


def build_graph() -> StateGraph:
    """
    Constructs and compiles the LangGraph StateGraph.

    Graph topology:
        START → llm → (conditional) → tools → llm  (loop)
                                    → synthesis → END
                                    → END (max iterations)
    """
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("llm", llm_node)
    graph.add_node("tools", tool_node)
    graph.add_node("synthesis", synthesis_node)

    # Entry point
    graph.set_entry_point("llm")

    # Conditional edge after LLM: call tools or finish
    graph.add_conditional_edges(
        "llm",
        should_continue,
        {
            "tools": "tools",
            "synthesis": "synthesis",
            "end": END,
        },
    )

    # After tools always go back to LLM
    graph.add_edge("tools", "llm")

    # After synthesis we're done
    graph.add_edge("synthesis", END)

    compiled = graph.compile()
    logger.info("[Graph] FinAgent graph compiled successfully")
    return compiled


# Module-level compiled graph — import and use directly
finagent_graph = build_graph()
