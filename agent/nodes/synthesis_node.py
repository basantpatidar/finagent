import logging
from langchain_core.messages import AIMessage

from finagent.agent.state import AgentState

logger = logging.getLogger(__name__)


def synthesis_node(state: AgentState) -> dict:
    """
    Final node — extracts the research brief text from the last AIMessage
    and stores it in state.research_brief for the API to stream/return.

    This node only fires after the LLM has produced a final text response
    (no more tool calls), so the last AIMessage should be the full brief.
    """
    logger.info("[SynthesisNode] Extracting final research brief")

    last_message = state["messages"][-1]

    if isinstance(last_message, AIMessage) and isinstance(last_message.content, str):
        brief = last_message.content
    elif isinstance(last_message, AIMessage) and isinstance(last_message.content, list):
        # Content may be a list of blocks (text + tool_use)
        text_blocks = [
            block.get("text", "")
            for block in last_message.content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        brief = "\n".join(text_blocks)
    else:
        brief = "Research complete. See message history for full analysis."
        logger.warning("[SynthesisNode] Could not extract text from last message type=%s", type(last_message))

    logger.info("[SynthesisNode] Brief length=%d chars", len(brief))

    return {"research_brief": brief}
