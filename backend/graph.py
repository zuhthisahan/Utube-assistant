from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

# Local Imports
from .state import AppState
from .nodes import ingest_video_node, agent_node
from .tools import ALL_TOOLS

# 1. Define Logic for Conditional Edge
def should_continue(state: AppState):
    messages = state['messages']
    last_message = messages[-1]
    
    # If the LLM made a tool call, go to "tools"
    if last_message.tool_calls:
        return "tools"
    return END

# 2. Build Graph
workflow = StateGraph(AppState)

# Add Nodes
workflow.add_node("ingest", ingest_video_node)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(ALL_TOOLS))

# Set the Entry Point Logic : The Router
def router(state: AppState):
    # Check the LAST message from the user
    last_msg = state['messages'][-1].content
    if "youtube.com" in last_msg or "youtu.be" in last_msg:
        return "ingest"
    return "agent"

workflow.set_conditional_entry_point(router, {
    "ingest": "ingest",
    "agent": "agent"
})

# Connections
# workflow.add_edge("ingest", "agent")
workflow.add_edge("ingest", END)
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")

# Compile
app = workflow.compile()