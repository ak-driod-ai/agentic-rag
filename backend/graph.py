from langgraph.graph import (
    StateGraph,
    END
)

from .state import AgentState
from .nodes import (
    route_question,
    direct_answer,
    retrieve_private_docs,
    grade_private_docs,
    rewrite_query,
    retrieve_web,
    grade_web_results,
    generate_from_private_kb,
    generate_from_web,
    insufficient_answer
)
from .config import MAX_RETRIES


# ==========================================
# ROUTER CONDITIONAL EDGE
# ==========================================

def route_decision(state: AgentState) -> str:
    """
    Decides whether to answer directly (greetings / general chat)
    or query the private enterprise knowledge base.
    """
    if state.get("route") == "direct_answer":
        return "direct_answer"
    return "retrieve_private_docs"


# ==========================================
# PRIVATE KB CONDITIONAL EDGE
# ==========================================

def private_grade_decision(state: AgentState) -> str:
    """
    Evaluates private document relevance.
    If good -> generate answer from private doc.
    If weak -> fallback to web search via Tavily.
    """
    grade = state.get("kb_grade", "weak")
    if grade == "good":
        return "generate_private"
    return "retrieve_web"


# ==========================================
# WEB SEARCH CONDITIONAL EDGE
# ==========================================

def web_grade_decision(state: AgentState) -> str:
    """
    Evaluates Tavily web results.
    If good -> generate answer from web search results.
    If weak and retries remaining -> rewrite query and re-search.
    If weak and retries exhausted -> return insufficient evidence response.
    """
    grade = state.get("web_grade", "weak")
    retry_count = state.get("retry_count", 0)

    if grade == "good":
        return "generate_web"

    if retry_count < MAX_RETRIES:
        return "rewrite_query"

    return "insufficient"


# ==========================================
# BUILD AGENTIC RAG GRAPH
# ==========================================

def build_graph():
    workflow = StateGraph(AgentState)

    # 1. Add Nodes
    workflow.add_node("route_question", route_question)
    workflow.add_node("direct_answer", direct_answer)
    workflow.add_node("retrieve_private_docs", retrieve_private_docs)
    workflow.add_node("grade_private_docs", grade_private_docs)
    workflow.add_node("generate_private", generate_from_private_kb)
    workflow.add_node("retrieve_web", retrieve_web)
    workflow.add_node("grade_web_results", grade_web_results)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("generate_web", generate_from_web)
    workflow.add_node("insufficient", insufficient_answer)

    # 2. Entry Point
    workflow.set_entry_point("route_question")

    # 3. Router Conditional Edge
    workflow.add_conditional_edges(
        "route_question",
        route_decision,
        {
            "direct_answer": "direct_answer",
            "retrieve_private_docs": "retrieve_private_docs"
        }
    )

    # 4. Direct Answer terminates
    workflow.add_edge("direct_answer", END)

    # 5. Private Retrieval -> Private Grading
    workflow.add_edge("retrieve_private_docs", "grade_private_docs")

    # 6. Private Grading Conditional Edge
    workflow.add_conditional_edges(
        "grade_private_docs",
        private_grade_decision,
        {
            "generate_private": "generate_private",
            "retrieve_web": "retrieve_web"
        }
    )

    # 7. Private Generation terminates
    workflow.add_edge("generate_private", END)

    # 8. Web Retrieval -> Web Grading
    workflow.add_edge("retrieve_web", "grade_web_results")

    # 9. Web Grading Conditional Edge (with Rewrite Loop)
    workflow.add_conditional_edges(
        "grade_web_results",
        web_grade_decision,
        {
            "generate_web": "generate_web",
            "rewrite_query": "rewrite_query",
            "insufficient": "insufficient"
        }
    )

    # 10. Query Rewrite loops back to Web Retrieval
    workflow.add_edge("rewrite_query", "retrieve_web")

    # 11. Web Generation and Insufficient terminate
    workflow.add_edge("generate_web", END)
    workflow.add_edge("insufficient", END)

    return workflow.compile()