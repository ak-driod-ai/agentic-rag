from typing import TypedDict, List, Dict, Any


class AgentState(TypedDict, total=False):
    # Multi-turn conversation history
    chat_history: List[Dict[str, str]]

    # Original user question in current turn
    question: str

    # Contextualized/rewritten query used for retrieval
    current_query: str

    # Router decision: "direct_answer" vs "retrieve_kb"
    route: str

    # Private KB documents from Qdrant
    private_docs: list

    # Web search results from Tavily
    web_results: list

    # Evaluation results: "good" vs "weak"
    kb_grade: str
    web_grade: str

    # Number of query rewrites attempted
    retry_count: int

    # Final synthesized answer
    answer: str

    # Source attribution (Private KB, Web, Direct)
    source_used: str

    # List of citation strings or URLs
    sources: list

    # Step-by-step decision log
    decision_trace: list