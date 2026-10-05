from typing import TypedDict


class AgentState(TypedDict, total=False):

    # Original user question
    question: str

    # Query currently being used for retrieval
    current_query: str

    # Router decision
    route: str

    # Private KB documents
    private_docs: list

    # Tavily results
    web_results: list

    # Evaluation results
    kb_grade: str
    web_grade: str

    # Number of rewrites
    retry_count: int

    # Final response
    answer: str

    # Where answer came from
    source_used: str

    # URLs / source references
    sources: list

    # Explain agent decisions
    decision_trace: list