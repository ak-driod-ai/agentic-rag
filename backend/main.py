import sys
import os

# Ensure UTF-8 output encoding across Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from .graph import build_graph
from .ingest import ingest


# Compile the graph once
_agent_graph = None


def get_agent():
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = build_graph()
    return _agent_graph


def run_agent(question: str) -> dict:
    """Executes the Agentic RAG graph with a user question."""
    agent = get_agent()

    initial_state = {
        "question": question,
        "current_query": question,
        "retry_count": 0,
        "private_docs": [],
        "web_results": [],
        "sources": [],
        "decision_trace": []
    }

    result = agent.invoke(initial_state)
    return result


def display_result(result: dict):
    print("\n" + "=" * 60)
    print("[AGENT RESPONSE]")
    print("=" * 60)
    print(result.get("answer", "No answer generated."))

    print("\n" + "=" * 60)
    print("[SOURCE USED]")
    print("=" * 60)
    print(result.get("source_used", "N/A"))

    sources = result.get("sources", [])
    if sources:
        print("\n" + "=" * 60)
        print("[CITATIONS / SOURCES]")
        print("=" * 60)
        for s in sources:
            print(f" * {s}")

    trace = result.get("decision_trace", [])
    if trace:
        print("\n" + "=" * 60)
        print("[DECISION TRACE & GRAPH EXECUTION]")
        print("=" * 60)
        for step in trace:
            print(f" -> {step}")
    print("=" * 60 + "\n")


def serve():
    import uvicorn
    print("\n" + "=" * 60)
    print(" 🚀 Starting Agentic RAG FastAPI Streaming Server")
    print(" 🌐 Web UI & Live Stream:  http://127.0.0.1:8000")
    print(" 📘 Interactive API Docs:  http://127.0.0.1:8000/docs")
    print("=" * 60 + "\n")
    uvicorn.run("backend.api:app", host="127.0.0.1", port=8000, reload=False)


def main():
    if "--serve" in sys.argv or "--api" in sys.argv:
        serve()
        return

    print("=" * 60)
    print(" Novatech Agentic RAG Assistant (LangGraph + Groq + Qdrant + Tavily)")
    print("=" * 60)

    # Ensure KB is loaded
    if "--ingest" in sys.argv:
        ingest()

    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        question = " ".join(sys.argv[1:])
        result = run_agent(question)
        display_result(result)
        return

    while True:
        try:
            question = input("\nAsk a question (or type 'exit' to quit): ").strip()
            if not question or question.lower() in ["exit", "quit", "q"]:
                break
            result = run_agent(question)
            display_result(result)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break


if __name__ == "__main__":
    main()