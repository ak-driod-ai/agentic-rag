import json
import re
from typing import Dict, Any

from .llm import get_llm
from .qdrant_store import search_documents
from .tavily_search import search_web
from .config import MAX_RETRIES


# ==========================================
# HELPER: DECISION TRACE & TEXT CLEANING
# ==========================================

def add_trace(state: Dict[str, Any], message: str) -> list:
    trace = list(state.get("decision_trace", []))
    trace.append(message)
    return trace


def clean_text_response(content: Any) -> str:
    """Strip thinking tags (<think>...</think>) and unnecessary whitespace."""
    if not isinstance(content, str):
        content = str(content)
    text = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    return text


def clean_json_response(content: Any) -> str:
    """Strip markdown code fences, remove think blocks, and extract JSON object."""
    text = clean_text_response(content)
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Match outermost JSON structure
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0).strip()
    return text


# ==========================================
# 1. ROUTER NODE
# ==========================================

def route_question(state: Dict[str, Any]) -> Dict[str, Any]:
    question = state["question"]
    llm = get_llm()

    prompt = f"""You are an intelligent query router for an enterprise Agentic RAG system for Novatech / NovaRetail.

Evaluate the user question and determine the routing path:

1. direct_answer:
   - Greetings, casual banter, pleasantries, simple conversational chat (e.g., "hi", "hello", "how are you", "who are you").
   - Simple static trivia (e.g., "tell me a joke").

2. retrieve_kb:
   - Novatech / NovaRetail internal policies, IT, VPN, passwords, MFA, laptops, software, support.
   - Any question requiring research, documentation, current events, recent news, external technologies, or facts that might need knowledge lookup or web search.

Return ONLY a JSON object with a single key "route" whose value is either "direct_answer" or "retrieve_kb".

Example:
{{"route": "direct_answer"}}
or
{{"route": "retrieve_kb"}}

User Question:
{question}
"""

    response = llm.invoke(prompt)
    content = clean_json_response(response.content)

    try:
        result = json.loads(content)
        route = result.get("route", "retrieve_kb")
        if route not in ["direct_answer", "retrieve_kb"]:
            route = "retrieve_kb"
    except Exception:
        # Fallback keyword heuristic
        lower_q = question.lower().strip()
        if lower_q in ["hi", "hello", "hey", "how are you", "who are you"]:
            route = "direct_answer"
        else:
            route = "retrieve_kb"

    state["route"] = route
    state["current_query"] = question
    state["retry_count"] = state.get("retry_count", 0)
    state["decision_trace"] = add_trace(state, f"Router Decision: {route} (Question: '{question}')")
    return state


# ==========================================
# 2. DIRECT ANSWER NODE
# ==========================================

def direct_answer(state: Dict[str, Any]) -> Dict[str, Any]:
    question = state["question"]
    llm = get_llm()

    prompt = f"""You are a helpful and professional AI assistant for Novatech/NovaRetail.
Respond directly and concisely to the user's conversational greeting or general knowledge question.

User Question: {question}
"""

    response = llm.invoke(prompt)
    state["answer"] = clean_text_response(response.content)
    state["source_used"] = "Direct LLM Response"
    state["sources"] = []
    state["decision_trace"] = add_trace(state, "Generated direct answer via LLM without document retrieval.")
    return state


# ==========================================
# 3. RETRIEVE PRIVATE KB NODE
# ==========================================

def retrieve_private_docs(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state.get("current_query", state["question"])
    documents = search_documents(query)

    state["private_docs"] = documents
    state["decision_trace"] = add_trace(
        state,
        f"Retrieved {len(documents)} document chunk(s) from Private Knowledge Base for query: '{query}'."
    )
    return state


# ==========================================
# 4. GRADE PRIVATE KB NODE
# ==========================================

def grade_private_docs(state: Dict[str, Any]) -> Dict[str, Any]:
    documents = state.get("private_docs", [])
    question = state["question"]

    if not documents:
        state["kb_grade"] = "weak"
        state["decision_trace"] = add_trace(state, "Private KB evaluation: No documents retrieved -> grade: weak")
        return state

    context = "\n\n".join([f"Chunk [{doc.get('chunk_id', i)}]:\n{doc.get('text', '')}" for i, doc in enumerate(documents)])
    llm = get_llm()

    prompt = f"""You are an objective evidence grader for the Novatech / NovaRetail knowledge base.

Determine whether the retrieved internal documentation contains sufficient, relevant facts to answer the user's question accurately.

User Question:
{question}

Retrieved Private Documentation:
{context}

Output ONLY a JSON object:
If the documents contain relevant information to answer the question:
{{"grade": "good"}}

If the documents are irrelevant or do NOT provide sufficient information to answer the question:
{{"grade": "weak"}}
"""

    response = llm.invoke(prompt)
    content = clean_json_response(response.content)

    try:
        result = json.loads(content)
        grade = result.get("grade", "weak").lower()
        if grade not in ["good", "weak"]:
            grade = "weak"
    except Exception:
        grade = "weak"

    state["kb_grade"] = grade
    state["decision_trace"] = add_trace(state, f"Private KB Evidence Grade: {grade}")
    return state


# ==========================================
# 5. GENERATE FROM PRIVATE KB NODE
# ==========================================

def generate_from_private_kb(state: Dict[str, Any]) -> Dict[str, Any]:
    question = state["question"]
    documents = state.get("private_docs", [])

    context = "\n\n".join([f"Source [{doc.get('source', 'internal_doc')}]:\n{doc.get('text', '')}" for doc in documents])
    llm = get_llm()

    prompt = f"""You are the internal IT and Knowledge Assistant for Novatech / NovaRetail.

Answer the employee's question accurately and professionally using ONLY the provided private company knowledge base excerpts.
Do not hallucinate facts outside the provided documentation.

Question:
{question}

Private Knowledge Excerpts:
{context}

Provide a clear, well-structured answer:
"""

    response = llm.invoke(prompt)
    state["answer"] = clean_text_response(response.content)
    state["source_used"] = "Private Knowledge Base (Novatech / NovaRetail)"

    # Extract distinct source document names
    sources = list({doc.get("source", "nova_it_handbook.txt") for doc in documents if doc.get("source")})
    state["sources"] = sources if sources else ["nova_it_handbook.txt"]
    state["decision_trace"] = add_trace(state, "Generated final response from Private Knowledge Base.")
    return state


# ==========================================
# 6. RETRIEVE WEB SEARCH NODE (TAVILY)
# ==========================================

def retrieve_web(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state.get("current_query", state["question"])
    results = search_web(query)

    state["web_results"] = results
    state["decision_trace"] = add_trace(
        state,
        f"Tavily Web Search executed for query: '{query}'. Retrieved {len(results)} result(s)."
    )
    return state


# ==========================================
# 7. GRADE WEB RESULTS NODE
# ==========================================

def grade_web_results(state: Dict[str, Any]) -> Dict[str, Any]:
    results = state.get("web_results", [])
    question = state["question"]

    if not results:
        state["web_grade"] = "weak"
        state["decision_trace"] = add_trace(state, "Web search evaluation: No web results found -> grade: weak")
        return state

    context = "\n\n".join([f"Result ({r.get('title', '')}) [URL: {r.get('url', '')}]:\n{r.get('content', '')}" for r in results])
    llm = get_llm()

    prompt = f"""You are an objective web search evaluator.

Determine whether the retrieved web search results contain adequate and relevant information to address the user's question.

User Question:
{question}

Web Search Results:
{context}

Output ONLY a JSON object:
If the search results contain sufficient facts to answer the question:
{{"grade": "good"}}

If the search results are irrelevant, empty, or unhelpful:
{{"grade": "weak"}}
"""

    response = llm.invoke(prompt)
    content = clean_json_response(response.content)

    try:
        result = json.loads(content)
        grade = result.get("grade", "weak").lower()
        if grade not in ["good", "weak"]:
            grade = "weak"
    except Exception:
        grade = "weak"

    state["web_grade"] = grade
    state["decision_trace"] = add_trace(state, f"Web Search Evidence Grade: {grade}")
    return state


# ==========================================
# 8. GENERATE FROM WEB SEARCH NODE
# ==========================================

def generate_from_web(state: Dict[str, Any]) -> Dict[str, Any]:
    question = state["question"]
    results = state.get("web_results", [])

    context = "\n\n".join([
        f"Title: {r.get('title', 'N/A')}\nURL: {r.get('url', '')}\nContent: {r.get('content', '')}"
        for r in results
    ])

    llm = get_llm()
    prompt = f"""You are an AI research assistant.
Answer the user's question clearly and accurately based on the provided web search evidence.
Include relevant context and cite facts properly.

Question:
{question}

Web Search Findings:
{context}

Synthesize a concise, factual answer:
"""

    response = llm.invoke(prompt)
    state["answer"] = clean_text_response(response.content)
    state["source_used"] = "Tavily Web Search"

    # Extract clean list of unique URLs
    urls = [r["url"] for r in results if r.get("url")]
    state["sources"] = list(dict.fromkeys(urls))
    state["decision_trace"] = add_trace(state, "Generated final response from Tavily Web Search evidence.")
    return state


# ==========================================
# 9. REWRITE QUERY NODE (FOR WEB SEARCH)
# ==========================================

def rewrite_query(state: Dict[str, Any]) -> Dict[str, Any]:
    question = state["question"]
    current_query = state.get("current_query", question)
    retry_count = state.get("retry_count", 0)

    llm = get_llm()
    prompt = f"""You are a search query optimizer.
The previous search query did not yield sufficient or specific search results on the web.
Rewrite the question into an optimized, concise web search query (3-7 words) targeting high-quality factual search engines.

Original Question: {question}
Previous Query: {current_query}

Return ONLY the rewritten search query string. Do not include quotes, preamble, or markdown.
"""

    response = llm.invoke(prompt)
    rewritten_query = clean_text_response(response.content).replace('"', '').replace("'", "")

    state["current_query"] = rewritten_query
    state["retry_count"] = retry_count + 1
    state["decision_trace"] = add_trace(
        state,
        f"Query rewritten (Attempt {state['retry_count']}/{MAX_RETRIES}): '{rewritten_query}'"
    )
    return state


# ==========================================
# 10. INSUFFICIENT EVIDENCE / FALLBACK NODE
# ==========================================

def insufficient_answer(state: Dict[str, Any]) -> Dict[str, Any]:
    question = state["question"]
    retry_count = state.get("retry_count", 0)

    state["answer"] = (
        f"I was unable to find sufficient or verified information to answer your question: '{question}'. "
        f"Judgement: The private Novatech knowledge base contains no matching records, "
        f"and external web search (including {retry_count} query refinement retry) did not yield conclusive evidence."
    )
    state["source_used"] = "Insufficient Evidence (Private KB & Web Exhausted)"
    state["sources"] = []
    state["decision_trace"] = add_trace(
        state,
        "Exhausted retrieval attempts (Private KB + Web Search with rewrite). Returning insufficient evidence verdict."
    )
    return state