import json
import asyncio
from typing import AsyncGenerator, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse

from .main import run_agent, get_agent
from .ingest import ingest


app = FastAPI(
    title="Agentic RAG API",
    description="Production FastAPI endpoint with LangGraph real-time streaming, decision tracing, and knowledge base routing.",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# PYDANTIC SCHEMAS
# ==========================================

class QueryRequest(BaseModel):
    question: str = Field(..., example="How do I connect to NovaRetail VPN?")


class QueryResponse(BaseModel):
    question: str
    answer: str
    source_used: str
    sources: list
    decision_trace: list


class IngestResponse(BaseModel):
    status: str
    message: str


# ==========================================
# STREAMING GENERATOR
# ==========================================

async def event_generator(question: str) -> AsyncGenerator[str, None]:
    """
    Streams LangGraph execution events step-by-step using Server-Sent Events (SSE).
    Emits node completion events and the final synthesized response.
    """
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

    final_state = dict(initial_state)

    try:
        # Initial event
        yield f"event: start\ndata: {json.dumps({'message': 'Agent graph initialized', 'question': question})}\n\n"
        await asyncio.sleep(0.01)

        # Stream node-by-node updates from LangGraph
        async for output in agent.astream(initial_state, stream_mode="updates"):
            for node_name, node_update in output.items():
                # Merge updates into final state
                final_state.update(node_update)

                trace_list = node_update.get("decision_trace", [])
                latest_trace = trace_list[-1] if trace_list else f"Executed node: {node_name}"

                payload = {
                    "node": node_name,
                    "trace": latest_trace,
                    "route": node_update.get("route"),
                    "kb_grade": node_update.get("kb_grade"),
                    "web_grade": node_update.get("web_grade"),
                    "source_used": node_update.get("source_used")
                }

                yield f"event: step\ndata: {json.dumps(payload)}\n\n"
                await asyncio.sleep(0.02)

        # Final result event
        result_payload = {
            "question": question,
            "answer": final_state.get("answer", "No answer generated."),
            "source_used": final_state.get("source_used", "N/A"),
            "sources": final_state.get("sources", []),
            "decision_trace": final_state.get("decision_trace", [])
        }
        yield f"event: result\ndata: {json.dumps(result_payload)}\n\n"
        yield "event: done\ndata: [DONE]\n\n"

    except Exception as e:
        error_payload = {"error": str(e)}
        yield f"event: error\ndata: {json.dumps(error_payload)}\n\n"


# ==========================================
# REST ENDPOINTS
# ==========================================

@app.get("/health", tags=["Health"])
def health_check():
    """Returns the service operational health status."""
    return {"status": "ok", "service": "Agentic RAG Assistant API"}


@app.post("/api/query", response_model=QueryResponse, tags=["RAG Query"])
def query_agent(payload: QueryRequest):
    """
    Standard synchronous query endpoint.
    Executes the full Agentic RAG graph and returns the complete JSON response.
    """
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = run_agent(payload.question)
    return QueryResponse(
        question=payload.question,
        answer=result.get("answer", "No answer generated."),
        source_used=result.get("source_used", "N/A"),
        sources=result.get("sources", []),
        decision_trace=result.get("decision_trace", [])
    )


@app.post("/api/stream", tags=["Streaming RAG"])
async def stream_agent_post(payload: QueryRequest):
    """
    Server-Sent Events (SSE) streaming endpoint via POST.
    Streams intermediate node decisions, retrieval progress, and final answer tokens.
    """
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    return StreamingResponse(
        event_generator(payload.question),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/api/stream", tags=["Streaming RAG"])
async def stream_agent_get(question: str):
    """
    Server-Sent Events (SSE) streaming endpoint via GET (browser EventSource friendly).
    """
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question query param is required.")

    return StreamingResponse(
        event_generator(question),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/api/ingest", response_model=IngestResponse, tags=["Knowledge Base"])
def trigger_ingest():
    """
    Triggers re-indexing of documents in the knowledge_base/ directory into Qdrant.
    """
    try:
        ingest()
        return IngestResponse(
            status="success",
            message="Knowledge base documents successfully processed and stored in Qdrant."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


# ==========================================
# BUILT-IN MODERN WEB UI
# ==========================================

@app.get("/", response_class=HTMLResponse, tags=["Web UI"])
def serve_ui():
    """Serves a sleek, modern interactive chat interface with real-time SSE streaming graph visualizer."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agentic RAG Assistant | Live Streaming</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-base: #0b0f19;
            --bg-surface: #111827;
            --bg-card: #1f2937;
            --bg-input: #1e293b;
            --primary: #3b82f6;
            --primary-glow: rgba(59, 130, 246, 0.35);
            --accent: #10b981;
            --accent-warm: #f59e0b;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
            --border: #374151;
            --border-highlight: #4b5563;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: var(--bg-base);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        header {
            background: rgba(17, 24, 39, 0.85);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--border);
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .brand-badge {
            background: linear-gradient(135deg, #3b82f6, #8b5cf6);
            padding: 0.4rem 0.75rem;
            border-radius: 8px;
            font-weight: 700;
            font-size: 0.85rem;
            letter-spacing: 0.5px;
        }

        .brand-title {
            font-size: 1.15rem;
            font-weight: 700;
            color: #ffffff;
        }

        .header-actions {
            display: flex;
            gap: 1rem;
            align-items: center;
        }

        .btn-secondary {
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: var(--text-main);
            padding: 0.5rem 1rem;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.85rem;
            font-weight: 500;
            transition: all 0.2s;
        }

        .btn-secondary:hover {
            border-color: var(--primary);
            background: #283548;
        }

        main {
            flex: 1;
            max-width: 1000px;
            width: 100%;
            margin: 0 auto;
            padding: 2rem 1.5rem;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        .chat-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            min-height: 450px;
        }

        .message-card {
            background: var(--bg-surface);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 1.5rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
            animation: fadeIn 0.3s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .user-message {
            border-left: 4px solid var(--primary);
            background: #141f33;
        }

        .assistant-message {
            border-left: 4px solid var(--accent);
        }

        .message-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.75rem;
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-muted);
        }

        .source-tag {
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.3);
            padding: 0.2rem 0.6rem;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
        }

        .steps-container {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            margin: 1rem 0;
            padding: 0.75rem 1rem;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 10px;
            border: 1px dashed var(--border);
        }

        .step-item {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            color: #93c5fd;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .step-spinner {
            width: 12px;
            height: 12px;
            border: 2px solid #3b82f6;
            border-top-color: transparent;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .message-body {
            line-height: 1.7;
            font-size: 0.95rem;
            white-space: pre-wrap;
            color: #e5e7eb;
        }

        .citations-box {
            margin-top: 1rem;
            padding-top: 0.75rem;
            border-top: 1px solid var(--border);
            font-size: 0.85rem;
        }

        .citations-title {
            font-weight: 600;
            color: var(--text-muted);
            margin-bottom: 0.4rem;
        }

        .citation-item {
            display: inline-block;
            background: var(--bg-card);
            padding: 0.25rem 0.6rem;
            border-radius: 6px;
            margin-right: 0.5rem;
            margin-bottom: 0.5rem;
            font-size: 0.8rem;
            color: #60a5fa;
            text-decoration: none;
            border: 1px solid var(--border);
        }

        .citation-item:hover {
            border-color: var(--primary);
            text-decoration: underline;
        }

        .input-bar {
            background: var(--bg-surface);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 0.75rem;
            display: flex;
            gap: 0.75rem;
            position: sticky;
            bottom: 1.5rem;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);
        }

        .input-bar:focus-within {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px var(--primary-glow);
        }

        input[type="text"] {
            flex: 1;
            background: transparent;
            border: none;
            outline: none;
            color: var(--text-main);
            font-size: 1rem;
            font-family: inherit;
            padding: 0.5rem 0.75rem;
        }

        .btn-send {
            background: var(--primary);
            color: #ffffff;
            border: none;
            border-radius: 10px;
            padding: 0.6rem 1.5rem;
            font-weight: 600;
            font-size: 0.95rem;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .btn-send:hover:not(:disabled) {
            background: #2563eb;
            transform: translateY(-1px);
        }

        .btn-send:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .quick-queries {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }

        .quick-btn {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border);
            color: var(--text-muted);
            padding: 0.35rem 0.75rem;
            border-radius: 20px;
            font-size: 0.8rem;
            cursor: pointer;
            transition: all 0.2s;
        }

        .quick-btn:hover {
            border-color: var(--primary);
            color: var(--text-main);
            background: rgba(59, 130, 246, 0.1);
        }
    </style>
</head>
<body>

    <header>
        <div class="brand">
            <div class="brand-badge">AGENTIC RAG</div>
            <div class="brand-title">Novatech Assistant API</div>
        </div>
        <div class="header-actions">
            <button class="btn-secondary" onclick="reindexKnowledgeBase()">⚡ Re-index KB</button>
            <a href="/docs" target="_blank" class="btn-secondary">📘 OpenAPI Docs</a>
        </div>
    </header>

    <main>
        <div class="quick-queries">
            <span style="font-size: 0.8rem; color: var(--text-muted); align-self: center;">Try sample:</span>
            <button class="quick-btn" onclick="setQuestion('How do I connect to NovaRetail VPN?')">🔒 Connect to VPN</button>
            <button class="quick-btn" onclick="setQuestion('What is the password complexity requirement?')">🔑 Password Policy</button>
            <button class="quick-btn" onclick="setQuestion('What are the latest updates about Python 3.13 features?')">🌐 Python 3.13 (Web)</button>
            <button class="quick-btn" onclick="setQuestion('Hello, who are you?')">👋 Hello</button>
        </div>

        <div class="chat-container" id="chatContainer">
            <div class="message-card assistant-message">
                <div class="message-header">
                    <span>AGENT READY</span>
                    <span class="source-tag">LangGraph + Groq + Qdrant + Tavily</span>
                </div>
                <div class="message-body">Hello! I am your enterprise Agentic RAG assistant. Ask me anything about NovaRetail/Novatech internal IT policies, or general factual questions requiring real-time web research.</div>
            </div>
        </div>

        <div class="input-bar">
            <input type="text" id="queryInput" placeholder="Ask an enterprise IT or web question..." onkeydown="if(event.key==='Enter') sendStreamQuery()" />
            <button class="btn-send" id="sendBtn" onclick="sendStreamQuery()">Send ➔</button>
        </div>
    </main>

    <script>
        function setQuestion(q) {
            document.getElementById('queryInput').value = q;
            sendStreamQuery();
        }

        async function reindexKnowledgeBase() {
            const btn = event.target;
            btn.innerText = "Indexing...";
            btn.disabled = true;
            try {
                const res = await fetch("/api/ingest", { method: "POST" });
                const data = await res.json();
                alert(data.message || "Ingestion complete!");
            } catch (err) {
                alert("Ingestion error: " + err);
            } finally {
                btn.innerText = "⚡ Re-index KB";
                btn.disabled = false;
            }
        }

        async function sendStreamQuery() {
            const input = document.getElementById('queryInput');
            const sendBtn = document.getElementById('sendBtn');
            const chat = document.getElementById('chatContainer');
            const question = input.value.trim();

            if (!question) return;

            input.value = '';
            input.disabled = true;
            sendBtn.disabled = true;

            // 1. Append User Card
            const userCard = document.createElement('div');
            userCard.className = 'message-card user-message';
            userCard.innerHTML = `
                <div class="message-header"><span>YOU</span></div>
                <div class="message-body">${escapeHtml(question)}</div>
            `;
            chat.appendChild(userCard);

            // 2. Append Assistant Streaming Card
            const assistantCard = document.createElement('div');
            assistantCard.className = 'message-card assistant-message';
            assistantCard.innerHTML = `
                <div class="message-header">
                    <span>AGENTIC GRAPH EXECUTION</span>
                    <span class="source-tag" id="sourceBadge">Executing...</span>
                </div>
                <div class="steps-container" id="stepsBox">
                    <div class="step-item"><div class="step-spinner"></div> Initializing graph pipeline...</div>
                </div>
                <div class="message-body" id="answerBox">Thinking...</div>
                <div class="citations-box" id="citationsBox" style="display:none;">
                    <div class="citations-title">Sources & References:</div>
                    <div id="citationLinks"></div>
                </div>
            `;
            chat.appendChild(assistantCard);
            assistantCard.scrollIntoView({ behavior: 'smooth' });

            const stepsBox = assistantCard.querySelector('#stepsBox');
            const answerBox = assistantCard.querySelector('#answerBox');
            const sourceBadge = assistantCard.querySelector('#sourceBadge');
            const citationsBox = assistantCard.querySelector('#citationsBox');
            const citationLinks = assistantCard.querySelector('#citationLinks');

            try {
                const response = await fetch('/api/stream', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question })
                });

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\\n\\n');
                    buffer = lines.pop(); // Keep last partial line in buffer

                    for (const block of lines) {
                        if (!block.trim()) continue;

                        const eventMatch = block.match(/^event: (.*)$/m);
                        const dataMatch = block.match(/^data: (.*)$/m);

                        const eventName = eventMatch ? eventMatch[1] : 'message';
                        const dataText = dataMatch ? dataMatch[1] : '';

                        if (eventName === 'step') {
                            const step = JSON.parse(dataText);
                            const stepDiv = document.createElement('div');
                            stepDiv.className = 'step-item';
                            stepDiv.innerHTML = `<span>➔</span> <span>${escapeHtml(step.trace)}</span>`;
                            stepsBox.appendChild(stepDiv);
                        } else if (eventName === 'result') {
                            const result = JSON.parse(dataText);
                            answerBox.innerText = result.answer;
                            sourceBadge.innerText = result.source_used;

                            if (result.sources && result.sources.length > 0) {
                                citationsBox.style.display = 'block';
                                citationLinks.innerHTML = result.sources.map(s => {
                                    if (s.startsWith('http')) {
                                        return `<a href="${escapeHtml(s)}" target="_blank" class="citation-item">${escapeHtml(s)}</a>`;
                                    }
                                    return `<span class="citation-item">📄 ${escapeHtml(s)}</span>`;
                                }).join('');
                            }
                        }
                    }
                }
            } catch (err) {
                answerBox.innerText = 'Execution error: ' + err;
                sourceBadge.innerText = 'Error';
            } finally {
                input.disabled = false;
                sendBtn.disabled = false;
                input.focus();
                // Remove spinner
                const spinner = stepsBox.querySelector('.step-spinner');
                if (spinner) spinner.parentElement.remove();
            }
        }

        function escapeHtml(str) {
            if (!str) return '';
            return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
        }
    </script>
</body>
</html>
"""
