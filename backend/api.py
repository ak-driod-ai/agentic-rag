import json
import uuid
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
    description="Production FastAPI endpoint with LangGraph real-time streaming, multi-turn conversational memory, and knowledge base routing.",
    version="1.1.0"
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
    thread_id: Optional[str] = Field(None, example="user_session_123")


class QueryResponse(BaseModel):
    thread_id: str
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

async def event_generator(question: str, thread_id: str = "default_session") -> AsyncGenerator[str, None]:
    """
    Streams LangGraph execution events step-by-step using Server-Sent Events (SSE).
    Preserves multi-turn state across sequential turns using the thread_id checkpointer.
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

    config = {"configurable": {"thread_id": thread_id}}
    final_state = dict(initial_state)

    try:
        # Initial start event
        yield f"event: start\ndata: {json.dumps({'message': 'Agent graph initialized', 'thread_id': thread_id, 'question': question})}\n\n"
        await asyncio.sleep(0.01)

        # Stream node-by-node updates from LangGraph
        async for output in agent.astream(initial_state, config=config, stream_mode="updates"):
            for node_name, node_update in output.items():
                final_state.update(node_update)

                trace_list = node_update.get("decision_trace", [])
                latest_trace = trace_list[-1] if trace_list else f"Executed node: {node_name}"

                payload = {
                    "thread_id": thread_id,
                    "node": node_name,
                    "trace": latest_trace,
                    "current_query": node_update.get("current_query"),
                    "route": node_update.get("route"),
                    "kb_grade": node_update.get("kb_grade"),
                    "web_grade": node_update.get("web_grade"),
                    "source_used": node_update.get("source_used")
                }

                yield f"event: step\ndata: {json.dumps(payload)}\n\n"
                await asyncio.sleep(0.02)

        # Final result event
        result_payload = {
            "thread_id": thread_id,
            "question": question,
            "answer": final_state.get("answer", "No answer generated."),
            "source_used": final_state.get("source_used", "N/A"),
            "sources": final_state.get("sources", []),
            "decision_trace": final_state.get("decision_trace", [])
        }
        yield f"event: result\ndata: {json.dumps(result_payload)}\n\n"
        yield "event: done\ndata: [DONE]\n\n"

    except Exception as e:
        error_payload = {"error": str(e), "thread_id": thread_id}
        yield f"event: error\ndata: {json.dumps(error_payload)}\n\n"


# ==========================================
# REST ENDPOINTS
# ==========================================

@app.get("/health", tags=["Health"])
def health_check():
    """Returns the service operational health status."""
    return {"status": "ok", "service": "Agentic RAG Assistant API", "memory_enabled": True}


@app.post("/api/query", response_model=QueryResponse, tags=["RAG Query"])
def query_agent(payload: QueryRequest):
    """
    Standard synchronous query endpoint with multi-turn memory.
    """
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    thread_id = payload.thread_id or str(uuid.uuid4())
    result = run_agent(payload.question, thread_id=thread_id)

    return QueryResponse(
        thread_id=thread_id,
        question=payload.question,
        answer=result.get("answer", "No answer generated."),
        source_used=result.get("source_used", "N/A"),
        sources=result.get("sources", []),
        decision_trace=result.get("decision_trace", [])
    )


@app.post("/api/stream", tags=["Streaming RAG"])
async def stream_agent_post(payload: QueryRequest):
    """
    Server-Sent Events (SSE) streaming endpoint via POST with multi-turn support.
    """
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    thread_id = payload.thread_id or str(uuid.uuid4())

    return StreamingResponse(
        event_generator(payload.question, thread_id=thread_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/api/stream", tags=["Streaming RAG"])
async def stream_agent_get(question: str, thread_id: Optional[str] = None):
    """
    Server-Sent Events (SSE) streaming endpoint via GET.
    """
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question query param is required.")

    sess_id = thread_id or str(uuid.uuid4())

    return StreamingResponse(
        event_generator(question, thread_id=sess_id),
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
# BUILT-IN MODERN WEB UI (MULTI-TURN ENABLED)
# ==========================================

@app.get("/", response_class=HTMLResponse, tags=["Web UI"])
def serve_ui():
    """Serves a sleek interactive chat interface with real-time SSE streaming and multi-turn memory."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agentic RAG Assistant | Multi-Turn Memory</title>
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
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
            --border: #374151;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

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

        .session-badge {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            background: rgba(255, 255, 255, 0.08);
            padding: 0.25rem 0.6rem;
            border-radius: 6px;
            color: #93c5fd;
            border: 1px solid var(--border);
        }

        .header-actions {
            display: flex;
            gap: 0.75rem;
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
            text-decoration: none;
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

        @keyframes spin { to { transform: rotate(360deg); } }

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
        }

        .btn-send:hover:not(:disabled) {
            background: #2563eb;
            transform: translateY(-1px);
        }

        .btn-send:disabled { opacity: 0.5; cursor: not-allowed; }

        .quick-queries {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            align-items: center;
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
            <div class="brand-title">Novatech Assistant</div>
            <span class="session-badge" id="sessionBadge">Thread: init</span>
        </div>
        <div class="header-actions">
            <button class="btn-secondary" onclick="startNewChat()">🔄 New Chat</button>
            <button class="btn-secondary" onclick="reindexKnowledgeBase()">⚡ Re-index KB</button>
            <a href="/docs" target="_blank" class="btn-secondary">📘 OpenAPI Docs</a>
        </div>
    </header>

    <main>
        <div class="quick-queries">
            <span style="font-size: 0.8rem; color: var(--text-muted);">Try multi-turn flow:</span>
            <button class="quick-btn" onclick="setQuestion('How do I connect to NovaRetail VPN?')">1️⃣ Connect to VPN</button>
            <button class="quick-btn" onclick="setQuestion('What if that application fails?')">2️⃣ Follow-up: What if that fails?</button>
            <button class="quick-btn" onclick="setQuestion('What are the password requirements?')">3️⃣ Password Policy</button>
        </div>

        <div class="chat-container" id="chatContainer">
            <div class="message-card assistant-message">
                <div class="message-header">
                    <span>AGENT READY (MULTI-TURN MEMORY ACTIVE)</span>
                    <span class="source-tag">LangGraph MemorySaver</span>
                </div>
                <div class="message-body">Hello! I am your enterprise Agentic Assistant with persistent conversational memory. You can ask questions and follow up with natural pronouns like <i>"What if that app fails?"</i> or <i>"Tell me more about it."</i></div>
            </div>
        </div>

        <div class="input-bar">
            <input type="text" id="queryInput" placeholder="Ask a question or follow-up in context..." onkeydown="if(event.key==='Enter') sendStreamQuery()" />
            <button class="btn-send" id="sendBtn" onclick="sendStreamQuery()">Send ➔</button>
        </div>
    </main>

    <script>
        let currentThreadId = "session_" + Math.random().toString(36).substring(2, 9);
        document.getElementById('sessionBadge').innerText = "Thread: " + currentThreadId;

        function startNewChat() {
            currentThreadId = "session_" + Math.random().toString(36).substring(2, 9);
            document.getElementById('sessionBadge').innerText = "Thread: " + currentThreadId;
            const chat = document.getElementById('chatContainer');
            chat.innerHTML = `
                <div class="message-card assistant-message">
                    <div class="message-header">
                        <span>NEW SESSION STARTED</span>
                        <span class="source-tag">Memory Reset</span>
                    </div>
                    <div class="message-body">New session initialized (${currentThreadId}). How can I assist you today?</div>
                </div>
            `;
        }

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

            // 1. User Message Card
            const userCard = document.createElement('div');
            userCard.className = 'message-card user-message';
            userCard.innerHTML = `
                <div class="message-header"><span>YOU</span></div>
                <div class="message-body">${escapeHtml(question)}</div>
            `;
            chat.appendChild(userCard);

            // 2. Assistant Message Card
            const assistantCard = document.createElement('div');
            assistantCard.className = 'message-card assistant-message';
            assistantCard.innerHTML = `
                <div class="message-header">
                    <span>AGENTIC GRAPH EXECUTION</span>
                    <span class="source-tag" id="sourceBadge">Contextualizing...</span>
                </div>
                <div class="steps-container" id="stepsBox">
                    <div class="step-item"><div class="step-spinner"></div> Resolving context & memory...</div>
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
                    body: JSON.stringify({ question, thread_id: currentThreadId })
                });

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\\n\\n');
                    buffer = lines.pop();

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
