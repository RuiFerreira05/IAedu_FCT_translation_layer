import hashlib
import json
import os
import time
import uuid

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

load_dotenv()

PORT = int(os.getenv("PORT", "8000"))

# ---------------------------------------------------------------------------
# Agent registry
# ---------------------------------------------------------------------------
# Each entry maps an OpenAI-style model ID to the upstream IAEDU agent config.
# Values come from environment variables so secrets stay out of the codebase.
# Add new agents by adding entries here and setting the matching env vars.

AGENTS: dict[str, dict[str, str]] = {
    "claude-opus-4.7": {
        "endpoint": os.getenv("CLAUDE_ENDPOINT", ""),
        "api_key": os.getenv("CLAUDE_API_KEY", ""),
        "channel_id": os.getenv("CLAUDE_CHANNEL_ID", ""),
    },
    "gpt-5.5": {
        "endpoint": os.getenv("GPT_ENDPOINT", ""),
        "api_key": os.getenv("GPT_API_KEY", ""),
        "channel_id": os.getenv("GPT_CHANNEL_ID", ""),
    },
}

# Default model when the client doesn't specify one (or sends an unknown name).
DEFAULT_MODEL = "claude-opus-4.7"


def get_agent_config(model_id: str) -> tuple[str, dict[str, str]]:
    """Resolve a requested model ID to a configured agent. Falls back to default."""
    if model_id in AGENTS and AGENTS[model_id]["endpoint"]:
        return model_id, AGENTS[model_id]
    # Fallback to default if requested model isn't configured
    if AGENTS.get(DEFAULT_MODEL, {}).get("endpoint"):
        return DEFAULT_MODEL, AGENTS[DEFAULT_MODEL]
    raise HTTPException(
        status_code=503,
        detail=f"No configured agent available for model '{model_id}'.",
    )


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="OpenAI API Wrapper for IAEDU Agents")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/v1/models")
async def get_models():
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": now,
                "owned_by": "iaedu",
            }
            for model_id, cfg in AGENTS.items()
            if cfg["endpoint"]  # only advertise agents that are actually configured
        ],
    }


import sqlite3
import hashlib
import json

DB_PATH = "/app/data/sessions.db"

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            history_hash TEXT PRIMARY KEY,
            thread_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

# Initialize database at startup
init_db()

def hash_history(messages: list) -> str:
    # Serialize to JSON with sorted keys for consistent hashing
    serialized = json.dumps(messages, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

def get_session_thread_id(history_hash: str) -> str | None:
    try:
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute("SELECT thread_id FROM sessions WHERE history_hash = ?", (history_hash,)).fetchone()
        conn.close()
        return row[0] if row else None
    except Exception as e:
        print(f"DB Error getting session: {e}", flush=True)
        return None

def save_session_thread_id(history_hash: str, thread_id: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("INSERT OR REPLACE INTO sessions (history_hash, thread_id) VALUES (?, ?)", (history_hash, thread_id))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB Error saving session: {e}", flush=True)

def derive_thread_id_stateful(messages: list) -> tuple[str, list]:
    """
    Statefully derive unique, stable thread ID.
    Returns: (thread_id, history_list_of_current_turn)
    """
    if not messages:
        tid = f"thread_{uuid.uuid4().hex}"
        return tid, []

    # History is all messages except the latest user message
    last_user_idx = -1
    for idx, msg in enumerate(reversed(messages)):
        if msg.get("role") == "user":
            last_user_idx = len(messages) - 1 - idx
            break

    if last_user_idx == -1:
        tid = f"thread_{uuid.uuid4().hex}"
        return tid, []

    history = messages[:last_user_idx]
    h_hash = hash_history(history)

    tid = get_session_thread_id(h_hash)
    if tid:
        print(f"Session found for hash {h_hash}: {tid}", flush=True)
        return tid, history

    has_prior_user = any(m.get("role") == "user" for m in history)
    if not has_prior_user:
        tid = f"thread_{uuid.uuid4().hex}"
        print(f"New chat detected. Generated new thread: {tid}", flush=True)
        # Save current history key mapping for immediate use (though we will update it at the end of response)
        save_session_thread_id(h_hash, tid)
        return tid, history

    # Fallback to stable hash of first user message
    first_user_content = ""
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if "### Task:" in content or "<chat_history>" in content:
                continue
            first_user_content = content
            break

    if first_user_content:
        tid = f"fallback_{hashlib.md5(first_user_content.encode('utf-8')).hexdigest()}"
        print(f"Session not found in DB. Falling back to first user message hash: {tid}", flush=True)
        save_session_thread_id(h_hash, tid)
        return tid, history

    tid = f"thread_{uuid.uuid4().hex}"
    print(f"Fallback generated random thread: {tid}", flush=True)
    save_session_thread_id(h_hash, tid)
    return tid, history


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    print(f"Chat completions request headers: {dict(request.headers)}", flush=True)
    print(f"Chat completions request body: {json.dumps(body)}", flush=True)
    messages = body.get("messages", [])
    is_stream = body.get("stream", False)
    requested_model = body.get("model", DEFAULT_MODEL)

    # Resolve which upstream agent to use
    resolved_model, agent_cfg = get_agent_config(requested_model)
    target_endpoint = agent_cfg["endpoint"]
    api_key = agent_cfg["api_key"]
    channel_id = agent_cfg["channel_id"]

    chat_id = f"chatcmpl-{int(time.time())}"

    # Extract latest user message
    user_message = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_message = msg.get("content", "")
            break
    if not user_message:
        user_message = "Hello"

    thread_id, history = derive_thread_id_stateful(messages)

    form_data = {
        "channel_id": (None, channel_id),
        "thread_id": (None, thread_id),
        "user_info": (None, "{}"),
        "message": (None, user_message),
    }
    headers = {"x-api-key": api_key}

    # ---------- Non-streaming branch ----------
    if not is_stream:
        full_text = ""
        async with httpx.AsyncClient(timeout=None) as client:
            req = client.build_request(
                "POST", target_endpoint, files=form_data, headers=headers
            )
            response = await client.send(req, stream=True)
            try:
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        event_data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event_data.get("type") == "token":
                        full_text += event_data.get("content", "")
            except (httpx.HTTPError, httpx.RemoteProtocolError) as e:
                print(f"Non-streaming error caught: {type(e).__name__}: {e}", flush=True)

        if full_text:
            H_next = history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": full_text}]
            next_hash = hash_history(H_next)
            save_session_thread_id(next_hash, thread_id)
            print(f"Saved session mapping for next turn (non-streaming): {next_hash} -> {thread_id}", flush=True)

        return JSONResponse(
            content={
                "id": chat_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": resolved_model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": full_text},
                        "finish_reason": "stop",
                    }
                ],
            }
        )

    # ---------- Streaming branch ----------
    def make_chunk(delta: dict, finish_reason=None) -> str:
        payload = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": resolved_model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
        }
        return f"data: {json.dumps(payload)}\n\n"

    async def stream_generator():
        print(f"Starting stream for model: {resolved_model}, endpoint: {target_endpoint}", flush=True)
        # Prime client with initial role chunk
        yield make_chunk({"role": "assistant", "content": ""})
        full_text = ""

        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST", target_endpoint, files=form_data, headers=headers
            ) as response:
                print(f"Upstream response status: {response.status_code}", flush=True)
                if response.status_code != 200:
                    # Let's read the error content
                    error_content = await response.aread()
                    print(f"Upstream error content: {error_content.decode('utf-8', errors='replace')}", flush=True)
                buffer = ""
                try:
                    async for raw in response.aiter_bytes():
                        if not raw:
                            continue
                        buffer += raw.decode("utf-8", errors="replace")
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                event_data = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            if event_data.get("type") == "token":
                                text_chunk = event_data.get("content", "")
                                if text_chunk:
                                    full_text += text_chunk
                                    yield make_chunk({"content": text_chunk})

                    # Flush any leftover buffered line
                    if buffer.strip():
                        try:
                            event_data = json.loads(buffer.strip())
                            if event_data.get("type") == "token":
                                text_chunk = event_data.get("content", "")
                                if text_chunk:
                                    full_text += text_chunk
                                    yield make_chunk({"content": text_chunk})
                        except json.JSONDecodeError:
                            pass
                except (httpx.HTTPError, httpx.RemoteProtocolError) as e:
                    print(f"Streaming error caught: {type(e).__name__}: {e}", flush=True)

        if full_text:
            H_next = history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": full_text}]
            next_hash = hash_history(H_next)
            save_session_thread_id(next_hash, thread_id)
            print(f"Saved session mapping for next turn (streaming): {next_hash} -> {thread_id}", flush=True)

        yield make_chunk({}, finish_reason="stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Content-Encoding": "none",
        },
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
