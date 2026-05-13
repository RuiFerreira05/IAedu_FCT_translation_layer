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


def derive_thread_id(messages: list) -> str:
    """Stable thread ID per chat (based on first message). Same logic as before."""
    if messages:
        first_message_content = messages[0].get("content", "default_seed")
        return hashlib.md5(first_message_content.encode()).hexdigest()
    return str(uuid.uuid4())


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
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

    thread_id = derive_thread_id(messages)

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
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                try:
                    event_data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event_data.get("type") == "token":
                    full_text += event_data.get("content", "")

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
        # Prime client with initial role chunk
        yield make_chunk({"role": "assistant", "content": ""})

        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST", target_endpoint, files=form_data, headers=headers
            ) as response:
                buffer = ""
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
                                yield make_chunk({"content": text_chunk})

                # Flush any leftover buffered line
                if buffer.strip():
                    try:
                        event_data = json.loads(buffer.strip())
                        if event_data.get("type") == "token":
                            text_chunk = event_data.get("content", "")
                            if text_chunk:
                                yield make_chunk({"content": text_chunk})
                    except json.JSONDecodeError:
                        pass

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
