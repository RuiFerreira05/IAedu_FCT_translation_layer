import hashlib
import json
import os
import time
import uuid

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

load_dotenv()

# Target API Configuration
TARGET_ENDPOINT = os.getenv("TARGET_ENDPOINT", "")
API_KEY = os.getenv("API_KEY", "")
CHANNEL_ID = os.getenv("CHANNEL_ID", "")


app = FastAPI(title="OpenAI API Wrapper for IAEDU Agent")

# 1. CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 2. Mandatory Models Endpoint for Open WebUI
@app.get("/v1/models")
async def get_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "iaedu-agent",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "iaedu",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    is_stream = body.get("stream", False)
    chat_id = f"chatcmpl-{int(time.time())}"

    user_message = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_message = msg.get("content", "")
            break
    if not user_message:
        user_message = "Hello"

    if messages:
        first_message_content = messages[0].get("content", "default_seed")
        thread_id = hashlib.md5(first_message_content.encode()).hexdigest()
    else:
        thread_id = str(uuid.uuid4())

    form_data = {
        "channel_id": (None, CHANNEL_ID),
        "thread_id": (None, thread_id),
        "user_info": (None, "{}"),
        "message": (None, user_message),
    }
    headers = {"x-api-key": API_KEY}

    # Non-streaming path (unchanged from yours, omitted here for brevity)
    if not is_stream:
        # ... your existing non-streaming code ...
        pass

    def make_chunk(delta: dict, finish_reason=None) -> str:
        payload = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "iaedu-agent",
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
        # Prime the client with an initial role chunk (this is what real OpenAI does)
        yield make_chunk({"role": "assistant", "content": ""})

        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST", TARGET_ENDPOINT, files=form_data, headers=headers
            ) as response:
                buffer = ""
                # Use aiter_bytes instead of aiter_lines to avoid line-buffering
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
            "Content-Encoding": "none",  # prevents gzip middleware from buffering
        },
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
