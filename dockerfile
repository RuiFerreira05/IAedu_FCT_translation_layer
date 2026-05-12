# Use the official uv image with Python 3.12 (adjust version as needed)
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

# Optimization variables for uv inside Docker
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# 1. Install dependencies first (for caching)
COPY uv.lock pyproject.toml ./
RUN uv sync --frozen --no-install-project --no-dev

# 2. Copy the rest of your source code
COPY . .
RUN uv sync --frozen --no-dev

# 3. Put the virtual environment on the path
ENV PATH="/app/.venv/bin:$PATH"

# 4. Define your startup command (example using FastAPI or a standard script)
# CMD ["fastapi", "run", "src/main.py", "--port", "8000", "--host", "0.0.0.0"]
CMD ["python", "main.py"]
