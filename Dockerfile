FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy project files
COPY pyproject.toml .
COPY ruleprobe/ ruleprobe/
COPY examples/ examples/

# Install dependencies
RUN uv pip install --system -e ".[api]"

# Expose FastAPI port
EXPOSE 8000

# Default: run the API server
CMD ["uvicorn", "ruleprobe.api:app", "--host", "0.0.0.0", "--port", "8000"]
