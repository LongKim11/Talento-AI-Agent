# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install uv for faster dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

# Copy dependency files
COPY pyproject.toml ./

# Create virtual environment and install dependencies
RUN uv venv --python 3.11 && \
    uv pip install --no-cache .

# Copy project files
COPY . .

# Expose port for LangGraph server
EXPOSE 2024

# Run LangGraph development server
CMD ["uv", "run", "langgraph", "dev", "--host", "0.0.0.0", "--port", "2024"]
