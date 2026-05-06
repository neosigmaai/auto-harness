FROM python:3.12-slim

WORKDIR /app

# Install git (needed for uv to fetch tau2 from git), Docker CLI for AgentBench,
# and uv. AgentBench OS images are built on the host and consumed via Docker.
RUN apt-get update && apt-get install -y --no-install-recommends git docker.io \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv

# Copy dependency manifest first for better layer caching
COPY pyproject.toml ./

# Install all dependencies (including tau2 from git) into a venv
RUN uv sync --no-dev

# Activate venv so plain `python` resolves to the venv interpreter
ENV PATH="/app/.venv/bin:$PATH"

# Copy project files
COPY . .

# workspace/ is mounted at runtime — create as fallback
RUN mkdir -p workspace

CMD ["bash"]
