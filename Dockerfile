FROM python:3.12-slim

WORKDIR /app

# Install git (needed for uv to fetch tau2 from git), Docker **client** (mini-swe-agent /
# swebench subprocess `docker`; daemon comes from mounted /var/run/docker.sock), and uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates curl \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc \
    && chmod a+r /etc/apt/keyrings/docker.asc \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian $(. /etc/os-release && echo \"$VERSION_CODENAME\") stable" > /etc/apt/sources.list.d/docker.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends docker-ce-cli \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv

# Copy manifest + only the installable package so ``uv sync`` can resolve the local project
COPY pyproject.toml ./
COPY agent ./agent

# Install all dependencies (including tau2 from git) + SWE-Bench harness (optional extra ``swe``)
RUN uv sync --no-dev --extra swe

# Activate venv so plain `python` resolves to the venv interpreter
ENV PATH="/app/.venv/bin:$PATH"

# Copy project files and re-sync so the editable install matches the full tree (deps stay resolved)
COPY . .
RUN uv sync --no-dev --extra swe \
    && python -c "import yaml; import sys; print('PyYAML OK', file=sys.stderr)"

# workspace/ is mounted at runtime — create as fallback
RUN mkdir -p workspace

CMD ["bash"]
