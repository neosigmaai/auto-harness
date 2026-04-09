FROM python:3.12-slim

WORKDIR /app

# Install git (needed for uv to fetch tau2 from git), uv, and Docker CLI
ENV DOCKER_CLI_VERSION=27.5.1
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    && arch="$(dpkg --print-architecture)" \
    && case "$arch" in \
        amd64) docker_arch="x86_64" ;; \
        arm64) docker_arch="aarch64" ;; \
        *) echo "Unsupported architecture: $arch" >&2; exit 1 ;; \
    esac \
    && curl -fsSLo /tmp/docker.tgz "https://download.docker.com/linux/static/stable/${docker_arch}/docker-${DOCKER_CLI_VERSION}.tgz" \
    && curl -fsSLo /tmp/docker.tgz.sha256 "https://download.docker.com/linux/static/stable/${docker_arch}/docker-${DOCKER_CLI_VERSION}.tgz.sha256" \
    && sha256sum -c /tmp/docker.tgz.sha256 \
    && tar -xzf /tmp/docker.tgz -C /tmp \
    && install -m 0755 /tmp/docker/docker /usr/local/bin/docker \
    && rm -rf /tmp/docker /tmp/docker.tgz /tmp/docker.tgz.sha256 /var/lib/apt/lists/*
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