# ─────────────────────────────────────────────────────────────────────────────
# Multi-stage, multi-arch Dockerfile
# Supports: linux/arm64 (Oracle Cloud, Apple M-series)
#           linux/amd64 (Intel/AMD servers)
#
# Why multi-stage?
#   Stage 1 (builder) has compilers + build tools → needed to install packages
#   Stage 2 (runtime) has only the finished app   → nothing a customer doesn't need
#   Result: image shrinks from ~1.4 GB → ~600 MB, deploys 2× faster
#
# Build single-arch (local test):
#   docker build -t nyc-airbnb-api .
#
# Build + push multi-arch (CI / production):
#   docker buildx build --platform linux/arm64,linux/amd64 \
#     -t <registry>/nyc-airbnb-api:latest --push .
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: builder ─────────────────────────────────────────────────────────
# TARGETPLATFORM = the machine that will RUN the image (Oracle ARM node).
# Python wheels with C extensions (numpy, onnxruntime, xgboost deps, ...) are
# compiled binaries — they are NOT cross-compilable like Go. Building this
# stage under $BUILDPLATFORM (your amd64 Mac) produces amd64 .so files that
# crash with "exec format error" once copied into an arm64 runtime image.
# Buildx runs this stage under QEMU emulation for TARGETPLATFORM instead, so
# pip installs the correct native wheels for wherever the image will run.
FROM --platform=$TARGETPLATFORM python:3.11-slim AS builder

# Build tools needed to compile Python packages from source on ARM.
# We only need these in the builder — they never reach the runtime image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Create an isolated virtual environment.
# Why? So we can copy the entire /venv folder into the runtime stage cleanly,
# without pulling in Python's system-wide packages.
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Install deps into the venv.
# Layer is cached until requirements.txt changes — fast rebuilds.
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir \
        gunicorn \
        uvicorn[standard]


# ── Stage 2: runtime ─────────────────────────────────────────────────────────
# TARGETPLATFORM = the machine that will RUN the image (Oracle ARM node).
# python:3.11-slim has official images for both arm64 and amd64 on Docker Hub.
FROM --platform=$TARGETPLATFORM python:3.11-slim AS runtime

# Runtime system deps only — curl for the healthcheck, nothing else.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the compiled venv from the builder — no compilers needed here.
COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH"

# Copy application source.
COPY src/          ./src/
COPY scripts/      ./scripts/
COPY gunicorn.conf.py ./

# Copy model artefacts if they exist.
# In CI the create_test_model.py script generates these before the build.
# In production they come from Oracle Object Storage (covered in Helm stage).
COPY models/nyc/   ./models/nyc/

# Non-root user — containers should never run as root in production.
# Why? If an attacker escapes the app, they get a low-privilege user,
# not root access to the node.
RUN adduser --disabled-password --gecos "" appuser \
 && chown -R appuser:appuser /app
USER appuser

EXPOSE 8001

# Healthcheck — Docker (and Kubernetes) use this to know if the container
# is alive. If it fails 3 times, the container is restarted automatically.
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8001/health/live || exit 1

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["gunicorn", "-c", "gunicorn.conf.py", "src.serving.api:app"]
