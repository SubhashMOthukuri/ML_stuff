# NYC Airbnb Price Prediction API
# Production image: gunicorn + uvicorn workers + ONNX Runtime
#
# Build:  docker build -t nyc-airbnb-api .
# Run:    docker run --env-file .env -p 8001:8001 nyc-airbnb-api

FROM python:3.11-slim

# System deps — only what we need (no extras)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cached unless requirements change)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir \
        gunicorn \
        uvicorn[standard] \
        onnxruntime \
        onnxmltools \
        slowapi \
        opentelemetry-sdk \
        opentelemetry-instrumentation-fastapi \
        opentelemetry-exporter-otlp-proto-grpc

# Copy source tree and model artifacts
COPY src/          ./src/
COPY models/nyc/   ./models/nyc/
COPY gunicorn.conf.py ./

# Non-root user for security
RUN adduser --disabled-password --gecos "" appuser \
 && chown -R appuser:appuser /app
USER appuser

# Expose API port
EXPOSE 8001

# Healthcheck — Docker marks container unhealthy if this fails
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8001/health/live || exit 1

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["gunicorn", "-c", "gunicorn.conf.py", "src.new_york_workflow.nyc_api:app"]
