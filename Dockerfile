# Multi-stage Dockerfile: separate build pipeline from serving

# ── Base: common dependencies ────────────────────────────────────────────
FROM python:3.11-slim AS base
WORKDIR /app
RUN pip install --no-cache-dir \
    sentence-transformers>=2.2.0 \
    numpy>=1.24 \
    pydantic>=2.0 \
    pydantic-settings>=2.0 \
    qdrant-client>=1.7.0

# Pre-download the BGE model so it's cached in the image
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

# ── Build: offline pipeline (heavy deps) ─────────────────────────────────
FROM base AS build
RUN pip install --no-cache-dir \
    hdbscan>=0.8.33 \
    networkx>=3.1 \
    node2vec>=0.4.6 \
    scikit-learn>=1.3 \
    openai>=1.0.0 \
    pandas>=2.0
COPY src/ src/
COPY data/ data/
CMD ["python", "-m", "audience_targeting.build_pipeline"]

# ── Serve: API (minimal deps) ────────────────────────────────────────────
FROM base AS serve
RUN pip install --no-cache-dir \
    fastapi>=0.104.0 \
    uvicorn[standard]>=0.24.0 \
    gunicorn>=21.2.0 \
    python-json-logger>=2.0.0 \
    slowapi>=0.1.9

RUN useradd -m -u 1000 appuser

COPY src/ src/
ENV PYTHONPATH=/app/src
USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["sh", "-c", "gunicorn audience_targeting.api:app -k uvicorn.workers.UvicornWorker --workers ${WEB_CONCURRENCY:-2} --bind 0.0.0.0:8000 --timeout 180 --graceful-timeout 30"]
