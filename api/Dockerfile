# Build context: repo ROOT (not api/).
#   docker build -f api/Dockerfile -t scalptrader-api .
#   fly deploy                 # fly always uses repo root as context
#
# The pyproject.toml at the repo root declares both the engine package
# (setuptools.packages.find where=["engine"]) and the shared deps; the api/
# app imports scalpengine.* so the engine source must be present at build time.

FROM python:3.13-slim

# ---- system deps (psycopg2 needs libpq) ---------------------------------- #
RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

# ---- non-root user -------------------------------------------------------- #
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# ---- install Python deps from the root pyproject -------------------------  #
# Copy only the files needed for pip install so layer is cached efficiently.
COPY pyproject.toml ./
# Engine source is needed both as a build-time dependency and at runtime.
COPY engine ./engine

# Install the project (engine package) from the root pyproject.
# torch is a heavy optional dep; we exclude it so the image stays slim.
# The api/requirements.txt pins the same packages; pip resolves from pyproject.
RUN pip install --no-cache-dir --no-deps -e ".[dev]" \
    && pip install --no-cache-dir \
        "fastapi>=0.110" \
        "uvicorn[standard]>=0.29" \
        "pydantic>=2.6" \
        "pydantic-settings>=2.2" \
        "SQLAlchemy>=2.0" \
        "alembic>=1.13" \
        "structlog>=24.1" \
        "exchange-calendars>=4.5" \
        "numpy>=1.26" \
        "pandas>=2.0" \
        "httpx>=0.27" \
        "PyJWT>=2.8" \
        "redis>=5.0" \
        "psycopg2-binary>=2.9" \
        "aiosqlite>=0.20" \
        "websockets>=12.0" \
        "alpaca-py>=0.21"

# ---- copy API source ------------------------------------------------------ #
COPY api ./api

# Set PYTHONPATH so `from scalpengine.*` resolves from /app/engine
ENV PYTHONPATH=/app/engine
ENV PORT=8080

# Drop to non-root user
USER appuser

EXPOSE 8080

# ---- healthcheck ---------------------------------------------------------- #
# Fly's [checks] block also probes /healthz, but a Docker-native healthcheck
# lets `docker ps` report healthy before container traffic is shifted.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/healthz')"

CMD ["uvicorn", "api.app.main:app", "--host", "0.0.0.0", "--port", "8080"]
