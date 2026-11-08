# Build stage: resolve and install dependencies with uv into a virtualenv.
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

# Runtime stage: slim image, non-root user, only the venv and app code.
FROM python:3.12-slim
RUN useradd --create-home --uid 10001 spend
WORKDIR /app
COPY --from=build --chown=spend:spend /app /app
# The service writes report.json and review_decisions.jsonl to /app/data, so it must exist and
# belong to the app user before we drop privileges.
RUN mkdir -p /app/data && chown spend:spend /app/data
USER spend
ENV PATH="/app/.venv/bin:$PATH" HOST=0.0.0.0 PORT=8849 SPENDINTEL_DATA_DIR=/app/data SPENDINTEL_LINES=120000
EXPOSE 8849
# Startup generates 120,000 lines and runs the whole pipeline, about 90 seconds on one core.
HEALTHCHECK --interval=15s --timeout=3s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8849/healthz', timeout=2)"
CMD ["spendintel", "serve"]
