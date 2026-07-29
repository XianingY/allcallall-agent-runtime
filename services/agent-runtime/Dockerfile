FROM python:3.11-slim

# Drop to a non-root user (matches the USER 10001 precedent used across the
# repo). uvicorn listens on 8090 (>1024), so no privileged bind is required.
RUN groupadd --gid 10001 appuser && \
    useradd --uid 10001 --gid appuser --no-create-home --shell /usr/sbin/nologin appuser

WORKDIR /app

COPY packages/shared ./packages/shared
COPY services/agent-runtime ./services/agent-runtime

RUN pip install --no-cache-dir ./packages/shared ./services/agent-runtime

ENV PY_AGENT_RUNTIME_HOST=0.0.0.0
ENV PY_AGENT_RUNTIME_PORT=8090
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

RUN chown -R 10001:10001 /app
USER 10001:10001

EXPOSE 8090

CMD ["sh", "-c", "uvicorn allcallall_agent_runtime.main:app --app-dir /app/services/agent-runtime --host ${PY_AGENT_RUNTIME_HOST:-0.0.0.0} --port ${PY_AGENT_RUNTIME_PORT:-8090}"]
