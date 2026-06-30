FROM python:3.11-slim

WORKDIR /app

COPY packages/shared ./packages/shared
COPY services/agent-runtime ./services/agent-runtime

RUN pip install --no-cache-dir ./packages/shared ./services/agent-runtime

ENV PY_AGENT_RUNTIME_HOST=0.0.0.0
ENV PY_AGENT_RUNTIME_PORT=8090

EXPOSE 8090

CMD ["sh", "-c", "uvicorn allcallall_agent_runtime.main:app --app-dir /app/services/agent-runtime --host ${PY_AGENT_RUNTIME_HOST:-0.0.0.0} --port ${PY_AGENT_RUNTIME_PORT:-8090}"]
