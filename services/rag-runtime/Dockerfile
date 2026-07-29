FROM python:3.11-slim

# Drop to a non-root user (matches the USER 10001 precedent used across the
# repo). uvicorn listens on 8091 (>1024), so no privileged bind is required.
RUN groupadd --gid 10001 appuser && \
    useradd --uid 10001 --gid appuser --no-create-home --shell /usr/sbin/nologin appuser

WORKDIR /app

COPY packages/shared ./packages/shared
COPY services/rag-runtime ./services/rag-runtime
RUN pip install --no-cache-dir ./packages/shared ./services/rag-runtime

ENV PY_RAG_RUNTIME_HOST=0.0.0.0
ENV PY_RAG_RUNTIME_PORT=8091
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

RUN chown -R 10001:10001 /app
USER 10001:10001

EXPOSE 8091

CMD ["sh", "-c", "uvicorn allcallall_rag_runtime.main:app --app-dir /app/services/rag-runtime --host ${PY_RAG_RUNTIME_HOST:-0.0.0.0} --port ${PY_RAG_RUNTIME_PORT:-8091}"]
