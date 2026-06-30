FROM python:3.11-slim

WORKDIR /app
COPY packages/shared ./packages/shared
COPY services/rag-runtime ./services/rag-runtime
RUN pip install --no-cache-dir ./packages/shared ./services/rag-runtime

ENV PY_RAG_RUNTIME_HOST=0.0.0.0
ENV PY_RAG_RUNTIME_PORT=8091

EXPOSE 8091

CMD ["sh", "-c", "uvicorn allcallall_rag_runtime.main:app --app-dir /app/services/rag-runtime --host ${PY_RAG_RUNTIME_HOST:-0.0.0.0} --port ${PY_RAG_RUNTIME_PORT:-8091}"]
