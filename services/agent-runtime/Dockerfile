FROM python:3.11-slim@sha256:0dd364ba7e10242f07755449e3a3d0e35f9efd987952737b90def6709ab0c5ce

# Drop to a non-root user (matches the USER 10001 precedent used across the
# repo). uvicorn listens on 8090 (>1024), so no privileged bind is required.
RUN groupadd --gid 10001 appuser && \
    useradd --uid 10001 --gid appuser --no-create-home --shell /usr/sbin/nologin appuser

WORKDIR /app

# uv resolves and installs from the frozen uv.lock so the image is byte-for-byte
# reproducible instead of depending on whatever pip resolves at build time.
# setuptools 79.0.1 (pre-installed in the base image) vendors wheel 0.45.1 and
# jaraco.context 5.3.0 under setuptools/_vendor/, both of which carry HIGH CVEs;
# upgrade it so the vendored copies are the fixed versions.
RUN pip install --no-cache-dir uv "setuptools>=84"

COPY pyproject.toml uv.lock ./
COPY packages/shared ./packages/shared
COPY services/agent-runtime ./services/agent-runtime

# uv sync installs the frozen dependency set (including the allcallall-shared
# workspace dependency); the workspace member itself is `virtual` in the lock,
# so install it explicitly as a non-editable wheel.
RUN uv sync --frozen --no-dev --no-cache --project services/agent-runtime \
    && VIRTUAL_ENV=/app/.venv uv pip install --no-cache --no-deps ./services/agent-runtime

ENV PATH="/app/.venv/bin:$PATH"
ENV PY_AGENT_RUNTIME_HOST=0.0.0.0
ENV PY_AGENT_RUNTIME_PORT=8090
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

RUN chown -R 10001:10001 /app
USER 10001:10001

EXPOSE 8090

CMD ["sh", "-c", "uvicorn allcallall_agent_runtime.main:app --host ${PY_AGENT_RUNTIME_HOST:-0.0.0.0} --port ${PY_AGENT_RUNTIME_PORT:-8090}"]
