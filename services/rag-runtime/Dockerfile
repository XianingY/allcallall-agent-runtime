# Builder stage: resolves and installs the frozen dependency set with uv. The
# build tooling (uv, uvx, the lockfile) stays in this stage and never reaches
# the runtime image.
FROM python:3.11-slim@sha256:0dd364ba7e10242f07755449e3a3d0e35f9efd987952737b90def6709ab0c5ce AS builder

WORKDIR /app

# uv resolves and installs from the frozen uv.lock so the image is byte-for-byte
# reproducible instead of depending on whatever pip resolves at build time.
RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
COPY packages/shared ./packages/shared
COPY services/rag-runtime ./services/rag-runtime

# uv sync installs the frozen dependency set. --no-editable makes the
# allcallall-shared workspace dependency a real wheel so the runtime venv is
# self-contained (an editable install would point back at source that the
# runtime stage does not carry). The service member itself is `virtual` in the
# lock, so install it explicitly as a non-editable wheel.
RUN uv sync --frozen --no-dev --no-cache --no-editable --project services/rag-runtime \
    && VIRTUAL_ENV=/app/.venv uv pip install --no-cache --no-deps ./services/rag-runtime

# Runtime stage: only the frozen venv and the application source. Ownership is
# set inline on COPY; a chown -R pass would duplicate every file into a new
# layer and bloat the image.
FROM python:3.11-slim@sha256:0dd364ba7e10242f07755449e3a3d0e35f9efd987952737b90def6709ab0c5ce AS runtime

# Drop to a non-root user (matches the USER 10001 precedent used across the
# repo). uvicorn listens on 8091 (>1024), so no privileged bind is required.
RUN groupadd --gid 10001 appuser && \
    useradd --uid 10001 --gid appuser --no-create-home --shell /usr/sbin/nologin appuser

# The base image's system setuptools vendors wheel 0.45.1 and jaraco.context
# 5.3.0 under setuptools/_vendor/, both of which carry HIGH CVEs; upgrade it so
# the vendored copies are the fixed versions. The app itself runs from
# /app/.venv and never imports system setuptools.
RUN pip install --no-cache-dir "setuptools>=84"

WORKDIR /app

COPY --from=builder --chown=10001:10001 /app/.venv /app/.venv
COPY --from=builder --chown=10001:10001 /app/services/rag-runtime /app/services/rag-runtime

ENV PATH="/app/.venv/bin:$PATH"
ENV PY_RAG_RUNTIME_HOST=0.0.0.0
ENV PY_RAG_RUNTIME_PORT=8091
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

USER 10001:10001

EXPOSE 8091

CMD ["sh", "-c", "uvicorn allcallall_rag_runtime.main:app --host ${PY_RAG_RUNTIME_HOST:-0.0.0.0} --port ${PY_RAG_RUNTIME_PORT:-8091}"]
