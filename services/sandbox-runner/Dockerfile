FROM python:3.12-slim@sha256:05cda9777409a9c3ffddd94a4c476b79f0769a0b4857f0c7ed9226b6800b0d6f

RUN groupadd --gid 10001 runner && useradd --uid 10001 --gid runner --no-create-home runner
WORKDIR /app

# uv resolves and installs from the frozen uv.lock so the image is byte-for-byte
# reproducible instead of depending on whatever pip resolves at build time.
RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
COPY services/sandbox-runner ./services/sandbox-runner

# uv sync installs the frozen dependency set; the workspace member itself is
# `virtual` in the lock, so install it explicitly as a non-editable wheel.
RUN uv sync --frozen --no-dev --no-cache --project services/sandbox-runner \
    && VIRTUAL_ENV=/app/.venv uv pip install --no-cache --no-deps ./services/sandbox-runner

USER 10001:10001
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8093
CMD ["uvicorn", "allcallall_sandbox_runner.main:app", "--host", "0.0.0.0", "--port", "8093", "--no-access-log"]
