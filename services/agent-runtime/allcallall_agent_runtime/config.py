from __future__ import annotations

from pydantic_settings import BaseSettings


class AgentRuntimeConfig(BaseSettings):
    """Centralized configuration for the Agent Runtime.

    All settings are loaded from environment variables with the ``PY_AGENT_`` prefix.
    """

    # Provider settings
    provider: str = "rules"
    provider_strict: bool = True

    # OpenAI settings
    openai_base_url: str = ""
    openai_api_key: str = ""
    openai_model: str = "gpt-4"
    openai_timeout_sec: float = 30.0

    # Tool bridge settings
    tool_bridge_base_url: str = ""
    tool_bridge_token: str = ""
    tool_bridge_timeout_sec: float = 10.0

    # Inbound API auth for /v1 run endpoints. When empty (default), run endpoints
    # are unauthenticated but emit a one-time warning. Set this to require a
    # `Authorization: Bearer <token>` header on every workflow run endpoint.
    api_token: str = ""

    # RAG runtime settings
    rag_runtime_base_url: str = ""
    rag_runtime_timeout_sec: float = 10.0

    # Agentic RAG settings
    enable_agentic_rag: bool = False
    rag_max_retrieval_steps: int = 3
    rag_min_confidence: float = 0.6

    # Resilience settings (retry with exponential backoff)
    provider_max_retries: int = 2
    tool_bridge_max_retries: int = 2
    rag_runtime_max_retries: int = 2
    retry_base_delay_sec: float = 0.5
    retry_max_delay_sec: float = 8.0

    # Durable checkpoints (backend selection; decoupled from the harness)
    checkpoint_store: str = ""  # "" (auto) | "none" | "mysql" | "sqlite" | "memory"
    checkpoint_mysql_enabled: bool = False
    checkpoint_mysql_dsn: str = ""
    checkpoint_mysql_pool_size: int = 4  # connections pooled per MySQLCheckpointSaver
    checkpoint_sqlite_path: str = ":memory:"  # file path or ":memory:" for reproducible test envs

    # Prompt settings
    prompt_version: str = ""
    enable_grounding_check: bool = False

    # Harness / loop engineering settings
    harness: str = "allcallall_v1"
    loop_max_steps: int = 5
    enable_critic: bool = True
    enable_memory_reflection: bool = True
    # Bounded-quality-retry budget for the two-tier CheckAgent loop.
    max_quality_retries: int = 1
    # Per-request wall-clock deadline for a single workflow run (seconds).
    # Exceeding it raises HarnessTimeoutExceeded, which the HTTP layer maps to a
    # 504/408 response. Set to 0 to disable the deadline.
    request_timeout_seconds: float = 120.0

    # Context compression / hierarchical memory (Module 4)
    model_history_max_tokens: int = 4000
    context_compression_strategy: str = "summary"  # "summary" | "full"
    enable_context_compression: bool = False  # opt-in; off keeps legacy behavior

    # Async write-tool queue (Module 6): enqueue approved write proposals after
    # the workflow run and execute them in the background via the Go tool bridge.
    # Off by default — when disabled, write proposals are returned to the caller
    # (legacy behavior) and nothing is enqueued.
    enable_tool_queue: bool = False

    # Skill registry hardening (Module 5): load skills from an explicit manifest
    # (listing allowed files + expected risk_level) instead of trusting
    # frontmatter self-reporting. Off by default.
    enable_skills: bool = False
    skill_manifest_path: str = ""

    model_config = {"env_prefix": "PY_AGENT_"}


config = AgentRuntimeConfig()
