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

    # Durable MySQL checkpoints
    checkpoint_mysql_enabled: bool = False
    checkpoint_mysql_dsn: str = ""

    # Prompt settings
    prompt_version: str = ""
    enable_grounding_check: bool = False

    # Harness / loop engineering settings
    harness: str = "allcallall_v1"
    loop_max_steps: int = 5
    enable_critic: bool = True
    enable_memory_reflection: bool = True

    model_config = {"env_prefix": "PY_AGENT_"}


config = AgentRuntimeConfig()
