from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration from environment variables or a `.env` file. Empty values count as unset."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_ignore_empty=True)

    llm_provider: Literal["azure_openai", "openai_compatible"] = "azure_openai"
    chat_model: str = Field("gpt-4o-mini", description="Model name, or deployment name on Azure.")
    embedding_model: str = Field("text-embedding-3-small", description="Model or deployment name.")
    embedding_dimensions: int = Field(1536, gt=0)
    temperature: float | None = None

    azure_openai_endpoint: str | None = None
    azure_openai_api_key: SecretStr | None = Field(None, description="Unset to use Microsoft Entra ID.")
    azure_openai_api_version: str = "2024-10-21"

    openai_base_url: str = Field("http://localhost:11434/v1", description="Defaults to local Ollama.")
    openai_api_key: SecretStr = SecretStr("not-needed-for-local")

    index_provider: Literal["memory", "azure_search"] = "memory"
    memory_index_path: Path | None = Path(".data/index.json")

    azure_search_endpoint: str | None = None
    azure_search_api_key: SecretStr | None = Field(None, description="Unset to use Microsoft Entra ID.")
    azure_search_index: str = "knowledge"
    azure_search_existing_index: bool = Field(
        False, description="Search an index owned by another system, read-only."
    )
    azure_search_source_field: str = Field("id", description="Existing index: document id field.")
    azure_search_content_field: str = Field("content", description="Existing index: text field.")
    azure_search_vector_field: str = Field("content_vector", description="Existing index: vector field.")

    agent_runtime: Literal["local", "foundry"] = "local"
    max_turns: int = Field(8, ge=1, le=50)

    foundry_project_endpoint: str | None = None
    foundry_agent_ids: dict[str, str] = Field(
        default_factory=dict, description='JSON map of agent name to Foundry id, e.g. {"triage": "asst_..."}.'
    )
    foundry_model: str | None = Field(None, description="Deployment used when provisioning agents.")
    foundry_search_connection_id: str | None = None
    foundry_agent_prefix: str = Field("agents-starter-", description="Prefix for agent display names.")
    foundry_run_timeout_seconds: float = Field(300, gt=0)

    api_key: SecretStr | None = Field(
        None, description="Require an X-API-Key header on all routes but /health."
    )

    @model_validator(mode="after")
    def _check_required(self) -> Settings:
        required = {
            "AZURE_OPENAI_ENDPOINT": self.llm_provider == "azure_openai" and not self.azure_openai_endpoint,
            "AZURE_SEARCH_ENDPOINT": self.index_provider == "azure_search" and not self.azure_search_endpoint,
            "FOUNDRY_PROJECT_ENDPOINT": self.agent_runtime == "foundry" and not self.foundry_project_endpoint,
        }
        if missing := [name for name, is_missing in required.items() if is_missing]:
            raise ValueError(f"Missing required settings: {', '.join(missing)}")
        return self
