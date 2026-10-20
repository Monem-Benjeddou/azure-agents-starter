import pytest
from pydantic import ValidationError

from agents_starter.config import Settings


def test_missing_endpoints_for_selected_providers_fail_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    with pytest.raises(ValidationError, match="AZURE_OPENAI_ENDPOINT, AZURE_SEARCH_ENDPOINT"):
        Settings(_env_file=None, index_provider="azure_search")  # type: ignore[call-arg]


def test_empty_env_values_mean_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "")
    monkeypatch.setenv("FOUNDRY_AGENT_IDS", '{"triage": "asst_1"}')

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.azure_openai_api_key is None
    assert settings.foundry_agent_ids == {"triage": "asst_1"}
