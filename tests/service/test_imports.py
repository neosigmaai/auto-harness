from autoharness_service.config import ServiceSettings, load_settings


def test_load_settings_defaults(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("AUTOHARNESS_SERVICE_MODE", raising=False)

    settings = load_settings()

    assert isinstance(settings, ServiceSettings)
    assert settings.database_url == "postgresql://autoharness:autoharness@localhost:5432/autoharness"
    assert settings.default_mode == "simulated"
    assert settings.default_sandbox_provider == "daytona"
    assert settings.max_local_concurrency == 4
