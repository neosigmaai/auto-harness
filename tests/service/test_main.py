import importlib


def test_main_background_flag_defaults_to_enabled(monkeypatch):
    monkeypatch.delenv("AUTOHARNESS_START_BACKGROUND", raising=False)
    module = importlib.import_module("autoharness_service.main")

    assert module.background_enabled() is True


def test_main_background_flag_can_disable_background_workers(monkeypatch):
    monkeypatch.setenv("AUTOHARNESS_START_BACKGROUND", "0")
    module = importlib.import_module("autoharness_service.main")

    assert module.background_enabled() is False
