# -----------------------------------------------------------------------------
# File:     tests/test_settings.py
# Purpose:  Unit tests for env settings and the config/models.yaml loader.
# Author:   Alexandre
# Created:  2026-09-25
# License:  Apache-2.0
# -----------------------------------------------------------------------------

from pathlib import Path

import pytest
from pydantic import ValidationError

from lab.settings import Provider, Settings, load_models_config

REPO_MODELS = Path(__file__).resolve().parents[1] / "config" / "models.yaml"


@pytest.fixture
def clean_env(tmp_path, monkeypatch):
    """Isola Settings de um .env real e de variáveis LAB_* do ambiente."""
    monkeypatch.chdir(tmp_path)
    for name in ("LAB_MAX_USD_PER_RUN", "LAB_CACHE_DIR", "LAB_MODELS_CONFIG", "LAB_CORPUS_CONFIG"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def test_settings_defaults(clean_env):
    s = Settings()
    assert s.lab_max_usd_per_run == 2.00
    assert s.lab_cache_dir == Path(".cache/llm")
    assert s.lab_models_config == Path("config/models.yaml")
    assert s.lab_corpus_config == Path("config/corpus.yaml")


def test_settings_read_environment(clean_env, monkeypatch):
    monkeypatch.setenv("LAB_MAX_USD_PER_RUN", "0.5")
    monkeypatch.setenv("LAB_CACHE_DIR", "/tmp/c")
    assert Settings().lab_max_usd_per_run == 0.5
    assert Settings().lab_cache_dir == Path("/tmp/c")


def test_settings_read_dotenv_and_ignore_unknown(clean_env):
    (clean_env / ".env").write_text("LAB_MAX_USD_PER_RUN=1.25\nPRIMARY_API_KEY=x\n")
    assert Settings().lab_max_usd_per_run == 1.25


def test_repo_models_yaml_is_valid():
    config = load_models_config(REPO_MODELS)
    assert {"generator", "judge", "embedding", "synthesizer"} <= config.roles.keys()
    assert all(role.provider in config.providers for role in config.roles.values())
    assert config.roles["embedding"].dims == 1024


def test_repo_models_yaml_keeps_primary_url_out_of_git():
    primary = load_models_config(REPO_MODELS).providers["primary"]
    assert primary.base_url is None
    assert primary.base_url_env == "PRIMARY_BASE_URL"


def test_provider_requires_exactly_one_base_url():
    with pytest.raises(ValidationError, match="exactly one"):
        Provider(api_key_env="K")
    with pytest.raises(ValidationError, match="exactly one"):
        Provider(base_url="https://x.test", base_url_env="X_URL", api_key_env="K")


def test_invalid_role_is_rejected(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(
        "roles:\n  generator: { provider: p, model: m, timeout_s: 0 }\n"
        "providers:\n  p: { base_url: 'https://x.test', api_key_env: K }\n"
    )
    with pytest.raises(ValidationError, match="timeout_s"):
        load_models_config(path)
