"""Tests for MemoryConfig loading via the main Config (pydantic field declaration)."""

from unittest.mock import MagicMock

from openchatbi.config_loader import Config, ConfigLoader
from openchatbi.memory_config import MemoryConfig, get_memory_config


def test_config_declares_memory_config_field():
    # pydantic BaseModel silently drops undeclared fields; this proves it is declared.
    assert "memory_config" in Config.model_fields


def test_memory_config_defaults_off():
    cfg = MemoryConfig()
    assert cfg.enable_pattern_memory is False
    assert cfg.enable_memory_decay_rerank is False


def test_get_memory_config_reads_from_main_config():
    config_dict = {
        "organization": "Test Company",
        "dialect": "presto",
        "default_llm": MagicMock(),
        "embedding_model": MagicMock(),
        "data_warehouse_config": {"uri": "sqlite:///:memory:", "include_tables": None, "database_name": "test_db"},
        "memory_config": {"enable_pattern_memory": True, "max_patterns_per_query": 3},
    }
    loader = ConfigLoader()
    loader.set(config_dict)

    mc = get_memory_config()
    assert mc.enable_pattern_memory is True
    assert mc.max_patterns_per_query == 3
    # unspecified keys keep their defaults
    assert mc.enable_memory_decay_rerank is False


def test_get_memory_config_defaults_when_unset():
    config_dict = {
        "organization": "Test Company",
        "dialect": "presto",
        "default_llm": MagicMock(),
        "embedding_model": MagicMock(),
        "data_warehouse_config": {"uri": "sqlite:///:memory:", "include_tables": None, "database_name": "test_db"},
    }
    loader = ConfigLoader()
    loader.set(config_dict)

    mc = get_memory_config()
    assert mc.enable_pattern_memory is False
