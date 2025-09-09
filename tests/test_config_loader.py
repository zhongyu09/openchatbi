"""Tests for configuration loading functionality."""

from unittest.mock import MagicMock, patch

import pytest
import yaml

from openchatbi.config_loader import Config, ConfigLoader


class TestConfigLoader:
    """Test configuration loading functionality."""

    def test_config_initialization(self):
        """Test Config model initialization."""
        config = Config(organization="TestOrg", dialect="presto")

        assert config.organization == "TestOrg"
        assert config.dialect == "presto"

    def test_config_from_dict(self):
        """Test creating Config from dictionary."""
        config_dict = {"organization": "TestOrg", "dialect": "mysql"}

        config = Config.from_dict(config_dict)
        assert config.organization == "TestOrg"
        assert config.dialect == "mysql"

    def test_config_loader_initialization(self):
        """Test ConfigLoader initialization."""
        loader = ConfigLoader()
        assert loader._config is not None

    def test_load_config_from_file(self, temp_dir):
        """Test loading configuration from YAML file."""
        config_data = {
            "organization": "TestOrg",
            "dialect": "presto",
            "default_llm": {"class": "langchain_openai.ChatOpenAI", "params": {"model": "gpt-4"}},
        }

        config_file = temp_dir / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        loader = ConfigLoader()

        with patch("langchain_openai.ChatOpenAI") as mock_openai:
            # Create a proper mock that satisfies BaseChatModel interface
            from langchain_core.language_models import BaseChatModel

            mock_llm_instance = MagicMock(spec=BaseChatModel)
            mock_openai.return_value = mock_llm_instance

            loader.load(str(config_file))

        config = loader.get()
        assert config.organization == "TestOrg"
        assert config.dialect == "presto"
        assert config.default_llm == mock_llm_instance

    def test_load_config_missing_file(self):
        """Test handling of missing configuration file."""
        loader = ConfigLoader()

        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path.yaml")

    def test_load_config_invalid_yaml(self, temp_dir):
        """Test handling of invalid YAML syntax."""
        config_file = temp_dir / "invalid_config.yaml"
        config_file.write_text("invalid: yaml: content: [")

        loader = ConfigLoader()

        with pytest.raises(yaml.YAMLError):
            loader.load(str(config_file))

    def test_load_config_with_bi_config_file(self, temp_dir):
        """Test loading configuration with BI config file."""
        bi_config_data = {"metrics": ["revenue", "users"], "dimensions": ["date", "region"]}

        bi_config_file = temp_dir / "bi_config.yaml"
        with open(bi_config_file, "w") as f:
            yaml.dump(bi_config_data, f)

        config_data = {"organization": "TestOrg", "bi_config_file": str(bi_config_file)}

        config_file = temp_dir / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        loader = ConfigLoader()
        loader.load(str(config_file))

        config = loader.get()
        assert config.bi_config["metrics"] == ["revenue", "users"]
        assert config.bi_config["dimensions"] == ["date", "region"]

    def test_load_config_with_catalog_store(self, temp_dir):
        """Test loading configuration with catalog store."""
        config_data = {
            "organization": "TestOrg",
            "catalog_store": {"store_type": "file_system", "data_path": str(temp_dir / "catalog_data")},
            "data_warehouse_config": {"uri": "sqlite:///:memory:", "include_tables": None, "database_name": "test_db"},
        }

        config_file = temp_dir / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        loader = ConfigLoader()

        loader.load(str(config_file))

        config = loader.get()
        # Just verify that a catalog store was created
        assert config.catalog_store is not None
        assert hasattr(config.catalog_store, "get_table_list")

    def test_load_config_with_llm_configs(self, temp_dir):
        """Test loading configuration with LLM configs."""
        config_data = {
            "organization": "TestOrg",
            "default_llm": {"class": "langchain_openai.ChatOpenAI", "params": {"model": "gpt-4", "temperature": 0.1}},
            "text2sql_llm": {"class": "langchain_openai.ChatOpenAI", "params": {"model": "gpt-3.5-turbo"}},
        }

        config_file = temp_dir / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        loader = ConfigLoader()

        with patch("langchain_openai.ChatOpenAI") as mock_openai:
            # Create proper mocks that satisfy BaseChatModel interface
            from langchain_core.language_models import BaseChatModel

            mock_instance1 = MagicMock(spec=BaseChatModel)
            mock_instance2 = MagicMock(spec=BaseChatModel)
            mock_openai.side_effect = [mock_instance1, mock_instance2]

            loader.load(str(config_file))

        config = loader.get()
        assert config.default_llm == mock_instance1
        assert config.text2sql_llm == mock_instance2

    def test_set_config(self):
        """Test setting configuration from dictionary."""
        config_dict = {"organization": "SetOrg", "dialect": "postgresql"}

        loader = ConfigLoader()
        loader.set(config_dict)

        config = loader.get()
        assert config.organization == "SetOrg"
        assert config.dialect == "postgresql"

    def test_get_config_not_loaded(self):
        """Test getting configuration when not loaded."""
        loader = ConfigLoader()
        loader._config = None

        with pytest.raises(ValueError, match="Configuration has not been loaded"):
            loader.get()

    def test_load_bi_config_missing_file(self, temp_dir):
        """Test loading missing BI config file."""
        nonexistent_file = temp_dir / "nonexistent_bi.yaml"

        loader = ConfigLoader()

        # Should not raise exception, just return empty dict
        result = loader.load_bi_config(str(nonexistent_file))
        assert result == {}

    def test_catalog_store_missing_store_type(self, temp_dir):
        """Test catalog store configuration without store_type."""
        config_data = {
            "organization": "TestOrg",
            "catalog_store": {
                "data_path": "/test/path"
                # Missing store_type
            },
        }

        config_file = temp_dir / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        loader = ConfigLoader()

        with pytest.raises(ValueError, match="catalog_store must have a store_type field"):
            loader.load(str(config_file))
