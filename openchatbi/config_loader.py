import importlib
import os
from importlib.util import find_spec
from typing import Any
from unittest.mock import MagicMock

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from openchatbi.catalog.factory import create_catalog_store
from openchatbi.utils import log


class LLMProviderConfig(BaseModel):
    """Resolved LLM objects for a single provider."""

    model_config = {"arbitrary_types_allowed": True}

    default_llm: BaseChatModel | MagicMock
    embedding_model: BaseModel | MagicMock | None = None
    text2sql_llm: BaseChatModel | MagicMock | None = None


class Config(BaseModel):
    """Configuration model for the OpenChatBI application.

    Attributes:
        organization (str): Organization name. Defaults to "The Company".
        dialect (str): SQL dialect to use. Defaults to "presto".
        default_llm (BaseChatModel): Default language model for general tasks.
        embedding_model (BaseModel): Language model for embedding generation.
        text2sql_llm (Optional[BaseChatModel]): Language model specifically for text-to-SQL tasks.
        bi_config (Dict[str, Any]): BI configuration loaded from YAML file. Defaults to empty dict.
        data_warehouse_config (Dict[str, Any]): Data warehouse configuration. Defaults to empty dict.
    """

    model_config = {"arbitrary_types_allowed": True}

    # General Configurations
    organization: str = "The Company"
    dialect: str = "presto"

    # LLM Configurations
    default_llm: BaseChatModel | MagicMock
    embedding_model: BaseModel | MagicMock | None = None
    text2sql_llm: BaseChatModel | MagicMock | None = None
    # Multiple LLM providers (optional)
    llm_provider: str | None = None
    llm_providers: dict[str, LLMProviderConfig] = {}

    # BI Configuration
    bi_config: dict[str, Any] = {}

    # Data Warehouse Configuration
    data_warehouse_config: dict[str, Any] = {}

    # Catalog Store
    catalog_store: Any = None

    # MCP Servers Configuration
    mcp_servers: list[dict[str, Any]] = []

    # Report Configuration
    report_directory: str = "./data"

    # Code Execution Configuration
    python_executor: str = "local"  # Options: "local", "restricted_local", "docker"

    # Visualization Configuration
    visualization_mode: str | None = "rule"  # Options: "rule", "llm", None (skip visualization)

    # Context Management Configuration
    context_config: dict[str, Any] = {}

    # Time Series Service Configuration
    timeseries_forecasting_service_url: str = "http://localhost:8765"

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "Config":
        """Creates a Config instance from a dictionary.

        Args:
            config (Dict[str, Any]): Dictionary containing configuration values.

        Returns:
            Config: A new Config instance with the provided values.
        """
        return cls(**config)


class ConfigLoader:
    """Singleton class to load and manage configuration settings for OpenChatBI.

    This class provides methods to load, get, and set configuration parameters
    for the application, including LLM models, SQL dialect, and other settings.
    """

    _instance = None
    _config: Config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    llm_configs = ["default_llm", "embedding_model", "text2sql_llm"]

    def get(self) -> Config:
        """Get the current configuration.

        Returns:
            Config: The current configuration instance.

        Raises:
            ValueError: If the configuration has not been loaded.
        """
        if self._config is None:
            raise ValueError("Configuration has not been loaded. Please call load() or set() first.")
        return self._config

    def load(self, config_file: str = None) -> None:
        """Load configuration from a YAML file.

        Args:
            config_file (str, optional): Path to configuration file. Uses CONFIG_FILE
                environment variable or 'openchatbi/config.yaml' if not provided.

        Raises:
            ImportError: If pyyaml is not installed.
            FileNotFoundError: If the configuration file cannot be found.
        """
        if config_file is None:
            config_file = os.getenv("CONFIG_FILE", "openchatbi/config.yaml")

        if not find_spec("yaml"):
            raise ImportError("Please install pyyaml to use this feature.")

        import yaml

        try:
            with open(config_file, encoding="utf-8") as file:
                config_data = yaml.safe_load(file)
                if config_data is None:
                    config_data = {}
        except FileNotFoundError:
            log(f"Configuration file not found: {config_file}, leave config un-loaded.")
            return
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file {config_file}: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to read configuration file {config_file}: {e}")

        self._process_config_dict(config_data)
        self._config = Config.from_dict(config_data)

    def _process_config_dict(self, config_data: dict[str, Any]) -> None:
        """
        Processes a configuration dictionary.
        """
        self._process_llm_providers(config_data)

        providers = config_data.get("llm_providers", {})
        selected_provider = None

        default_llm_value = config_data.get("default_llm")
        if isinstance(default_llm_value, str):
            # Simplified multi-provider config: default_llm: <provider_name>
            if not providers:
                raise ValueError("default_llm is a provider name but llm_providers is missing.")
            selected_provider = default_llm_value
        elif providers:
            # Backwards-compat: allow selecting provider via llm_provider
            legacy_provider = config_data.get("llm_provider")
            if isinstance(legacy_provider, str):
                selected_provider = legacy_provider
            elif "default_llm" not in config_data:
                # Pick the first provider in config order for backwards-compatible YAML behavior
                selected_provider = next(iter(providers.keys()), None)
            elif isinstance(default_llm_value, dict):
                raise ValueError(
                    "When using llm_providers, set default_llm to a provider name (e.g. default_llm: openai), "
                    "not a class config."
                )

        if providers:
            if not selected_provider or selected_provider not in providers:
                raise ValueError(
                    f"Unknown LLM provider '{selected_provider}'. Available: {sorted(providers.keys())}"
                )
            # Store selected provider for runtime lookups (UI/API can still override per-request)
            config_data["llm_provider"] = selected_provider
            # Populate top-level LLM objects for legacy call sites
            config_data["default_llm"] = providers[selected_provider].default_llm
            config_data.setdefault("embedding_model", providers[selected_provider].embedding_model)
            config_data.setdefault("text2sql_llm", providers[selected_provider].text2sql_llm)
        elif "default_llm" not in config_data:
            raise ValueError("Missing LLM config key: default_llm")

        if not config_data.get("embedding_model"):
            log("WARN: Missing LLM config key: embedding_model, will use BM25 based retrival only")
        if "data_warehouse_config" not in config_data:
            raise ValueError("Missing Data Warehouse config key: data_warehouse_config")

        # Load BI configuration
        if "bi_config_file" in config_data:
            bi_config = self.load_bi_config(config_data["bi_config_file"])
            bi_config.update(config_data.get("bi_config", {}))
            config_data["bi_config"] = bi_config

        if "catalog_store" in config_data:
            if "store_type" not in config_data["catalog_store"]:
                raise ValueError("catalog_store must have a store_type field.")
            catalog_store = create_catalog_store(
                **config_data["catalog_store"],
                auto_load=config_data["catalog_store"].get("auto_load", True),
                data_warehouse_config=config_data.get("data_warehouse_config"),
            )
        else:
            log("Catalog store config key `catalog_store` not found. Using default file system store.")
            catalog_store = create_catalog_store(
                store_type="file_system",
                auto_load=True,
                data_warehouse_config=config_data.get("data_warehouse_config"),
            )
        config_data["catalog_store"] = catalog_store

        for config_key in self.llm_configs:
            config_item = config_data.get(config_key)
            if not isinstance(config_item, dict) or "class" not in config_item:
                continue
            config_data[config_key] = self._instantiate_from_config_dict(config_item, config_key=config_key)

    def _instantiate_from_config_dict(self, config_item: dict[str, Any], *, config_key: str) -> Any:
        try:
            class_path = config_item["class"]
            if "." not in class_path:
                raise ValueError(f"Invalid class path format: {class_path}")
            module_name, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            llm_cls = getattr(module, class_name)
            params = config_item.get("params", {})
            return llm_cls(**params)
        except (ImportError, AttributeError, ValueError, TypeError) as e:
            raise RuntimeError(f"Failed to load {config_key} class '{config_item.get('class', '')}': {e}") from e

    def _process_llm_providers(self, config_data: dict[str, Any]) -> None:
        """Resolve llm_providers into instantiated provider configs (if present)."""
        raw_providers = config_data.get("llm_providers")
        if not raw_providers:
            return
        if not isinstance(raw_providers, dict):
            raise ValueError("llm_providers must be a mapping of provider_name -> config")

        providers: dict[str, LLMProviderConfig] = {}
        for provider_name, provider_cfg in raw_providers.items():
            if isinstance(provider_cfg, LLMProviderConfig):
                providers[str(provider_name)] = provider_cfg
                continue
            if not isinstance(provider_cfg, dict):
                raise ValueError(f"llm_providers.{provider_name} must be a mapping")

            resolved_cfg: dict[str, Any] = dict(provider_cfg)
            for config_key in self.llm_configs:
                config_item = resolved_cfg.get(config_key)
                if not isinstance(config_item, dict) or "class" not in config_item:
                    continue
                resolved_cfg[config_key] = self._instantiate_from_config_dict(
                    config_item, config_key=f"llm_providers.{provider_name}.{config_key}"
                )

            if "default_llm" not in resolved_cfg or resolved_cfg["default_llm"] is None:
                raise ValueError(f"llm_providers.{provider_name} missing default_llm")

            providers[str(provider_name)] = LLMProviderConfig(**resolved_cfg)

        config_data["llm_providers"] = providers

    def load_bi_config(self, bi_config_file: str) -> dict[str, Any]:
        """Load BI configuration from a YAML file.

        Args:
            bi_config_file (str): Path to the BI configuration file.
                Defaults to 'example/bi.yaml'.

        Returns:
            Dict[str, Any]: The loaded BI configuration as a dictionary.

        Raises:
            ImportError: If pyyaml is not installed.
            FileNotFoundError: If the BI configuration file cannot be found.
        """
        if not find_spec("yaml"):
            raise ImportError("Please install pyyaml to use this feature.")

        import yaml

        bi_config_data = {}

        try:
            with open(bi_config_file, encoding="utf-8") as file:
                bi_config_data = yaml.safe_load(file) or {}
        except FileNotFoundError:
            log(f"Warning: BI config file '{bi_config_file}' not found. Ignore load BI config from yaml file.")
        except yaml.YAMLError as e:
            log(f"Warning: Invalid YAML in BI config file '{bi_config_file}': {e}. Using empty config.")
        except Exception as e:
            log(f"Warning: Failed to read BI config file '{bi_config_file}': {e}. Using empty config.")

        return bi_config_data

    def set(self, config: dict[str, Any]) -> None:
        """Set the configuration from a dictionary.

        Args:
            config (Dict[str, Any]): Dictionary containing configuration values.
        """
        self._process_config_dict(config)
        self._config = Config.from_dict(config)
