import importlib
import os
from importlib.util import find_spec
from typing import Any

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from openchatbi.catalog.factory import create_catalog_store
from openchatbi.utils import log


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

    # General Configurations
    organization: str = "The Company"
    dialect: str = "presto"

    # LLM Configurations
    default_llm: BaseChatModel = None
    embedding_model: BaseModel = None
    text2sql_llm: BaseChatModel | None = None

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
            with open(config_file, "r", encoding="utf-8") as file:
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

    def _process_config_dict(self, config_data: Config) -> None:
        """
        Processes a configuration dictionary.
        """
        if "default_llm" not in config_data:
            raise ValueError(f"Missing LLM config key: default_llm")
        if "embedding_model" not in config_data:
            raise ValueError(f"Missing LLM config key: embedding_model")
        if "data_warehouse_config" not in config_data:
            raise ValueError(f"Missing Data Warehouse config key: data_warehouse_config")

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
            if config_key not in config_data or "class" not in config_data[config_key]:
                continue
            try:
                class_path = config_data[config_key]["class"]
                if "." not in class_path:
                    raise ValueError(f"Invalid class path format: {class_path}")
                module_name, class_name = class_path.rsplit(".", 1)
                module = importlib.import_module(module_name)
                llm_cls = getattr(module, class_name)
                params = config_data[config_key].get("params", {})
                config_data[config_key] = llm_cls(**params)
            except (ImportError, AttributeError, ValueError, TypeError) as e:
                raise RuntimeError(
                    f"Failed to load {config_key} class '{config_data[config_key]['class']}': {e}"
                ) from e

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
            with open(bi_config_file, "r", encoding="utf-8") as file:
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
