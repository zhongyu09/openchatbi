Configuration
=============

The configuration system consists of two main classes:

- **Config**: Defines the configuration model.
- **ConfigLoader**: Manages loading and accessing configuration.

Config
------

.. autoclass:: openchatbi.config_loader.Config
    :exclude-members: organization, dialect, default_llm, embedding_model, text2sql_llm, bi_config, data_warehouse_config, catalog_store, mcp_servers, report_directory, python_executor

ConfigLoader
------------

.. autoclass:: openchatbi.config_loader.ConfigLoader
    :members:
    :undoc-members:
    :show-inheritance: