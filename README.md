# OpenChatBI

OpenChatBI is an open source, chat-based intelligent BI tool powered by large language models, designed to help users 
query, analyze, and visualize data through natural language conversations. Built on LangGraph and LangChain ecosystem, 
it provides chat agents and workflows that support natural language to SQL conversion and streamlined data analysis.

<img src="https://github.com/zhongyu09/openchatbi/raw/main/example/demo.gif" alt="Demo" width="800">

## Core Features

1. **Natural Language Interaction**: Get data analysis results by asking questions in natural language
2. **Automatic SQL Generation**: Convert natural language queries into SQL statements using advanced text2sql workflows
   with schema linking and well organized prompt engineering
3. **Data Visualization**: Generate intuitive data visualizations (via plotly)
4. **Data Catalog Management**: Automatically discovers and indexes database table structures, supports flexible catalog 
   storage backends, and easily maintains business explanations for tables and columns as well as optimizes Prompts.
5. **Knowledge Base Integration**: Answer complex questions by combining catalog based knowledge retrival and external
   knowledge base retrival (via MCP tools)
6. **Code Execution**: Execute Python code for data analysis and visualization
7. **Interactive Problem-Solving**: Proactively ask users for more context when information is incomplete
8. **Persistent Memory**: Conversation management and user characteristic memory based on LangGraph checkpointing
9. **MCP Support**: Integration with MCP tools by configuration
10. **Web UI Interface**: Provide 2 sample UI: simple and streaming web interfaces using Gradio and Streamlit, easy to
   integrate with other web applications

## Roadmap

1. **Time Series Forecasting**: Forecasting models deployed in-house
2. **Root Cause Analysis Algorithm**: Multi-dimensional drill-down capabilities for anomaly investigation

# Getting started

## Installation & Setup

### Prerequisites

- Python 3.11 or higher
- Access to a supported LLM provider (OpenAI, Anthropic, etc.)
- Data Warehouse (Database) credentials (like Presto, PostgreSQL, MySQL, etc.)
- Docker (optional, required only for `docker` executor mode)

### Installation

1. **Using uv (recommended):**

```bash
git clone git@github.com:zhongyu09/openchatbi
uv sync
```

2. **Using pip:**

```bash
pip install git+https://github.com/zhongyu09/openchatbi@main
```

3. **For development:**

```bash
git clone git@github.com:zhongyu09/openchatbi
uv sync --group dev
```

4. If you have issues when installing pysqlite3 on macOS, try to install sqlite using Homebrew first:

```bash
brew install sqlite
brew info sqlite
export LDFLAGS="-L/opt/homebrew/opt/sqlite/lib"
export CPPFLAGS="-I/opt/homebrew/opt/sqlite/include"
```

### Run Demo

Run demo using **example dataset** from spider dataset, you need to provide "YOUR OPENAI API KEY" or change config to other LLM.
```bash
cp example/config.yaml openchatbi/config.yaml
sed -i 's/YOUR_API_KEY_HERE/[YOUR OPENAI API KEY]/g' openchatbi/config.yaml
python run_streamlit_ui.py
```

### Configuration

1. **Create configuration file**

Copy the configuration template:
```bash
cp openchatbi/config.yaml.template openchatbi/config.yaml
```
Or create an empty YAML file.

2. **Configure your LLMs:**

```yaml
default_llm:
  class: langchain_openai.ChatOpenAI
  params:
    api_key: YOUR_API_KEY_HERE
    model: gpt-4.1
    temperature: 0.02
    max_tokens: 8192
embedding_model:
  class: langchain_openai.OpenAIEmbeddings
  params:
    api_key: YOUR_API_KEY_HERE
    model: text-embedding-3-large
    chunk_size: 1024
```

3. **Configure your data warehouse:**

```yaml
organization: Your Company
dialect: presto
data_warehouse_config:
  uri: "presto://user@host:8080/catalog/schema"
  include_tables:
    - your_table_name
  database_name: "catalog.schema"
```

### Running the Application

1. **Invoking LangGraph:**

```bash
export CONFIG_FILE=YOUR_CONFIG_FILE_PATH
```

```python
from openchatbi import get_default_graph

graph = get_default_graph()
graph.invoke({"messages": [{"role": "user", "content": "Show me ctr trends for the past 7 days"}]},
    config={"configurable": {"thread_id": "1"}})
```

```
# System-generated SQL
SELECT date, SUM(clicks)/SUM(impression) AS ctr
FROM ad_performance
WHERE date >= CURRENT_DATE - 7 DAYS
GROUP BY date
ORDER BY date;
```

2. **Sample Web UI:**

Streamlit based UI:
```bash
streamlit run sample_ui streamlit_ui.py
```

Run Gradio based UI:
```bash
python sample_ui/streaming_ui.py
```

## Configuration Instructions

The configuration template is provided at `config.yaml.template`. Key configuration sections include:

### Basic Settings

- `organization`: Organization name (e.g., "Your Company")
- `dialect`: Database dialect (e.g., "presto")
- `bi_config_file`: Path to BI configuration file (e.g., "example/bi.yaml")

### Catalog Store Configuration

- `catalog_store`: Configuration for data catalog storage
    - `store_type`: Storage type (e.g., "file_system")
    - `data_path`: Path to catalog data stored by file system (e.g., "./example")

### Data Warehouse Configuration

- `data_warehouse_config`: Database connection settings
    - `uri`: Connection string for your database
    - `include_tables`: List of tables to include in catalog, leave empty to include all tables
    - `database_name`: Database name for catalog
    - `token_service`: Token service URL (for data warehouse that need token authentication like Presto)
    - `user_name` / `password`: Token service credentials

### LLM Configuration

Various LLMs are supported based on LangChain, see LangChain API
Document(https://python.langchain.com/api_reference/reference.html#integrations) for full list that support
`chat_models`. You can configure different LLMs for different tasks:

- `default_llm`: Primary language model for general tasks
- `embedding_model`: Model for embedding generation
- `text2sql_llm`: Specialized model for SQL generation

Commonly used LLM providers and their corresponding classes and installation commands:

- **Anthropic**: `langchain_anthropic.ChatAnthropic`, `pip install langchain-anthropic`
- **OpenAI**: `langchain_openai.ChatOpenAI`, `pip install langchain-openai`
- **Azure OpenAI**: `langchain_openai.AzureChatOpenAI`, `pip install langchain-openai`
- **Google Vertex AI**: `chat_models.ChatVertexAI`, `langchain-google-vertexai`
- **Bedrock**: `chat_models.bedrock.ChatBedrock`, `pip install langchain-aws`
- **Huggingface**: `chat_models.huggingface.ChatHuggingFace`, `pip install langchain-huggingface`
- **Deepseek**: `chat_models.ChatDeepSeek`, `pip install langchain-deepseek`
- **Ollama**: `chat_models.ChatOllama`, `pip install langchain-ollama`

### Advanced Configuration

OpenChatBI supports sophisticated customization through prompt engineering and catalog management features:

- **Prompt Engineering Configuration**: Customize system prompts, business glossaries, and data warehouse introductions
- **Data Catalog Management**: Configure table metadata, column descriptions, and SQL generation rules
- **Business Rules**: Define table selection criteria and domain-specific SQL constraints

For detailed configuration options and examples, see the [Advanced Features](#advanced-features) section.

## Architecture Overview

OpenChatBI is built using a modular architecture with clear separation of concerns:

1. **LangGraph Workflows**: Core orchestration using state machines for complex multi-step processes
2. **Catalog Management**: Flexible data catalog system supporting multiple storage backends
3. **Text2SQL Pipeline**: Advanced natural language to SQL conversion with schema linking
4. **Code Execution**: Sandboxed Python execution environment for data analysis
5. **Tool Integration**: Extensible tool system for human interaction and knowledge search
6. **Persistent Memory**: SQLite-based conversation state management

## Technology Stack

- **Frameworks**: LangGraph, LangChain, FastAPI, Gradio/Streamlit
- **Large Language Models**: Azure OpenAI (GPT-4), Anthropic Claude, OpenAI GPT models
- **Databases**: Presto, Trino, MySQL with SQLAlchemy support
- **Code Execution**: Local Python, RestrictedPython, Docker containerization
- **Development**: Python 3.11+, with modern tooling (Black, Ruff, MyPy, Pytest)
- **Storage**: SQLite for conversation checkpointing, file system catalog storage

## Project Structure

```
openchatbi/
├── README.md                    # Project documentation
├── pyproject.toml               # Modern Python project configuration
├── Dockerfile.python-executor  # Docker image for isolated code execution
├── run_tests.py                # Test runner script
├── run_streamlit_ui.py         # Streamlit UI launcher
├── openchatbi/                 # Core application code
│   ├── __init__.py             # Package initialization
│   ├── config.yaml.template    # Configuration template
│   ├── config_loader.py        # Configuration management
│   ├── constants.py            # Application constants
│   ├── agent_graph.py          # Main LangGraph workflow
│   ├── graph_state.py          # State definition for workflows
│   ├── utils.py                # Utility functions
│   ├── catalog/                # Data catalog management
│   │   ├── __init__.py         # Package initialization
│   │   ├── catalog_loader.py   # Catalog loading logic
│   │   ├── catalog_store.py    # Catalog storage interface
│   │   ├── entry.py            # Catalog entry points
│   │   ├── factory.py          # Catalog factory patterns
│   │   ├── helper.py           # Catalog helper functions
│   │   ├── schema_retrival.py  # Schema retrieval logic
│   │   └── token_service.py    # Token service integration
│   ├── code/                   # Code execution framework
│   │   ├── __init__.py         # Package initialization
│   │   ├── executor_base.py    # Base executor interface
│   │   ├── local_executor.py   # Local Python execution
│   │   ├── restricted_local_executor.py # RestrictedPython execution
│   │   └── docker_executor.py  # Docker-based isolated execution
│   ├── llm/                    # LLM integration layer
│   │   ├── __init__.py         # Package initialization
│   │   └── llm.py              # LLM management and retry logic
│   ├── prompts/                # Prompt templates and engineering
│   │   ├── __init__.py         # Package initialization
│   │   ├── agent_prompt.md     # Main agent prompts
│   │   ├── extraction_prompt.md # Information extraction prompts
│   │   ├── system_prompt.py    # System prompt management
│   │   ├── table_selection_prompt.md # Table selection prompts
│   │   └── text2sql_prompt.md  # Text-to-SQL prompts
│   ├── text2sql/               # Text-to-SQL conversion pipeline
│   │   ├── __init__.py         # Package initialization
│   │   ├── data.py             # Data and retriever for Text-to-SQL
│   │   ├── extraction.py       # Information extraction
│   │   ├── generate_sql.py     # SQL generation and execution logic
│   │   ├── schema_linking.py   # Schema linking process
│   │   ├── sql_graph.py        # SQL generation LangGraph workflow
│   │   ├── text2sql_utils.py   # Text2SQL utilities
│   │   └── visualization.py    # Data visualization functions
│   └── tool/                   # LangGraph tools and functions
│       ├── ask_human.py        # Human-in-the-loop interactions
│       ├── memory.py           # Memory management tool
│       ├── mcp_tools.py        # MCP (Model Context Protocol) integration
│       ├── run_python_code.py  # Configurable Python code execution
│       ├── save_report.py      # Report saving functionality
│       └── search_knowledge.py # Knowledge base search
├── sample_api/                 # API implementations
│   └── async_api.py            # Asynchronous FastAPI example
├── sample_ui/                  # Web interface implementations
│   ├── memory_ui.py            # Memory-enhanced UI interface
│   ├── plotly_utils.py         # Plotly utilities and helpers
│   ├── simple_ui.py            # Simple non-streaming Gradio UI
│   ├── streaming_ui.py         # Streaming Gradio UI with real-time updates
│   ├── streamlit_ui.py         # Streaming Streamlit UI with enhanced features
│   └── style.py                # UI styling and CSS
├── example/                    # Example configurations and data
│   ├── bi.yaml                 # BI configuration example
│   ├── config.yaml             # Application config example
│   ├── table_info.yaml         # Table information
│   ├── table_columns.csv       # Table column registry
│   ├── common_columns.csv      # Common column definitions
│   ├── sql_example.yaml        # SQL examples for retrieval
│   ├── table_selection_example.csv # Table selection examples
│   └── tracking_orders.sqlite  # Sample SQLite database
├── tests/                      # Test suite
│   ├── __init__.py             # Package initialization
│   ├── conftest.py             # Test configuration
│   ├── test_*.py               # Test modules for various components
│   └── README.md               # Testing documentation
├── docs/                       # Documentation
│   ├── source/                 # Sphinx documentation source
│   ├── build/                  # Built documentation
│   ├── Makefile                # Documentation build scripts
│   └── make.bat                # Windows build script
└── .github/                    # GitHub workflows and templates
    └── workflows/              # CI/CD workflows
```

## Advanced Features

### Visualization configuration
You can choose rule-based or llm-based visualization or disable visualization.
```yaml
# Options: "rule" (rule-based), "llm" (LLM-based), or null (skip visualization)
visualization_mode: llm
```

### Prompt Engineering
#### Basic Knowledge & Glossary

You can define basic knowledge and glossary in `example/bi.yaml`, for example:

```yaml
basic_knowledge_glossary: |
  # Basic Knowledge Introduction
    The basic knowledge about your company and its business, including key concepts, metrics, and processes.
  # Glossary
    Common terms and their definitions used in your business context.
```

#### Data Warehouse Introduction

You can provide a brief introduction of your data warehouse in `example/bi.yaml`, for example:

```yaml
data_warehouse_introduction: |
  # Data Warehouse Introduction
    This data warehouse is built on Presto and contains various tables related to XXXXX.
    The main fact tables include XXXX metrics, while dimension tables include XXXXX.
    The data is updated hourly and is used for reporting and analysis purposes.
```

#### Table Selection Rules

You can configure table selection rules in `example/bi.yaml`, for example:

```yaml
table_selection_extra_rule: |
  - All tables with is_valid can support both valid and invalid traffics
```

#### Custom SQL Rules

You can define your additional SQL Generation rules for tables in `example/table_info.yaml`, for example:

```yaml
sql_rule: |
  ### SQL Rules
  - All event_date in the table are stored in **UTC**. If the user specifies a timezone (e.g., CET, PST), convert between timezones accordingly.

```


### Catalog Management

#### Introduction

High-quality catalog data is essential for accurate Text2SQL generation and data analysis. OpenChatBI automatically 
discovers and indexes data warehouse table structures while providing flexible management for business metadata, column 
descriptions, and query optimization rules.

#### Catalog Structure

The catalog system organizes metadata in a hierarchical structure:

**Database Level**
- Top-level container for all tables and schemas

**Table Level**
- `description`: Business functionality and purpose of the table
- `selection_rule`: Guidelines for when and how to use this table in queries
- `sql_rule`: Specific SQL generation rules and constraints for this table

**Column Level**
- **Required Fields**: Essential metadata for each column to enable effective Text2SQL generation
  - `column_name`: Technical database column name
  - `display_name`: Human-readable name for business users
  - `alias`: Alternative names or abbreviations
  - `type`: Data type (string, integer, date, etc.)
  - `category`: Business category, dimension or metric
  - `tag`: Additional labels for filtering and organization
  - `description`: Detailed explanation of column purpose and usage
- **Two Types** of Columns
  - **Common Columns**: Columns with standardized business meanings shared across tables
  - **Table-Specific Columns**: Columns with context-dependent meanings that vary between tables
- **Derived Metrics**: Virtual metrics calculated from existing columns using SQL formulas
  - Computed dynamically during query execution rather than stored as physical columns
  - Examples: CTR (clicks/impressions), conversion rates, profit margins
  - Enable complex business calculations without pre-computing values
  
#### Loading Catalog from Database

OpenChatBI can automatically discover and load table structures from your data warehouse:

1. **Automatic Discovery**: Connects to your configured data warehouse and scans table schemas
2. **Metadata Extraction**: Extracts column names, data types, and basic structural information
3. **Incremental Updates**: Supports updating catalog data as your database schema evolves

Configure automatic catalog loading in your `config.yaml`:

```yaml
catalog_store:
  store_type: file_system
  data_path: ./catalog_data
data_warehouse_config:
  include_tables:
    - your_table_pattern
  # Leave empty to include all accessible tables
```

#### File System Catalog Store

The file system catalog store organizes metadata across multiple files for maintainability and version control:

**Core Table Information**
- `table_info.yaml`: Comprehensive table metadata organized hierarchically (database → table → information)
  - `type`: Table classification (e.g., "fact" for Fact Tables, "dimension" for Dimension Tables)
  - `description`: Business functionality and purpose
  - `selection_rule`: Usage guidelines in markdown list format (each line starts with `-`)
  - `sql_rule`: SQL generation rules in markdown header format (each rule starts with `####`)
  - `derived_metric`: Virtual metrics with calculation formulas, organized by groups:
    ```md
    #### Derived Ratio Metrics
    Click-through Rate (alias CTR): SUM(clicks) / SUM(impression)
    Conversion Rate (alias CVR): SUM(conversions) / SUM(clicks)
    ```

**Column Management**
- `table_columns.csv`: Basic column registry with schema `db_name,table_name,column_name`
- `table_spec_columns.csv`: Table-specific column metadata with full schema:
  `db_name,table_name,column_name,display_name,alias,type,category,tag,description`
- `common_columns.csv`: Shared column definitions across tables with schema:
  `column_name,display_name,alias,type,category,tag,description`

**Query Examples and Training Data**
- `table_selection_example.csv`: Table selection training examples with schema `question,selected_tables`
- `sql_example.yaml`: Query examples organized by database and table structure:
  ```yaml
  your_database:
    ad_performance: |
      Q: Show me CTR trends for the past 7 days
      A: SELECT date, SUM(clicks)/SUM(impressions) AS ctr
         FROM ad_performance
         WHERE date >= CURRENT_DATE - INTERVAL 7 DAY
         GROUP BY date
         ORDER BY date;
  ```


### Python Code Execution Configuration

OpenChatBI supports multiple execution environments for running Python code with different security and performance characteristics:

```yaml
# Python Code Execution Configuration
python_executor: local  # Options: "local", "restricted_local", "docker"
```

#### Executor Types

- **`local`** (Default)
  - **Performance**: Fastest execution
  - **Security**: Least secure (code runs in current Python process)
  - **Capabilities**: Full Python capabilities and library access
  - **Use Case**: Development environments, trusted code execution

- **`restricted_local`**
  - **Performance**: Moderate execution speed
  - **Security**: Moderate security with RestrictedPython sandboxing
  - **Capabilities**: Limited Python features (no imports, file access, etc.)
  - **Use Case**: Semi-trusted environments with controlled execution

- **`docker`**
  - **Performance**: Slower due to container overhead
  - **Security**: Highest security with complete process isolation
  - **Capabilities**: Full Python capabilities within isolated container
  - **Use Case**: Production environments, untrusted code execution
  - **Requirements**: Docker must be installed and running

#### Docker Executor Setup

For production deployments or when running untrusted code, the Docker executor provides complete isolation:

1. **Install Docker**: Download and install Docker Desktop or Docker Engine
2. **Configure executor**: Set `python_executor: docker` in your config
3. **Automatic setup**: OpenChatBI will automatically build the required Docker image
4. **Fallback behavior**: If Docker is unavailable, automatically falls back to local executor

**Docker Executor Features**:
- Pre-installed data science libraries (pandas, numpy, matplotlib, seaborn)
- Network isolation for security
- Automatic container cleanup
- Resource isolation from host system


## Development & Testing

### Code Quality Tools

The project uses modern Python tooling for code quality:

```bash
# Format code
uv run black .

# Lint code  
uv run ruff check .

# Type checking
uv run mypy openchatbi/

# Security scanning
uv run bandit -r openchatbi/
```

### Testing

Run the test suite:

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=openchatbi --cov-report=html

# Run specific test files
uv run pytest test/test_generate_sql.py
uv run pytest test/test_agent_graph.py
```

### Pre-commit Hooks

Install pre-commit hooks for automatic code quality checks:

```bash
uv run pre-commit install
```

## Contribution Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Contact & Support

- **Author**: Yu Zhong ([zhongyu8@gmail.com](mailto:zhongyu8@gmail.com))
- **Repository**: [github.com/zhongyu09/openchatbi](https://github.com/zhongyu09/openchatbi)
- **Issues**: [Report bugs and feature requests](https://github.com/zhongyu09/openchatbi/issues)