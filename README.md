# OpenChatBI

OpenChatBI is an intelligent chat-based BI tool powered by large language models, designed to help users query, analyze,
and visualize data through natural language conversations. It uses LangGraph and LangChain ecosystem to build chat agent 
and workflows that support natural language to SQL conversion and data analysis.

## Core Features

1. **Natural Language Interaction**: Get data analysis results by asking questions in natural language
2. **Automatic SQL Generation**: Convert natural language queries into SQL statements using advanced text2sql workflows
   with schema linking and well organized prompt engineering
3. **Knowledge Base Integration**: Answer complex questions by combining catalog based knowledge retrival and external
   knowledge base retrival (via MCP tools)
4. **Code Execution**: Execute Python code for data analysis and visualization
5. **Interactive Problem-Solving**: Proactively ask users for more context when information is incomplete
6. **Persistent Memory**: Conversation management and user characteristic memory based on LangGraph checkpointing
7. **MCP Support**: Integration with MCP tools by configuration
8. **Web UI Interface**: Provide 2 sample UI: simple and streaming web interfaces using Gradio, easy to integrate with
   other web applications

## Roadmap

1. **Data Visualization**: Generate intuitive charts and dashboards
2. **Time Series Forecasting**: Forecasting models deployed in-house
3. **Root Cause Analysis Algorithm**: Multi-dimensional drill-down capabilities for anomaly investigation

# Getting started

## Installation & Setup

### Prerequisites

- Python 3.11 or higher
- Access to a supported LLM provider (OpenAI, Anthropic, etc.)
- Data Warehose (Database) credentials (like Presto, Postgre SQL, MySQL, etc.)

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

graph = get_default_graph
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

## Architecture Overview

OpenChatBI is built using a modular architecture with clear separation of concerns:

1. **LangGraph Workflows**: Core orchestration using state machines for complex multi-step processes
2. **Catalog Management**: Flexible data catalog system supporting multiple storage backends
3. **Text2SQL Pipeline**: Advanced natural language to SQL conversion with schema linking
4. **Code Execution**: Sandboxed Python execution environment for data analysis
5. **Tool Integration**: Extensible tool system for human interaction and knowledge search
6. **Persistent Memory**: SQLite-based conversation state management

## Technology Stack

- **Frameworks**: LangGraph, LangChain, FastAPI, Gradio
- **Large Language Models**: Azure OpenAI (GPT-4), Anthropic Claude, OpenAI GPT models
- **Databases**: Presto, Trino, MySQL with SQLAlchemy support
- **Development**: Python 3.11+, with modern tooling (Black, Ruff, MyPy, Pytest)
- **Storage**: SQLite for conversation checkpointing, file system catalog storage

## Project Structure

```
openchatbi/
├── README.md                   # Project documentation
├── pyproject.toml              # Modern Python project configuration
├── run_tests.py                # Test runner script
├── openchatbi/                 # Core application code
│   ├── __init__.py             # Package initialization
│   ├── config.yaml             # Configuration file
│   ├── config_loader.py        # Configuration management
│   ├── constants.py            # Application constants
│   ├── agent_graph.py          # Main LangGraph workflow
│   ├── graph_state.py          # State definition for workflows
│   ├── utils.py                # Utility functions
│   ├── catalog/                # Data catalog management
│   │   ├── catalog_loader.py   # Catalog loading logic
│   │   ├── catalog_store.py    # Catalog storage interface
│   │   ├── entry.py            # Catalog entry points
│   │   ├── factory.py          # Catalog factory patterns
│   │   ├── helper.py           # Catalog helper functions
│   │   ├── schema_retrival.py  # Schema retrieval logic
│   │   └── token_service.py    # Token service integration
│   ├── code/                   # Code execution framework
│   │   ├── executor_base.py    # Base executor interface
│   │   ├── local_executor.py   # Local code execution
│   │   └── restricted_local_executor.py # Local code execution with restriction
│   ├── llm/                    # LLM integration layer
│   │   └── llm.py              # LLM management and retry logic
│   ├── prompts/                # Prompt templates and engineering
│   │   ├── agent_prompt.md     # Main agent prompts
│   │   ├── extraction_prompt.md # Information extraction prompts
│   │   ├── system_prompt.py    # System prompt management
│   │   ├── table_selection_prompt.md # Table selection prompts
│   │   └── text2sql_prompt.md  # Text-to-SQL prompts
│   ├── text2sql/               # Text-to-SQL conversion pipeline
│   │   ├── data.py             # Data and retriver for Text-to-SQL 
│   │   ├── extraction.py       # Information extraction
│   │   ├── generate_sql.py     # SQL generation logic
│   │   ├── schema_linking.py   # Schema linking process
│   │   ├── sql_graph.py        # SQL generation LangGraph workflow
│   │   └── text2sql_utils.py   # Text2SQL utilities
│   └── tool/                   # LangGraph tools and functions
│       ├── ask_human.py        # Human-in-the-loop interactions
│       ├── memory.py           # Memory management tool
│       ├── run_python_code.py  # Python code execution tool
│       ├── search_knowledge.py # Knowledge base search
│       └── mcp_tools.py        # Integrate with MCP tools
├── sample_api/                 # API implementations
│   └── async_api.py            # Asynchronous API example
└── sample_ui/                  # Web interface implementations
    ├── memory_ui.py            # Memory-enhanced UI interface
    ├── simple_ui.py            # Simple non-streaming UI based on Gradio
    ├── streaming_ui.py         # Streaming UI
    └── style.py                # UI styling and CSS
```

## Advanced Features

### Basic Knowledge & Glossary

You can define basic knowledge and glossary in `example/bi.yaml`, for example:

```yaml
basic_knowledge_glossary: |
  # Basic Knowledge Introduction
    The basic knowledge about your company and its business, including key concepts, metrics, and processes.
  # Glossary
    Common terms and their definitions used in your business context.
```

### Data Warehouse Introduction

You can provide a brief introduction of your data warehouse in `example/bi.yaml`, for example:

```yaml
data_warehouse_introduction: |
  # Data Warehouse Introduction
    This data warehouse is built on Presto and contains various tables related to XXXXX.
    The main fact tables include XXXX metrics, while dimension tables include XXXXX.
    The data is updated hourly and is used for reporting and analysis purposes.
```

### Table Selection Rules

You can configure table selection rules in `example/bi.yaml`, for example:

```yaml
table_selection_extra_rule: |
  - All tables with is_valid can support both valid and invalid traffics
```

### Custom SQL Rules

You can define SQL rules for tables in `example/table_info.yaml`, for example:

```yaml
sql_rule: |
  ### SQL Rules
  - All event_date in the table are stored in **UTC**. If the user specifies a timezone (e.g., CET, PST), convert between timezones accordingly.

```

## Development & Testing

### Code Quality Tools

The project uses modern Python tooling for code quality:

```bash
# Format code
uv run black .
uv run isort .

# Lint code  
uv run ruff check .
uv run pylint openchatbi/

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