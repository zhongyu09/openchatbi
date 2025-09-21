"""System prompt templates and business configuration."""

import importlib.resources

from openchatbi import config

# Configuration from BI config
BASIC_KNOWLEDGE = config.get().bi_config.get("basic_knowledge_glossary", "")
DATA_WAREHOUSE_INTRODUCTION = config.get().bi_config.get("data_warehouse_introduction", "")
AGENT_EXTRA_TOOL_USE_RULE = config.get().bi_config.get("extra_tool_use_rule", "")
ORGANIZATION = config.get().organization


def _load_sql_dialects():
    """Auto-scan sql_dialect folder and load available dialect rules."""

    dialect_dir = importlib.resources.files("openchatbi.prompts.sql_dialect")
    _dialect_rule_dict = {}

    for item in dialect_dir.iterdir():
        if item.is_file() and item.name.endswith(".md"):
            dialect_name = item.name[:-3]
            with item.open() as f:
                prompt = f.read()
                _dialect_rule_dict[dialect_name] = prompt
    return _dialect_rule_dict


def _get_agent_prompt_template() -> str:
    """Load and configure agent prompt template."""
    with importlib.resources.files("openchatbi.prompts").joinpath("agent_prompt.md").open("r") as f:
        prompt = f.read()

    prompt = (
        prompt.replace("[organization]", ORGANIZATION)
        .replace("[basic_knowledge_glossary]", BASIC_KNOWLEDGE)
        .replace("[extra_tool_use_rule]", AGENT_EXTRA_TOOL_USE_RULE)
    )
    return prompt


def _get_extraction_prompt_template() -> str:
    """Load and configure extraction prompt template."""
    with importlib.resources.files("openchatbi.prompts").joinpath("extraction_prompt.md").open("r") as f:
        prompt = f.read()

    prompt = prompt.replace("[organization]", ORGANIZATION).replace("[basic_knowledge_glossary]", BASIC_KNOWLEDGE)
    return prompt


def _get_table_selection_prompt_template() -> str:
    """Load and configure table selection prompt template."""
    with importlib.resources.files("openchatbi.prompts").joinpath("schema_linking_prompt.md").open("r") as f:
        prompt = f.read()
    prompt = prompt.replace("[organization]", ORGANIZATION).replace("[basic_knowledge_glossary]", BASIC_KNOWLEDGE)
    return prompt


def _get_text2sql_prompt_template() -> str:
    """Load and configure text2sql prompt template."""
    with importlib.resources.files("openchatbi.prompts").joinpath("text2sql_prompt.md").open("r") as f:
        prompt = f.read()
    prompt = (
        prompt.replace("[organization]", ORGANIZATION)
        .replace("[basic_knowledge_glossary]", BASIC_KNOWLEDGE)
        .replace("[data_warehouse_introduction]", DATA_WAREHOUSE_INTRODUCTION)
    )
    return prompt


def _get_visualization_prompt_template() -> str:
    """Load visualization prompt template."""
    with importlib.resources.files("openchatbi.prompts").joinpath("visualization_prompt.md").open("r") as f:
        prompt = f.read()
    return prompt


# Supported SQL dialects
DIALECT_RULES = _load_sql_dialects()
AGENT_PROMPT_TEMPLATE = _get_agent_prompt_template()
EXTRACTION_PROMPT_TEMPLATE = _get_extraction_prompt_template()
TABLE_SELECTION_PROMPT_TEMPLATE = _get_table_selection_prompt_template()
TEXT2SQL_PROMPT_TEMPLATE = _get_text2sql_prompt_template()
VISUALIZATION_PROMPT_TEMPLATE = _get_visualization_prompt_template()


def get_text2sql_dialect_prompt_template(dialect: str) -> str:
    """Get text2sql prompt template for specific SQL dialect."""
    prompt = TEXT2SQL_PROMPT_TEMPLATE.replace("[dialect]", dialect).replace(
        "[sql_dialect_rules]", DIALECT_RULES.get(dialect, "")
    )
    return prompt
