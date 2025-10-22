"""System prompt templates and business configuration."""

import importlib.resources

from openchatbi import config

# Global cache variables for lazy loading (only for file I/O operations)
_dialect_rules_cache = None
_agent_prompt_template_cache = None
_extraction_prompt_template_cache = None
_table_selection_prompt_template_cache = None
_text2sql_prompt_template_cache = None
_visualization_prompt_template_cache = None
_summary_prompt_template_cache = None


def get_basic_knowledge():
    """Get basic knowledge from config."""
    try:
        return config.get().bi_config.get("basic_knowledge_glossary", "")
    except ValueError:
        return ""


def get_data_warehouse_introduction():
    """Get data warehouse introduction from config."""
    try:
        return config.get().bi_config.get("data_warehouse_introduction", "")
    except ValueError:
        return ""


def get_agent_extra_tool_use_rule():
    """Get agent extra tool use rule from config."""
    try:
        return config.get().bi_config.get("extra_tool_use_rule", "")
    except ValueError:
        return ""


def get_organization():
    """Get organization from config."""
    try:
        return config.get().organization
    except ValueError:
        return "The Company"


def get_dialect_rules():
    """Get SQL dialect rules with lazy loading and caching."""
    global _dialect_rules_cache
    if _dialect_rules_cache is None:
        dialect_dir = importlib.resources.files("openchatbi.prompts.sql_dialect")
        _dialect_rules_cache = {}

        for item in dialect_dir.iterdir():
            if item.is_file() and item.name.endswith(".md"):
                dialect_name = item.name[:-3]
                with item.open() as f:
                    prompt = f.read()
                    _dialect_rules_cache[dialect_name] = prompt
    return _dialect_rules_cache


def get_agent_prompt_template() -> str:
    """Get agent prompt template with caching."""
    global _agent_prompt_template_cache
    if _agent_prompt_template_cache is None:
        with importlib.resources.files("openchatbi.prompts").joinpath("agent_prompt.md").open("r") as f:
            prompt = f.read()

        _agent_prompt_template_cache = (
            prompt.replace("[organization]", get_organization())
            .replace("[basic_knowledge_glossary]", get_basic_knowledge())
            .replace("[extra_tool_use_rule]", get_agent_extra_tool_use_rule())
        )
    return _agent_prompt_template_cache


def get_extraction_prompt_template() -> str:
    """Get extraction prompt template with caching."""
    global _extraction_prompt_template_cache
    if _extraction_prompt_template_cache is None:
        with importlib.resources.files("openchatbi.prompts").joinpath("extraction_prompt.md").open("r") as f:
            prompt = f.read()

        _extraction_prompt_template_cache = prompt.replace("[organization]", get_organization()).replace(
            "[basic_knowledge_glossary]", get_basic_knowledge()
        )
    return _extraction_prompt_template_cache


def get_table_selection_prompt_template() -> str:
    """Get table selection prompt template with caching."""
    global _table_selection_prompt_template_cache
    if _table_selection_prompt_template_cache is None:
        with importlib.resources.files("openchatbi.prompts").joinpath("schema_linking_prompt.md").open("r") as f:
            prompt = f.read()
        _table_selection_prompt_template_cache = prompt.replace("[organization]", get_organization()).replace(
            "[basic_knowledge_glossary]", get_basic_knowledge()
        )
    return _table_selection_prompt_template_cache


def get_text2sql_prompt_template() -> str:
    """Get text2sql prompt template with caching."""
    global _text2sql_prompt_template_cache
    if _text2sql_prompt_template_cache is None:
        with importlib.resources.files("openchatbi.prompts").joinpath("text2sql_prompt.md").open("r") as f:
            prompt = f.read()
        _text2sql_prompt_template_cache = (
            prompt.replace("[organization]", get_organization())
            .replace("[basic_knowledge_glossary]", get_basic_knowledge())
            .replace("[data_warehouse_introduction]", get_data_warehouse_introduction())
        )
    return _text2sql_prompt_template_cache


def get_visualization_prompt_template() -> str:
    """Get visualization prompt template with caching."""
    global _visualization_prompt_template_cache
    if _visualization_prompt_template_cache is None:
        with importlib.resources.files("openchatbi.prompts").joinpath("visualization_prompt.md").open("r") as f:
            _visualization_prompt_template_cache = f.read()
    return _visualization_prompt_template_cache


def get_summary_prompt_template() -> str:
    """Get summary prompt template with caching."""
    global _summary_prompt_template_cache
    if _summary_prompt_template_cache is None:
        with importlib.resources.files("openchatbi.prompts").joinpath("summary_prompt.md").open("r") as f:
            _summary_prompt_template_cache = f.read()
    return _summary_prompt_template_cache


def get_text2sql_dialect_prompt_template(dialect: str) -> str:
    """Get text2sql prompt template for specific SQL dialect."""
    prompt = get_text2sql_prompt_template()
    if not prompt:
        prompt = "Generate SQL query for the given question in [dialect] dialect."

    dialect_rules = get_dialect_rules()
    prompt = prompt.replace("[dialect]", dialect).replace("[sql_dialect_rules]", dialect_rules.get(dialect, ""))
    return prompt


def reset_cache():
    """Reset all cached values. Useful for testing."""
    global _dialect_rules_cache, _agent_prompt_template_cache
    global _extraction_prompt_template_cache, _table_selection_prompt_template_cache
    global _text2sql_prompt_template_cache, _visualization_prompt_template_cache
    global _summary_prompt_template_cache

    _dialect_rules_cache = None
    _agent_prompt_template_cache = None
    _extraction_prompt_template_cache = None
    _table_selection_prompt_template_cache = None
    _text2sql_prompt_template_cache = None
    _visualization_prompt_template_cache = None
    _summary_prompt_template_cache = None
