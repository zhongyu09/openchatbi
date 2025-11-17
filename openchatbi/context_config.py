"""Configuration for context management settings."""

from dataclasses import dataclass

from openchatbi import config


@dataclass
class ContextConfig:
    """Configuration class for context management settings."""

    # Enable/disable context management
    enabled: bool = True

    # Token limits for triggering context management
    summary_trigger_tokens: int = 12000

    # Message retention (how many recent messages to always preserve)
    keep_recent_messages: int = 20

    # Historical tool output compression limits
    max_tool_output_length: int = 2000  # Max length for historical tool outputs
    max_sql_result_rows: int = 50  # Max rows to keep in CSV results
    max_code_output_lines: int = 50  # Max lines for code execution output

    # Conversation summarization
    enable_summarization: bool = True
    enable_conversation_summary: bool = True
    summary_max_messages: int = 50  # Max messages to include in summary context

    # Content preservation settings
    preserve_tool_errors: bool = True  # Always preserve error messages in full
    preserve_recent_sql: bool = True  # Preserve SQL content (less aggressive compression)


def get_context_config() -> ContextConfig:
    """Get the current context configuration.

    This function loads context configuration from the main config system.
    Falls back to default configuration if not available.

    Returns:
        ContextConfig: The current context configuration
    """
    try:
        main_config = config.get()

        # Check if context_config exists in the main config
        if hasattr(main_config, "context_config") and main_config.context_config:
            context_config_dict = main_config.context_config
            # Create ContextConfig from the loaded configuration
            context_config = ContextConfig()
            for key, value in context_config_dict.items():
                if hasattr(context_config, key):
                    setattr(context_config, key, value)
            return context_config
    except (ImportError, ValueError, AttributeError):
        # Fall back to default if config system is not available or configured
        pass

    return ContextConfig()


def update_context_config(**kwargs) -> ContextConfig:
    """Update context configuration with new values.

    Args:
        **kwargs: Configuration parameters to update

    Returns:
        ContextConfig: Updated configuration
    """
    config = get_context_config()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config
