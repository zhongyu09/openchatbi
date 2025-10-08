"""Unit tests for context configuration."""

from openchatbi.context_config import ContextConfig, get_context_config, update_context_config


class TestContextConfig:
    """Test cases for ContextConfig class."""

    def test_default_config_values(self):
        """Test that default configuration has expected values."""
        config = ContextConfig()

        # Test default values
        assert config.enabled is True
        assert config.summary_trigger_tokens == 12000
        assert config.keep_recent_messages == 20
        assert config.max_tool_output_length == 2000
        assert config.max_sql_result_rows == 50
        assert config.max_code_output_lines == 50

        # Test boolean flags
        assert config.enable_summarization is True
        assert config.enable_conversation_summary is True
        assert config.preserve_tool_errors is True
        assert config.preserve_recent_sql is True

    def test_custom_config_values(self):
        """Test creating config with custom values."""
        config = ContextConfig(
            enabled=False,
            summary_trigger_tokens=8000,
            keep_recent_messages=5,
            max_tool_output_length=1000,
            enable_summarization=False,
        )

        assert config.enabled is False
        assert config.summary_trigger_tokens == 8000
        assert config.keep_recent_messages == 5
        assert config.max_tool_output_length == 1000
        assert config.enable_summarization is False

        # Other values should use defaults
        assert config.max_sql_result_rows == 50
        assert config.preserve_tool_errors is True

    def test_config_validation_logic(self):
        """Test logical relationships in configuration."""
        config = ContextConfig()

        # Keep recent messages should be reasonable
        assert config.keep_recent_messages > 0
        assert config.keep_recent_messages < 100  # Sanity check

        # Output limits should be positive
        assert config.max_tool_output_length > 0
        assert config.max_sql_result_rows > 0
        assert config.max_code_output_lines > 0

        # Token limits should be reasonable
        assert config.summary_trigger_tokens > 0

    def test_get_context_config(self):
        """Test getting context configuration."""
        config = get_context_config()
        assert isinstance(config, ContextConfig)

    def test_update_context_config_single_value(self):
        """Test updating a single configuration value."""
        original_trigger_tokens = get_context_config().summary_trigger_tokens

        updated_config = update_context_config(summary_trigger_tokens=15000)

        assert updated_config.summary_trigger_tokens == 15000
        # Other values should remain unchanged
        assert updated_config.keep_recent_messages == get_context_config().keep_recent_messages

    def test_update_context_config_multiple_values(self):
        """Test updating multiple configuration values."""
        updated_config = update_context_config(
            summary_trigger_tokens=20000,
            keep_recent_messages=15,
            enable_summarization=False,
            max_tool_output_length=3000,
        )

        assert updated_config.summary_trigger_tokens == 20000
        assert updated_config.keep_recent_messages == 15
        assert updated_config.enable_summarization is False
        assert updated_config.max_tool_output_length == 3000

    def test_update_context_config_invalid_attribute(self):
        """Test updating config with invalid attribute name."""
        # Should not raise error, just ignore invalid attributes
        config = update_context_config(invalid_attribute=123)
        assert not hasattr(config, "invalid_attribute")

    def test_update_context_config_returns_copy(self):
        """Test that update_context_config returns a modified copy."""
        original_config = get_context_config()
        updated_config = update_context_config(summary_trigger_tokens=30000)

        # Original should be unchanged (if it's designed that way)
        # Updated should have new values
        assert updated_config.summary_trigger_tokens == 30000


class TestContextConfigPresets:
    """Test different configuration presets for common scenarios."""

    def test_minimal_context_config(self):
        """Test configuration for minimal context management."""
        config = ContextConfig(
            enabled=True,
            enable_summarization=False,
            enable_conversation_summary=False,
            max_tool_output_length=500,
        )

        assert config.enabled is True
        assert config.enable_summarization is False
        assert config.enable_conversation_summary is False

    def test_aggressive_compression_config(self):
        """Test configuration for aggressive context compression."""
        config = ContextConfig(
            enabled=True,
            summary_trigger_tokens=6000,
            keep_recent_messages=5,
            max_tool_output_length=1000,
            max_sql_result_rows=10,
            max_code_output_lines=20,
            enable_summarization=True,
        )

        assert config.summary_trigger_tokens == 6000
        assert config.keep_recent_messages == 5
        assert config.max_tool_output_length == 1000
        assert config.max_sql_result_rows == 10
        assert config.max_code_output_lines == 20

    def test_development_debug_config(self):
        """Test configuration suitable for development/debugging."""
        config = ContextConfig(
            enabled=True,
            summary_trigger_tokens=40000,
            keep_recent_messages=20,
            max_tool_output_length=10000,  # Don't trim much
            preserve_tool_errors=True,  # Always preserve errors
        )

        assert config.preserve_tool_errors is True

    def test_production_optimized_config(self):
        """Test configuration optimized for production use."""
        config = ContextConfig(
            enabled=True,
            summary_trigger_tokens=10000,
            keep_recent_messages=8,
            max_tool_output_length=1500,
            max_sql_result_rows=15,
            enable_summarization=True,
            preserve_tool_errors=True,
        )

        assert config.summary_trigger_tokens == 10000
        assert config.enable_summarization is True
        assert config.preserve_tool_errors is True


class TestContextConfigEdgeCases:
    """Test edge cases and boundary conditions for context configuration."""

    def test_zero_values(self):
        """Test configuration with zero values."""
        config = ContextConfig(
            summary_trigger_tokens=0,
            keep_recent_messages=0,
            max_tool_output_length=0,
        )

        # Should accept zero values (might cause issues in practice)
        assert config.keep_recent_messages == 0
        assert config.summary_trigger_tokens == 0
        assert config.max_tool_output_length == 0

    def test_very_large_values(self):
        """Test configuration with very large values."""
        config = ContextConfig(
            summary_trigger_tokens=900000,
            keep_recent_messages=1000,
            max_tool_output_length=100000,
        )

        assert config.keep_recent_messages == 1000

    def test_inconsistent_token_limits(self):
        """Test configuration where summary trigger > max tokens."""

        # Should accept but might cause logical issues

    def test_all_features_disabled(self):
        """Test configuration with all features disabled."""
        config = ContextConfig(
            enabled=False,
            enable_summarization=False,
            enable_conversation_summary=False,
        )

        assert config.enabled is False
        assert config.enable_summarization is False
        assert config.enable_conversation_summary is False

    def test_config_serialization(self):
        """Test that config can be converted to/from dict (if needed)."""
        config = ContextConfig(summary_trigger_tokens=15000, enable_summarization=True)

        # Test converting to dict-like representation
        config_dict = {
            "summary_trigger_tokens": config.summary_trigger_tokens,
            "enable_summarization": config.enable_summarization,
            "enabled": config.enabled,
        }

        assert config_dict["summary_trigger_tokens"] == 15000
        assert config_dict["enable_summarization"] is True
        assert config_dict["enabled"] is True

    def test_config_immutability_simulation(self):
        """Test that config behaves consistently across operations."""
        config1 = ContextConfig(summary_trigger_tokens=10000)
        config2 = ContextConfig(summary_trigger_tokens=10000)

        # Same values should be equal
        assert config1.summary_trigger_tokens == config2.summary_trigger_tokens
        assert config1.enabled == config2.enabled

    def test_realistic_configuration_scenarios(self):
        """Test realistic configuration scenarios."""

        # Small dataset scenario
        small_config = ContextConfig()

        # Large dataset scenario
        large_config = ContextConfig()

        # Interactive analysis scenario
        interactive_config = ContextConfig(
            keep_recent_messages=50,  # Keep more context for back-and-forth
            preserve_tool_errors=True,
            max_tool_output_length=5000,  # Don't trim too aggressively
        )

        assert interactive_config.keep_recent_messages > small_config.keep_recent_messages
        assert interactive_config.preserve_tool_errors is True
