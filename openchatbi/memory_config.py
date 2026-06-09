"""Configuration for memory & pattern-learning settings (mirrors context_config.py)."""

from dataclasses import dataclass


@dataclass
class MemoryConfig:
    """Configuration class for memory pattern learning and decay reranking.

    All behavior-changing flags default OFF to guarantee zero regression.
    """

    # Auto-capture of successful SQL into the LearnedSQLStore (source='auto').
    enable_pattern_memory: bool = False
    # Decay/importance reranking of langmem long-term user memory.
    enable_memory_decay_rerank: bool = False

    # Namespace for captured SQL patterns (schema-level only; never PII).
    pattern_scope: str = "global"
    # Half-life (days) controlling exponential recency decay.
    importance_decay_half_life_days: float = 30.0
    # Drop retrieved items whose composite score is below this floor.
    min_retrieval_score: float = 0.2
    # Cap on blended few-shot examples injected per query.
    max_patterns_per_query: int = 5
    # How often prune_stale may run (hours).
    prune_interval_hours: int = 24


def get_memory_config() -> MemoryConfig:
    """Get the current memory configuration.

    Loads `memory_config` from the main config system, falling back to defaults
    when the config system is unavailable or the field is unset.

    Returns:
        MemoryConfig: The current memory configuration.
    """
    try:
        from openchatbi.config_loader import ConfigLoader

        main_config = ConfigLoader().get()

        if hasattr(main_config, "memory_config") and main_config.memory_config:
            memory_config_dict = main_config.memory_config
            memory_config = MemoryConfig()
            for key, value in memory_config_dict.items():
                if hasattr(memory_config, key):
                    setattr(memory_config, key, value)
            return memory_config
    except (ImportError, ValueError, AttributeError):
        pass

    return MemoryConfig()
