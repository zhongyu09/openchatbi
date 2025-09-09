"""Tests for ask_human tool functionality."""

import pytest
from pydantic import ValidationError

from openchatbi.tool.ask_human import AskHuman


class TestAskHuman:
    """Test AskHuman model functionality."""

    def test_ask_human_basic_initialization(self):
        """Test basic AskHuman model creation."""
        question = "What time period should I analyze?"
        options = ["Last 7 days", "Last 30 days", "Last year"]

        ask_human = AskHuman(question=question, options=options)

        assert ask_human.question == question
        assert ask_human.options == options

    def test_ask_human_empty_options(self):
        """Test AskHuman with empty options list."""
        ask_human = AskHuman(question="Simple question?", options=[])

        assert ask_human.question == "Simple question?"
        assert ask_human.options == []

    def test_ask_human_validation_error(self):
        """Test AskHuman model validation."""
        with pytest.raises(ValidationError):
            AskHuman()  # Missing required fields

        with pytest.raises(ValidationError):
            AskHuman(question="Test")  # Missing options field

    def test_ask_human_serialization(self):
        """Test AskHuman model serialization."""
        ask_human = AskHuman(question="Which analysis method?", options=["Statistical", "Machine Learning"])

        data = ask_human.model_dump()
        assert data["question"] == "Which analysis method?"
        assert data["options"] == ["Statistical", "Machine Learning"]
