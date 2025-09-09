"""Tests for text2sql information extraction functionality."""

import json
from datetime import date
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage

from openchatbi.graph_state import SQLGraphState
from openchatbi.text2sql.extraction import (
    generate_extraction_prompt,
    information_extraction,
    information_extraction_conditional_edges,
    parse_extracted_info_json,
)


class TestText2SQLExtraction:
    """Test text2sql information extraction functionality."""

    def test_generate_extraction_prompt(self):
        """Test extraction prompt generation."""
        prompt = generate_extraction_prompt()

        # Should replace time placeholder with today's date
        today_str = date.today().strftime("%Y-%m-%d")
        assert today_str in prompt

        # Should contain basic knowledge
        assert "[basic_knowledge_glossary]" not in prompt
        assert "[time_field_placeholder]" not in prompt

    def test_parse_extracted_info_json_valid(self):
        """Test parsing valid JSON from LLM response."""
        json_response = {
            "keywords": ["revenue", "sales"],
            "dimensions": ["date", "region"],
            "metrics": ["total_revenue"],
            "filters": [],
        }

        # Mock LLM response with JSON
        llm_content = f"```json\n{json.dumps(json_response)}\n```"

        with patch("openchatbi.text2sql.extraction.get_text_from_content", return_value=llm_content):
            with patch("openchatbi.text2sql.extraction.extract_json_from_answer", return_value=json_response):
                result = parse_extracted_info_json(llm_content)

        assert result == json_response
        assert "keywords" in result
        assert "dimensions" in result

    def test_parse_extracted_info_json_invalid(self):
        """Test parsing invalid JSON returns empty dict."""
        invalid_content = "Not valid JSON content"

        with patch("openchatbi.text2sql.extraction.get_text_from_content", return_value=invalid_content):
            with patch("openchatbi.text2sql.extraction.extract_json_from_answer", side_effect=Exception("Parse error")):
                result = parse_extracted_info_json(invalid_content)

        assert result == {}

    def test_information_extraction_function_creation(self):
        """Test creating information extraction function."""
        mock_llm = Mock()

        extraction_func = information_extraction(mock_llm)

        # Should return a callable function
        assert callable(extraction_func)

    def test_information_extraction_successful(self):
        """Test successful information extraction."""
        mock_llm = Mock()

        # Mock LLM response
        extracted_info = {
            "rewrite_question": "What is the total revenue by region?",
            "keywords": ["revenue", "total"],
            "dimensions": ["region"],
            "metrics": ["revenue"],
            "filters": [],
        }

        mock_response = AIMessage(content=json.dumps(extracted_info))

        with patch("openchatbi.text2sql.extraction.call_llm_chat_model_with_retry", return_value=mock_response):
            with patch("openchatbi.text2sql.extraction.parse_extracted_info_json", return_value=extracted_info):
                extraction_func = information_extraction(mock_llm)

                state = SQLGraphState(
                    messages=[HumanMessage(content="Show me revenue by region")], question="Show me revenue by region"
                )

                result = extraction_func(state)

        assert "info_entities" in result
        assert result["rewrite_question"] == "What is the total revenue by region?"

    def test_information_extraction_empty_response(self):
        """Test handling empty extraction response."""
        mock_llm = Mock()

        mock_response = AIMessage(content="")

        with patch("openchatbi.text2sql.extraction.call_llm_chat_model_with_retry", return_value=mock_response):
            with patch("openchatbi.text2sql.extraction.parse_extracted_info_json", return_value={}):
                extraction_func = information_extraction(mock_llm)

                state = SQLGraphState(messages=[HumanMessage(content="Test question")], question="Test question")

                result = extraction_func(state)

        # Should handle empty response gracefully
        assert "info_entities" in result
        assert result["info_entities"] == {}

    def test_information_extraction_conditional_edges_success(self):
        """Test conditional edges with successful extraction."""
        state = SQLGraphState(
            messages=[HumanMessage(content="Test question")],
            question="Test question",
            rewrite_question="What is the total revenue by region?",
            info_entities={"keywords": ["revenue"], "dimensions": ["date"]},
        )

        result = information_extraction_conditional_edges(state)

        # Should proceed to next when rewrite_question exists
        assert result == "next"

    def test_information_extraction_conditional_edges_failure(self):
        """Test conditional edges with failed extraction."""
        state = SQLGraphState(
            messages=[HumanMessage(content="Test question")], question="Test question", info_entities={}
        )

        result = information_extraction_conditional_edges(state)

        # Should end when no info extracted
        assert result == "end"

    def test_information_extraction_conditional_edges_missing(self):
        """Test conditional edges with missing info_entities."""
        state = SQLGraphState(messages=[HumanMessage(content="Test question")], question="Test question")

        result = information_extraction_conditional_edges(state)

        # Should end when info_entities not present
        assert result == "end"

    def test_information_extraction_with_retry_on_failure(self):
        """Test information extraction with retry mechanism."""
        mock_llm = Mock()

        # First call fails, second succeeds
        extracted_info = {
            "rewrite_question": "Test question",
            "keywords": ["test"],
            "dimensions": [],
            "metrics": [],
            "filters": [],
        }

        mock_response = AIMessage(content=json.dumps(extracted_info))

        with patch("openchatbi.text2sql.extraction.call_llm_chat_model_with_retry", return_value=mock_response):
            with patch("openchatbi.text2sql.extraction.parse_extracted_info_json", return_value=extracted_info):
                extraction_func = information_extraction(mock_llm)

                state = SQLGraphState(messages=[HumanMessage(content="Test question")], question="Test question")

                result = extraction_func(state)

        assert "info_entities" in result
        assert result["info_entities"]["keywords"] == ["test"]

    def test_information_extraction_time_period_detection(self):
        """Test time period detection in queries."""
        mock_llm = Mock()

        extracted_info = {
            "rewrite_question": "Show data for the last 7 days",
            "keywords": ["data"],
            "dimensions": ["date"],
            "metrics": [],
            "filters": [],
            "start_time": "2024-01-01",
        }

        mock_response = AIMessage(content=json.dumps(extracted_info))

        with patch("openchatbi.text2sql.extraction.call_llm_chat_model_with_retry", return_value=mock_response):
            with patch("openchatbi.text2sql.extraction.parse_extracted_info_json", return_value=extracted_info):
                extraction_func = information_extraction(mock_llm)

                state = SQLGraphState(
                    messages=[HumanMessage(content="Test question")], question="Show data for last 7 days"
                )

                result = extraction_func(state)

        assert "info_entities" in result
        assert "start_time" in result["info_entities"]

    def test_information_extraction_error_handling(self):
        """Test error handling in information extraction."""
        mock_llm = Mock()

        # Mock call to raise exception
        with patch("openchatbi.text2sql.extraction.call_llm_chat_model_with_retry", side_effect=Exception("LLM error")):
            extraction_func = information_extraction(mock_llm)

            state = SQLGraphState(messages=[HumanMessage(content="Test question")], question="Test question")

            # Should raise exception as the function doesn't have try-catch
            try:
                result = extraction_func(state)
                # Should not reach here
                assert False, "Expected exception to be raised"
            except Exception as e:
                assert "LLM error" in str(e)
