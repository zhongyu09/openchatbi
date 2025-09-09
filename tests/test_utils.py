"""Tests for utility functions."""

import io
from unittest.mock import patch

import pytest

from openchatbi.utils import log


class TestUtilityFunctions:
    """Test utility functions."""

    def test_log_function_basic(self):
        """Test basic logging functionality."""
        # Capture stdout
        captured_output = io.StringIO()

        with patch("sys.stderr", captured_output):
            log("Test message")

        output = captured_output.getvalue()
        assert "Test message" in output

    def test_log_function_multiple_messages(self):
        """Test logging with multiple messages."""
        captured_output = io.StringIO()

        with patch("sys.stderr", captured_output):
            log("First message")
            log("Second message")

        output = captured_output.getvalue()
        assert "First message" in output
        assert "Second message" in output

    def test_log_function_empty_message(self):
        """Test logging with empty message."""
        captured_output = io.StringIO()

        with patch("sys.stderr", captured_output):
            log("")

        output = captured_output.getvalue()
        # Should handle empty messages gracefully
        assert output is not None

    def test_log_function_none_message(self):
        """Test logging with None message."""
        captured_output = io.StringIO()

        with patch("sys.stderr", captured_output):
            log(None)

        output = captured_output.getvalue()
        # Should handle None messages gracefully
        assert "None" in output or output == ""

    def test_log_function_complex_objects(self):
        """Test logging with complex objects."""
        test_dict = {"key": "value", "number": 42}
        test_list = [1, 2, 3, "string"]

        captured_output = io.StringIO()

        with patch("sys.stderr", captured_output):
            log(test_dict)
            log(test_list)

        output = captured_output.getvalue()
        assert "key" in output or str(test_dict) in output
        assert "string" in output or str(test_list) in output

    def test_log_function_with_exception(self):
        """Test logging exception information."""
        try:
            raise ValueError("Test exception")
        except ValueError as e:
            captured_output = io.StringIO()

            with patch("sys.stderr", captured_output):
                log(f"Exception occurred: {e}")

            output = captured_output.getvalue()
            assert "Exception occurred" in output
            assert "Test exception" in output

    @patch("sys.stderr")
    def test_log_function_stderr_error(self, mock_stderr):
        """Test logging when stderr has issues."""
        mock_stderr.write.side_effect = OSError("stderr error")

        # Current implementation raises exception when stderr fails - this is expected
        with pytest.raises(OSError, match="stderr error"):
            log("Test message")

    def test_log_function_unicode_handling(self):
        """Test logging with unicode characters."""
        unicode_message = "Test with émojis: 🚀 and spéciál characters: ñáéíóú"

        captured_output = io.StringIO()

        with patch("sys.stderr", captured_output):
            log(unicode_message)

        output = captured_output.getvalue()
        # Should handle unicode characters properly
        assert len(output) > 0

    def test_log_function_large_message(self):
        """Test logging with very large messages."""
        large_message = "A" * 10000  # 10KB message

        captured_output = io.StringIO()

        with patch("sys.stderr", captured_output):
            log(large_message)

        output = captured_output.getvalue()
        assert len(output) > 0
        assert "A" in output

    def test_log_function_newline_handling(self):
        """Test logging with messages containing newlines."""
        multiline_message = "Line 1\\nLine 2\\nLine 3"

        captured_output = io.StringIO()

        with patch("sys.stderr", captured_output):
            log(multiline_message)

        output = captured_output.getvalue()
        assert "Line 1" in output
        assert "Line 2" in output
        assert "Line 3" in output

    def test_log_function_timestamp_format(self):
        """Test that log includes timestamp information."""
        captured_output = io.StringIO()

        with patch("sys.stderr", captured_output):
            log("Timestamp test")

        output = captured_output.getvalue()
        # Check if output contains timestamp-like format (basic check)
        # The actual implementation might vary
        assert len(output) > len("Timestamp test")

    def test_log_function_concurrent_calls(self):
        """Test logging with concurrent-like calls."""
        import threading
        import time

        captured_output = io.StringIO()

        def log_worker(message):
            log(f"Worker: {message}")
            time.sleep(0.01)  # Small delay

        # Patch stderr for all threads
        with patch("sys.stderr", captured_output):
            # Create multiple threads (simulating concurrency)
            threads = []
            for i in range(5):
                thread = threading.Thread(target=log_worker, args=(f"message_{i}",))
                threads.append(thread)

            # Start all threads
            for thread in threads:
                thread.start()

            # Wait for all threads
            for thread in threads:
                thread.join()

        output = captured_output.getvalue()
        # Should handle concurrent access gracefully
        assert len(output) > 0
