"""Tests for run_python_code tool functionality."""

from unittest.mock import patch

from openchatbi.tool.run_python_code import run_python_code


class TestRunPythonCode:
    """Test run_python_code tool functionality."""

    def test_run_python_code_basic(self):
        """Test basic Python code execution."""
        reasoning = "Testing basic print functionality"
        code = "print('Hello, World!')"

        with patch("openchatbi.tool.run_python_code.LocalExecutor") as mock_executor:
            mock_instance = mock_executor.return_value
            mock_instance.run_code.return_value = (True, "Hello, World!\n")

            result = run_python_code.run({"reasoning": reasoning, "code": code})

            assert isinstance(result, str)
            assert "Hello, World!" in result

    def test_run_python_code_with_variables(self):
        """Test Python code execution with variables."""
        reasoning = "Testing variable operations"
        code = """
x = 10
y = 20
result = x + y
print(f"Result: {result}")
"""

        with patch("openchatbi.tool.run_python_code.LocalExecutor") as mock_executor:
            mock_instance = mock_executor.return_value
            mock_instance.run_code.return_value = (True, "Result: 30\n")

            result = run_python_code.run({"reasoning": reasoning, "code": code})

            assert isinstance(result, str)
            assert "Result: 30" in result

    def test_run_python_code_data_analysis(self):
        """Test Python code for data analysis operations."""
        reasoning = "Performing data analysis calculations"
        code = """
import math
data = [1, 2, 3, 4, 5]
mean = sum(data) / len(data)
print(f"Mean: {mean}")
"""

        with patch("openchatbi.tool.run_python_code.LocalExecutor") as mock_executor:
            mock_instance = mock_executor.return_value
            mock_instance.run_code.return_value = (True, "Mean: 3.0\n")

            result = run_python_code.run({"reasoning": reasoning, "code": code})

            assert isinstance(result, str)
            assert "Mean: 3.0" in result

    def test_run_python_code_matplotlib_plot(self):
        """Test Python code for creating plots."""
        reasoning = "Creating a matplotlib visualization"
        code = """
import matplotlib.pyplot as plt
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
plt.plot(x, y)
plt.title('Sample Plot')
print('Plot created successfully')
"""

        with patch("openchatbi.tool.run_python_code.LocalExecutor") as mock_executor:
            mock_instance = mock_executor.return_value
            mock_instance.run_code.return_value = (True, "Plot created successfully\n")

            result = run_python_code.run({"reasoning": reasoning, "code": code})

            assert isinstance(result, str)
            assert "Plot created successfully" in result

    def test_run_python_code_syntax_error(self):
        """Test Python code execution with syntax errors."""
        reasoning = "Testing error handling for syntax errors"
        code = "print('Hello World'"  # Missing closing parenthesis

        with patch("openchatbi.tool.run_python_code.LocalExecutor") as mock_executor:
            mock_instance = mock_executor.return_value
            mock_instance.run_code.return_value = (False, "SyntaxError: unexpected EOF while parsing")

            result = run_python_code.run({"reasoning": reasoning, "code": code})

            assert isinstance(result, str)
            assert "Error:" in result
            assert "SyntaxError" in result

    def test_run_python_code_runtime_error(self):
        """Test Python code execution with runtime errors."""
        reasoning = "Testing error handling for runtime errors"
        code = """
x = 10
y = 0
result = x / y
print(result)
"""

        with patch("openchatbi.tool.run_python_code.LocalExecutor") as mock_executor:
            mock_instance = mock_executor.return_value
            mock_instance.run_code.return_value = (False, "ZeroDivisionError: division by zero")

            result = run_python_code.run({"reasoning": reasoning, "code": code})

            assert isinstance(result, str)
            assert "Error:" in result
            assert "ZeroDivisionError" in result

    def test_run_python_code_import_error(self):
        """Test Python code execution with import errors."""
        reasoning = "Testing error handling for import errors"
        code = """
import nonexistent_module
print('This should not print')
"""

        with patch("openchatbi.tool.run_python_code.LocalExecutor") as mock_executor:
            mock_instance = mock_executor.return_value
            mock_instance.run_code.return_value = (False, "ModuleNotFoundError: No module named 'nonexistent_module'")

            result = run_python_code.run({"reasoning": reasoning, "code": code})

            assert isinstance(result, str)
            assert "Error:" in result
            assert "ModuleNotFoundError" in result

    def test_run_python_code_multiline_output(self):
        """Test Python code with multiple print statements."""
        reasoning = "Testing multiple output lines"
        code = """
for i in range(3):
    print(f"Line {i + 1}")
"""

        with patch("openchatbi.tool.run_python_code.LocalExecutor") as mock_executor:
            mock_instance = mock_executor.return_value
            mock_instance.run_code.return_value = (True, "Line 1\nLine 2\nLine 3\n")

            result = run_python_code.run({"reasoning": reasoning, "code": code})

            assert isinstance(result, str)
            assert "Line 1" in result
            assert "Line 2" in result
            assert "Line 3" in result

    def test_run_python_code_with_sql_data(self):
        """Test Python code working with SQL-like data."""
        reasoning = "Processing SQL query results"
        code = """
data = [
    {'name': 'Alice', 'age': 30, 'salary': 50000},
    {'name': 'Bob', 'age': 25, 'salary': 45000},
    {'name': 'Charlie', 'age': 35, 'salary': 55000}
]
total_salary = sum(row['salary'] for row in data)
print(f"Total salary: ${total_salary}")
"""

        with patch("openchatbi.tool.run_python_code.LocalExecutor") as mock_executor:
            mock_instance = mock_executor.return_value
            mock_instance.run_code.return_value = (True, "Total salary: $150000\n")

            result = run_python_code.run({"reasoning": reasoning, "code": code})

            assert isinstance(result, str)
            assert "Total salary: $150000" in result

    def test_run_python_code_empty_code(self):
        """Test Python code execution with empty code."""
        reasoning = "Testing empty code handling"
        code = ""

        with patch("openchatbi.tool.run_python_code.LocalExecutor") as mock_executor:
            mock_instance = mock_executor.return_value
            mock_instance.run_code.return_value = (True, "")

            result = run_python_code.run({"reasoning": reasoning, "code": code})

            assert isinstance(result, str)
            assert result == ""

    def test_run_python_code_whitespace_only(self):
        """Test Python code execution with whitespace only."""
        reasoning = "Testing whitespace-only code"
        code = "   \n  \t  \n   "

        with patch("openchatbi.tool.run_python_code.LocalExecutor") as mock_executor:
            mock_instance = mock_executor.return_value
            mock_instance.run_code.return_value = (True, "")

            result = run_python_code.run({"reasoning": reasoning, "code": code})

            assert isinstance(result, str)
            assert result == ""

    def test_run_python_code_with_comments(self):
        """Test Python code execution with comments."""
        reasoning = "Testing code with comments"
        code = """
# This is a comment
x = 5  # Another comment
print(f"Value: {x}")  # Final comment
"""

        with patch("openchatbi.tool.run_python_code.LocalExecutor") as mock_executor:
            mock_instance = mock_executor.return_value
            mock_instance.run_code.return_value = (True, "Value: 5\n")

            result = run_python_code.run({"reasoning": reasoning, "code": code})

            assert isinstance(result, str)
            assert "Value: 5" in result

    def test_run_python_code_security_restrictions(self):
        """Test Python code with potentially restricted operations."""
        reasoning = "Testing security restrictions"
        code = """
# Attempting file operations
try:
    with open('/etc/passwd', 'r') as f:
        content = f.read()
        print("File read successfully")
except Exception as e:
    print(f"Security restriction: {e}")
"""

        with patch("openchatbi.tool.run_python_code.LocalExecutor") as mock_executor:
            mock_instance = mock_executor.return_value
            mock_instance.run_code.return_value = (
                False,
                "PermissionError: [Errno 13] Permission denied: '/etc/passwd'",
            )

            result = run_python_code.run({"reasoning": reasoning, "code": code})

            assert isinstance(result, str)
            assert "Error:" in result

    def test_run_python_code_timeout_handling(self):
        """Test Python code execution timeout scenarios."""
        reasoning = "Testing timeout handling"
        code = """
import time
time.sleep(10)  # Long running operation
print("This might timeout")
"""

        with patch("openchatbi.tool.run_python_code.LocalExecutor") as mock_executor:
            mock_instance = mock_executor.return_value
            mock_instance.run_code.return_value = (False, "TimeoutError: Code execution timed out")

            result = run_python_code.run({"reasoning": reasoning, "code": code})

            assert isinstance(result, str)
            assert "Error:" in result
            assert "TimeoutError" in result

    def test_run_python_code_memory_limit(self):
        """Test Python code execution with memory limitations."""
        reasoning = "Testing memory limit handling"
        code = """
# Creating a large list that might exceed memory limits
large_list = [0] * (10**8)
print(f"Created list with {len(large_list)} elements")
"""

        with patch("openchatbi.tool.run_python_code.LocalExecutor") as mock_executor:
            mock_instance = mock_executor.return_value
            mock_instance.run_code.return_value = (False, "MemoryError: Unable to allocate memory")

            result = run_python_code.run({"reasoning": reasoning, "code": code})

            assert isinstance(result, str)
            assert "Error:" in result
            assert "MemoryError" in result

    def test_run_python_code_return_values(self):
        """Test that return values are not captured (only prints)."""
        reasoning = "Testing return value handling"
        code = """
def calculate():
    return 42

result = calculate()
print(f"Function returned: {result}")
# The return value itself should not be captured
calculate()
"""

        with patch("openchatbi.tool.run_python_code.LocalExecutor") as mock_executor:
            mock_instance = mock_executor.return_value
            mock_instance.run_code.return_value = (True, "Function returned: 42\n")

            result = run_python_code.run({"reasoning": reasoning, "code": code})

            assert isinstance(result, str)
            assert "Function returned: 42" in result
            # Should not contain the raw return value
            assert result.strip() == "Function returned: 42"

    def test_run_python_code_exception_details(self):
        """Test detailed exception information."""
        reasoning = "Testing detailed exception handling"
        code = """
def faulty_function():
    raise ValueError("This is a custom error message")

faulty_function()
"""

        with patch("openchatbi.tool.run_python_code.LocalExecutor") as mock_executor:
            mock_instance = mock_executor.return_value
            mock_instance.run_code.return_value = (False, "ValueError: This is a custom error message")

            result = run_python_code.run({"reasoning": reasoning, "code": code})

            assert isinstance(result, str)
            assert "Error:" in result
            assert "ValueError" in result
            assert "custom error message" in result

    def test_run_python_code_executor_selection(self):
        """Test that LocalExecutor is properly instantiated and used."""
        reasoning = "Testing executor instantiation"
        code = "print('Executor test')"

        with patch("openchatbi.tool.run_python_code.LocalExecutor") as mock_executor_class:
            mock_instance = mock_executor_class.return_value
            mock_instance.run_code.return_value = (True, "Executor test\n")

            result = run_python_code.run({"reasoning": reasoning, "code": code})

            # Verify LocalExecutor was instantiated
            mock_executor_class.assert_called_once()
            # Verify run_code was called with the correct code
            mock_instance.run_code.assert_called_once_with(code)

            assert isinstance(result, str)
            assert "Executor test" in result
