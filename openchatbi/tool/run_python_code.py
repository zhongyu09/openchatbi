"""Tool for running python code."""

from langchain.tools import tool
from pydantic import BaseModel, Field

from openchatbi.code.local_executor import LocalExecutor
from openchatbi.code.restricted_local_executor import RestrictedLocalExecutor
from openchatbi.code.docker_executor import DockerExecutor, check_docker_status
from openchatbi.config_loader import ConfigLoader
from openchatbi.utils import log


class PythonCodeInput(BaseModel):
    reasoning: str = Field(description="Reason for using this run python code tool")
    code: str = Field(description="The python code to execute")


def _create_executor():
    """Create appropriate executor based on configuration."""
    config_loader = ConfigLoader()
    config = config_loader.get()
    executor_type = config.python_executor.lower()

    log(f"Creating executor of type: {executor_type}")

    if executor_type == "docker":
        # Check if Docker is available before creating DockerExecutor
        is_available, status_message = check_docker_status()
        if not is_available:
            log(f"Docker is not available ({status_message}), falling back to LocalExecutor")
            return LocalExecutor()
        log("Docker is available, creating DockerExecutor")
        return DockerExecutor()
    elif executor_type == "restricted_local":
        log("Creating RestrictedLocalExecutor")
        return RestrictedLocalExecutor()
    elif executor_type == "local":
        log("Creating LocalExecutor")
        return LocalExecutor()
    else:
        log(f"Unknown executor type '{executor_type}', using LocalExecutor as fallback")
        return LocalExecutor()


@tool("run_python_code", args_schema=PythonCodeInput, return_direct=False, infer_schema=True)
def run_python_code(reasoning: str, code: str) -> str:
    """Run python code string. Note: Only print outputs are visible, function return values will be ignored. Use print statements to see results.
    Returns:
        str: The print outputs of the python code
    """
    log(f"Run Python Code, Reasoning: {reasoning}")

    try:
        executor = _create_executor()
        log(f"Using {executor.__class__.__name__} for code execution")
        success, output = executor.run_code(code)
        if success:
            return output
        else:
            return f"Error: {output}"
    except Exception as e:
        log(f"Failed to create executor: {e}")
        # Fallback to LocalExecutor if configuration fails
        log("Falling back to LocalExecutor")
        executor = LocalExecutor()
        success, output = executor.run_code(code)
        if success:
            return output
        else:
            return f"Error: {output}"
