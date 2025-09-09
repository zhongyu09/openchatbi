"""Tool for running python code."""

from langchain.tools import tool
from pydantic import BaseModel, Field

from openchatbi.code.local_executor import LocalExecutor
from openchatbi.utils import log


class PythonCodeInput(BaseModel):
    reasoning: str = Field(description="Reason for using this run python code tool")
    code: str = Field(description="The python code to execute")


@tool("run_python_code", args_schema=PythonCodeInput, return_direct=False, infer_schema=True)
def run_python_code(reasoning: str, code: str) -> str:
    """Run python code string. Note: Only print outputs are visible, function return values will be ignored. Use print statements to see results.
    Returns:
        str: The print outputs of the python code
    """
    log(f"Run Python Code, Reasoning: {reasoning}")
    executor = LocalExecutor()
    success, output = executor.run_code(code)
    if success:
        return output
    else:
        return f"Error: {output}"
