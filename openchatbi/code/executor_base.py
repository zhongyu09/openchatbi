from typing import Any


class ExecutorBase:
    """Base class for executing python code."""

    _variable: dict

    def __init__(self, variable: dict = None):
        if variable is None:
            self._variable = {}
        else:
            self._variable = variable

    def run_code(self, code: str) -> (bool, str):
        """Execute python code."""
        raise NotImplementedError()

    def set_variable(self, key: str, value: Any) -> None:
        """Set variable."""
        self._variable[key] = value
