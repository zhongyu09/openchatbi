import sys
from io import StringIO

from openchatbi.code.executor_base import ExecutorBase


class LocalExecutor(ExecutorBase):

    def run_code(self, code: str) -> str:
        safe_globals = {"__builtins__": __builtins__}
        original_stdout = sys.stdout
        output_buffer = StringIO()
        sys.stdout = output_buffer
        try:
            exec(code, safe_globals, safe_globals)
            output = output_buffer.getvalue()
            return True, output
        except Exception as e:
            return False, str(e)
        finally:
            sys.stdout = original_stdout
