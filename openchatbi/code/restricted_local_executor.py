import sys
from io import StringIO
from RestrictedPython import compile_restricted, safe_globals, utility_builtins
from RestrictedPython.Guards import safe_builtins, safer_getattr

from openchatbi.code.executor_base import ExecutorBase


class RestrictedLocalExecutor(ExecutorBase):

    def run_code(self, code: str) -> (bool, str):
        try:
            # compile restricted code
            byte_code = compile_restricted(code, "<string>", "exec")
            if byte_code is None:
                return False, "Failed to compile restricted code"

            restricted_locals = {}
            restricted_globals = safe_globals.copy()

            # Set up restricted environment with necessary functions
            restricted_globals.update(safe_builtins)
            restricted_globals["_getattr_"] = safer_getattr
            restricted_globals["__builtins__"] = utility_builtins

            # Add variable definitions to the restricted locals
            for key, value in self._variable.items():
                restricted_locals[key] = value

            # Capture print output
            original_stdout = sys.stdout
            output_buffer = StringIO()
            sys.stdout = output_buffer

            # Use the standard print function for RestrictedPython
            restricted_globals["_print_"] = lambda *args, **kwargs: print(*args, **kwargs)

            exec(byte_code, restricted_globals, restricted_locals)
            output = output_buffer.getvalue()

            return True, output

        except Exception as e:
            return False, str(e)
        finally:
            if "original_stdout" in locals():
                sys.stdout = original_stdout
