from RestrictedPython import compile_restricted, safe_globals, utility_builtins

from openchatbi.code.executor_base import ExecutorBase


class RestrictedLocalExecutor(ExecutorBase):

    def run_code(self, code: str) -> (bool, str):
        try:
            # compile restricted code
            byte_code = compile_restricted(code, "<string>", "exec")
            restricted_locals = {}
            restricted_globals = safe_globals.copy()
            restricted_globals["_print_"] = print
            restricted_globals["__builtins__"] = utility_builtins

            exec(byte_code, restricted_globals, restricted_locals)
            result = restricted_locals.get("result")
            print("RestrictedPython result:", result)
            return True, result
        except Exception as e:
            print("Error:", e)
            return False, str(e)
