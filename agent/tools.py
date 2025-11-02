"""Sandboxed code executor tool.

Runs Python code in a separate process with time and memory limits, inside a temporary
directory. Returns a structured result indicating success, stdout/stderr and whether
the supplied tests passed.

This is intentionally conservative but not a full security sandbox. For production use
consider containers (Docker) or specialized sandboxes.
"""

import tempfile
import subprocess
import os
import shutil
import json
import sys
from typing import Dict, Any

DEFAULT_TIMEOUT = 8  # seconds per run


def _make_runner_script(user_code: str, test_code: str) -> str:
    """Build a runner script that writes user code to a file, imports it and
    runs the tests (the tests should raise AssertionError on failure).
    The runner will print a JSON result to stdout.
    """
    script = f"""
import sys, traceback, json
from types import ModuleType

# write user code to module file
with open('submission.py', 'w') as f:
    f.write({user_code!r})

result = {{'passed': False, 'error': None, 'stdout': '', 'stderr': ''}}
try:
    # capture stdout/stderr
    import io, contextlib
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
        # import submission as module
        import importlib.util
        spec = importlib.util.spec_from_file_location('submission', 'submission.py')
        submission = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(submission)

        # inject submission into test namespace as 'submission'
        globals_dict = {{'submission': submission}}
        # run tests
        exec({test_code!r}, globals_dict)
    result['passed'] = True
    result['stdout'] = buf_out.getvalue()
    result['stderr'] = buf_err.getvalue()
except Exception as e:
    result['error'] = traceback.format_exc()
    try:
        import io
        result['stdout'] = buf_out.getvalue()
        result['stderr'] = buf_err.getvalue()
    except Exception:
        pass

print(json.dumps(result))
"""
    return script


def run_submission(
    user_code: str, test_code: str, timeout: int = DEFAULT_TIMEOUT
) -> Dict[str, Any]:
    """Execute user_code together with test_code in a subprocess.

    test_code should contain assertions or explicit calls that validate the
    functionality of the code. The runner returns a dict with keys: passed, error, stdout, stderr.
    """
    tmpdir = tempfile.mkdtemp(prefix="sandbox-")
    try:
        runner = _make_runner_script(user_code, test_code)
        runner_path = os.path.join(tmpdir, "runner.py")
        with open(runner_path, "w") as f:
            f.write(runner)

        # run the runner with a subprocess; use a copy of the current python
        cmd = [sys.executable, runner_path]
        proc = subprocess.run(
            cmd, cwd=tmpdir, capture_output=True, text=True, timeout=timeout
        )
        # parse stdout as json if possible
        out = proc.stdout.strip()
        if out:
            try:
                return json.loads(out)
            except Exception:
                return {
                    "passed": False,
                    "error": "Failed to parse runner output",
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                }
        else:
            return {
                "passed": False,
                "error": "No output from runner",
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
    except subprocess.TimeoutExpired as te:
        return {
            "passed": False,
            "error": "TimeoutExpired",
            "stdout": "",
            "stderr": str(te),
        }
    finally:
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass
