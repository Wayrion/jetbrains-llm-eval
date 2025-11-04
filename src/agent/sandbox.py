from __future__ import annotations

import os
import sys
import textwrap
import tempfile
import subprocess
import resource
import uuid
import shutil
from typing import Dict, Any, Callable


SandboxRunner = Callable[[str, str, str, int], Dict[str, Any]]


def _build_runner_script(entry_point: str) -> str:
    return textwrap.dedent(
        f"""
        import builtins, os, sys, runpy, types, importlib, time
        import signal

        # Ensure sandbox directory is importable even in isolated mode (-I)
        sys.path.insert(0, os.getcwd())

        # Block network by disabling socket
        def _blocked_socket(*args, **kwargs):
            raise RuntimeError("Network is disabled in sandbox")
        import socket
        socket.socket = _blocked_socket  # type: ignore
        socket.create_connection = _blocked_socket  # type: ignore

        # Restrict open to current working directory
        _orig_open = builtins.open
        def _safe_open(file, *args, **kwargs):
            file = os.fspath(file)
            if not os.path.abspath(file).startswith(os.getcwd()):
                raise PermissionError("Access outside sandbox is not allowed")
            return _orig_open(file, *args, **kwargs)
        builtins.open = _safe_open  # type: ignore

        # Install a simple alarm for long-running code
        signal.signal(signal.SIGALRM, lambda signum, frame: (_ for _ in ()).throw(TimeoutError("Time limit exceeded")))
        signal.alarm(int(os.environ.get("SANDBOX_TIMEOUT", "10")))

        # Import the candidate module and then run tests via runpy
        if __name__ == "__main__":
            # Load candidate module
            cand = importlib.import_module("candidate")
            # Optionally ensure entry point exists
            if not hasattr(cand, "{entry_point}"):
                raise AttributeError("Entry point '{entry_point}' not found in candidate")

            # Inject entry point symbol into test globals so tests can reference it directly
            fn = getattr(cand, "{entry_point}")
            init_globals = {{"{entry_point}": fn, "candidate": cand}}

            # Execute tests module which will raise on failure
            runpy.run_module("tests", init_globals=init_globals, run_name="__main__")
        """
    )


def _prepare_sandbox_payload(
    tmp_dir: str, candidate_code: str, tests_code: str, entry_point: str
) -> str:
    cand_path = os.path.join(tmp_dir, "candidate.py")
    tests_path = os.path.join(tmp_dir, "tests.py")
    runner_path = os.path.join(tmp_dir, "runner.py")

    with open(cand_path, "w", encoding="utf-8") as f:
        f.write(candidate_code)
    with open(tests_path, "w", encoding="utf-8") as f:
        f.write(tests_code)
    with open(runner_path, "w", encoding="utf-8") as f:
        f.write(_build_runner_script(entry_point))
    return runner_path


def run_python_with_tests(
    candidate_code: str, tests_code: str, entry_point: str, timeout_s: int = 10
) -> Dict[str, Any]:
    """
    Execute user code plus tests in an isolated subprocess and capture pass/fail and logs.
    """
    with tempfile.TemporaryDirectory(prefix="sandbox_") as tmp:
        runner_path = _prepare_sandbox_payload(
            tmp, candidate_code, tests_code, entry_point
        )

        env = os.environ.copy()
        env["PYTHONPATH"] = tmp
        env["SANDBOX_TIMEOUT"] = str(timeout_s)

        # Preexec to set rlimits
        def _limit_resources():
            # CPU seconds
            resource.setrlimit(resource.RLIMIT_CPU, (timeout_s, timeout_s))
            # Address space / memory (approx 1GB)
            mem = 1_000_000_000
            resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
            # Open files
            resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))

        proc = subprocess.Popen(
            [sys.executable, "-I", "-B", runner_path],
            cwd=tmp,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            preexec_fn=_limit_resources,
            text=True,
        )
        try:
            out, err = proc.communicate(timeout=timeout_s + 2)
        except subprocess.TimeoutExpired:
            proc.kill()
            return {
                "passed": False,
                "stdout": "",
                "stderr": "TimeoutExpired",
                "exit_code": -1,
            }
        passed = proc.returncode == 0
        return {
            "passed": passed,
            "stdout": out,
            "stderr": err,
            "exit_code": proc.returncode,
        }


def run_python_in_docker(
    candidate_code: str, tests_code: str, entry_point: str, timeout_s: int = 10
) -> Dict[str, Any]:
    """Execute code inside a temporary Python Docker container."""

    docker_bin = os.environ.get("DOCKER_BIN", "docker")
    if shutil.which(docker_bin) is None:
        return {
            "passed": False,
            "stdout": "",
            "stderr": f"Docker binary '{docker_bin}' not found",
            "exit_code": -1,
        }

    image = os.environ.get("SANDBOX_DOCKER_IMAGE", "python:3.13-slim")
    cpus = os.environ.get("SANDBOX_DOCKER_CPUS", "1")
    memory = os.environ.get("SANDBOX_DOCKER_MEMORY", "1g")
    pids_limit = os.environ.get("SANDBOX_DOCKER_PIDS", "128")
    container_name = f"sandbox_{uuid.uuid4().hex}"

    with tempfile.TemporaryDirectory(prefix="sandbox_") as tmp:
        _prepare_sandbox_payload(tmp, candidate_code, tests_code, entry_point)

        cmd = [
            docker_bin,
            "run",
            "--rm",
            "--network",
            "none",
            "--cpus",
            cpus,
            "--memory",
            memory,
            "--pids-limit",
            pids_limit,
            "--security-opt",
            "no-new-privileges",
            "--workdir",
            "/workspace",
            "--volume",
            f"{tmp}:/workspace",
            "--env",
            "PYTHONPATH=/workspace",
            "--env",
            f"SANDBOX_TIMEOUT={timeout_s}",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--name",
            container_name,
        ]

        if hasattr(os, "getuid") and hasattr(os, "getgid"):
            cmd.extend(["--user", f"{os.getuid()}:{os.getgid()}"])

        cmd.append(image)
        cmd.extend(["python", "-I", "-B", "/workspace/runner.py"])

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                text=True,
            )
        except FileNotFoundError:
            return {
                "passed": False,
                "stdout": "",
                "stderr": f"Docker binary '{docker_bin}' not found",
                "exit_code": -1,
            }

        try:
            out, err = proc.communicate(timeout=timeout_s + 5)
        except subprocess.TimeoutExpired:
            subprocess.run(
                [docker_bin, "kill", container_name],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            proc.kill()
            out, err = proc.communicate()
            return {
                "passed": False,
                "stdout": out,
                "stderr": (err or "") + "\nTimeoutExpired",
                "exit_code": -1,
            }

        passed = proc.returncode == 0
        return {
            "passed": passed,
            "stdout": out,
            "stderr": err,
            "exit_code": proc.returncode,
        }


_SANDBOX_RUNNERS: Dict[str, SandboxRunner] = {
    "process": run_python_with_tests,
    "docker": run_python_in_docker,
}


def get_sandbox_runner(mode: str) -> SandboxRunner:
    key = (mode or "").lower()
    if key not in _SANDBOX_RUNNERS:
        raise ValueError(f"Unknown sandbox mode '{mode}'")
    return _SANDBOX_RUNNERS[key]
