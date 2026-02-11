"""Constrained Python execution used for tool-integrated reasoning."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class SandboxPolicy:
    timeout_sec: int = 6
    max_code_chars: int = 8_000
    max_output_chars: int = 4_000
    max_memory_mb: int = 512
    allowed_imports: tuple[str, ...] = (
        "math",
        "itertools",
        "functools",
        "fractions",
        "collections",
        "statistics",
        "sympy",
        "numpy",
    )


@dataclass(frozen=True)
class SandboxResult:
    success: bool
    stdout: str
    stderr: str
    error: str | None
    exception_type: str | None
    duration_sec: float


_BLOCKED_NAMES = {
    "open",
    "exec",
    "eval",
    "compile",
    "input",
    "__import__",
    "globals",
    "locals",
    "vars",
    "help",
    "breakpoint",
    "quit",
    "exit",
}


class CodeSafetyError(ValueError):
    """Raised when code violates sandbox policy."""


def validate_code_safety(code: str, policy: SandboxPolicy) -> None:
    """Static checks to reject unsafe code before execution."""

    if len(code) > policy.max_code_chars:
        raise CodeSafetyError("Code block too large for sandbox policy")

    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise CodeSafetyError(f"Syntax error in code: {exc}") from exc

    if sum(1 for _ in ast.walk(tree)) > 1_500:
        raise CodeSafetyError("Code block AST too large")

    for node in ast.walk(tree):
        if isinstance(node, (ast.With, ast.AsyncWith, ast.Try, ast.Raise, ast.Lambda, ast.ClassDef)):
            raise CodeSafetyError(f"Disallowed construct: {type(node).__name__}")

        if isinstance(node, ast.Name) and node.id in _BLOCKED_NAMES:
            raise CodeSafetyError(f"Blocked symbol used: {node.id}")

        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise CodeSafetyError("Dunder attribute access is blocked")

        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in policy.allowed_imports:
                    raise CodeSafetyError(f"Import blocked: {top}")

        if isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".")[0]
            if module not in policy.allowed_imports:
                raise CodeSafetyError(f"Import blocked: {module}")


def _build_runner(code: str, policy: SandboxPolicy) -> str:
    allowed = list(policy.allowed_imports)
    escaped_code = json.dumps(code)
    escaped_allowed = json.dumps(allowed)

    runner = f"""
import io
import json
import math
import traceback
from contextlib import redirect_stdout, redirect_stderr

try:
    import resource
except Exception:
    resource = None

ALLOWED_IMPORTS = set({escaped_allowed})
USER_CODE = {escaped_code}
MAX_OUTPUT_CHARS = {policy.max_output_chars}
MAX_MEMORY_MB = {policy.max_memory_mb}

if resource is not None:
    try:
        memory = MAX_MEMORY_MB * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        resource.setrlimit(resource.RLIMIT_CPU, ({policy.timeout_sec}, {policy.timeout_sec + 1}))
    except Exception:
        pass

safe_builtins = {{
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "pow": pow,
    "print": print,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}}

# Preload commonly needed math modules into globals.
globals_dict = {{"__builtins__": safe_builtins, "math": math}}

for module_name in ALLOWED_IMPORTS:
    try:
        globals_dict[module_name] = __import__(module_name)
    except Exception:
        pass

stdout_buffer = io.StringIO()
stderr_buffer = io.StringIO()
error = None
exception_type = None

try:
    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        exec(compile(USER_CODE, "<sandbox>", "exec"), globals_dict, {{}})
except Exception as exc:
    error = str(exc)
    exception_type = type(exc).__name__
    stderr_buffer.write(traceback.format_exc())

stdout = stdout_buffer.getvalue()[:MAX_OUTPUT_CHARS]
stderr = stderr_buffer.getvalue()[:MAX_OUTPUT_CHARS]

print(json.dumps({{
    "stdout": stdout,
    "stderr": stderr,
    "error": error,
    "exception_type": exception_type,
}}))
"""

    return textwrap.dedent(runner)


def execute_python(code: str, policy: SandboxPolicy | None = None) -> SandboxResult:
    """Execute code in a constrained subprocess and capture output."""

    policy = policy or SandboxPolicy()
    start = time.perf_counter()

    try:
        validate_code_safety(code, policy)
    except CodeSafetyError as exc:
        duration = time.perf_counter() - start
        return SandboxResult(
            success=False,
            stdout="",
            stderr="",
            error=str(exc),
            exception_type="CodeSafetyError",
            duration_sec=duration,
        )

    runner = _build_runner(code, policy)

    try:
        process = subprocess.run(
            [sys.executable, "-I", "-c", runner],
            check=False,
            capture_output=True,
            text=True,
            timeout=policy.timeout_sec + 1,
        )
    except subprocess.TimeoutExpired:
        duration = time.perf_counter() - start
        return SandboxResult(
            success=False,
            stdout="",
            stderr="",
            error=f"Sandbox timed out after {policy.timeout_sec}s",
            exception_type="TimeoutExpired",
            duration_sec=duration,
        )

    duration = time.perf_counter() - start

    payload = None
    stdout = process.stdout.strip()
    if stdout:
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            payload = None

    if payload is None:
        stderr_excerpt = process.stderr.strip()[: policy.max_output_chars]
        return SandboxResult(
            success=False,
            stdout=stdout[: policy.max_output_chars],
            stderr=stderr_excerpt,
            error="Sandbox output parse failure",
            exception_type="RuntimeError",
            duration_sec=duration,
        )

    error = payload.get("error")
    return SandboxResult(
        success=(error is None),
        stdout=(payload.get("stdout") or "")[: policy.max_output_chars],
        stderr=(payload.get("stderr") or "")[: policy.max_output_chars],
        error=error,
        exception_type=payload.get("exception_type"),
        duration_sec=duration,
    )


def summarize_result(result: SandboxResult) -> str:
    """Short string for logs/debug traces."""

    if result.success:
        out = result.stdout.strip().replace("\n", " ")
        return f"ok ({result.duration_sec:.2f}s): {out[:120]}"

    reason = result.error or "unknown error"
    return f"fail ({result.duration_sec:.2f}s): {reason}"
