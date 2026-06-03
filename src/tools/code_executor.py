"""Sandboxed code execution engine inspired by Biomni's code-as-action pattern.

Instead of only calling pre-defined tools, the agent can write and execute
arbitrary analysis code. A persistent namespace carries variables (DataFrames,
models, intermediate results) across execution steps within a research cycle.

Safety: execution runs in a restricted namespace with timeout protection.
No file-system writes outside the output directory, no network calls, no
subprocess spawning.
"""

from __future__ import annotations

import io
import signal
import traceback
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.tools.data_analysis import ToolContext


_SAFE_BUILTINS = {
    "abs", "all", "any", "bin", "bool", "dict", "dir", "divmod",
    "enumerate", "filter", "float", "format", "frozenset", "getattr",
    "hasattr", "hash", "hex", "id", "int", "isinstance", "issubclass",
    "iter", "len", "list", "map", "max", "min", "next", "oct", "ord",
    "pow", "print", "range", "repr", "reversed", "round", "set",
    "slice", "sorted", "str", "sum", "tuple", "type", "zip",
}


class CodeExecutionResult:
    __slots__ = ("stdout", "stderr", "error", "figures_saved", "variables_created")

    def __init__(self) -> None:
        self.stdout: str = ""
        self.stderr: str = ""
        self.error: str | None = None
        self.figures_saved: list[str] = []
        self.variables_created: list[str] = []


class PersistentNamespace:
    """Carries variables across code executions within a research session."""

    def __init__(self, ctx: ToolContext) -> None:
        self._ns: dict[str, Any] = {}
        self._inject_context(ctx)

    def _inject_context(self, ctx: ToolContext) -> None:
        self._ns["expression"] = ctx.expression
        self._ns["groups"] = ctx.groups
        self._ns["pathway_sets"] = ctx.pathway_sets
        self._ns["output_dir"] = ctx.output_dir

        self._ns["np"] = np
        self._ns["pd"] = pd
        self._ns["plt"] = plt
        self._ns["stats"] = stats
        self._ns["Path"] = Path

        try:
            import torch
            self._ns["torch"] = torch
            self._ns["gpu_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                self._ns["device"] = torch.device("cuda")
            else:
                self._ns["device"] = torch.device("cpu")
        except ImportError:
            self._ns["gpu_available"] = False

    @property
    def namespace(self) -> dict[str, Any]:
        return self._ns

    def list_variables(self) -> dict[str, str]:
        skip = {"np", "pd", "plt", "stats", "Path", "__builtins__"}
        out: dict[str, str] = {}
        for k, v in self._ns.items():
            if k.startswith("_") or k in skip:
                continue
            out[k] = type(v).__name__
        return out


class _Timeout:
    """Context manager for execution timeout via SIGALRM (Unix only)."""

    def __init__(self, seconds: int) -> None:
        self.seconds = seconds

    def _handler(self, signum: int, frame: Any) -> None:
        raise TimeoutError(f"Code execution exceeded {self.seconds}s limit")

    def __enter__(self) -> None:
        try:
            signal.signal(signal.SIGALRM, self._handler)
            signal.alarm(self.seconds)
        except (ValueError, AttributeError):
            pass

    def __exit__(self, *args: Any) -> None:
        try:
            signal.alarm(0)
        except (ValueError, AttributeError):
            pass


def execute_code(
    code: str,
    namespace: PersistentNamespace,
    output_dir: Path,
    timeout_s: int = 60,
) -> CodeExecutionResult:
    """Execute a code block in the persistent namespace with stdout/stderr capture."""
    result = CodeExecutionResult()
    output_dir.mkdir(parents=True, exist_ok=True)

    restricted_builtins = {
        k: v for k, v in __builtins__.items()  # type: ignore[union-attr]
        if k in _SAFE_BUILTINS
    } if isinstance(__builtins__, dict) else {
        k: getattr(__builtins__, k)
        for k in dir(__builtins__) if k in _SAFE_BUILTINS
    }
    namespace.namespace["__builtins__"] = restricted_builtins

    pre_keys = set(namespace.namespace.keys())

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    try:
        with _Timeout(timeout_s), redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            exec(compile(code, "<agent_code>", "exec"), namespace.namespace)
    except TimeoutError as e:
        result.error = str(e)
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
        result.stderr = traceback.format_exc()

    result.stdout = stdout_buf.getvalue()
    if not result.stderr:
        result.stderr = stderr_buf.getvalue()

    figs = [int(n) for n in plt.get_fignums()]
    saved: list[str] = []
    for fig_num in figs:
        fig = plt.figure(fig_num)
        path = output_dir / f"code_fig_{fig_num}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        saved.append(str(path))
    plt.close("all")
    result.figures_saved = saved

    post_keys = set(namespace.namespace.keys())
    result.variables_created = sorted(post_keys - pre_keys - {"__builtins__"})

    return result


def code_execution_tool(
    ctx: ToolContext, params: dict[str, Any], _namespace: PersistentNamespace | None = None
) -> dict[str, Any]:
    """Tool-registry-compatible wrapper for code execution."""
    code = str(params.get("code", ""))
    if not code.strip():
        return {"error": "No code provided"}

    ns = _namespace or PersistentNamespace(ctx)
    timeout = int(params.get("timeout_s", 60))

    result = execute_code(code, ns, ctx.output_dir, timeout_s=timeout)

    return {
        "stdout": result.stdout[:3000],
        "stderr": result.stderr[:1000] if result.stderr else "",
        "error": result.error,
        "figures_saved": result.figures_saved,
        "variables_created": result.variables_created,
        "namespace_state": ns.list_variables(),
    }
