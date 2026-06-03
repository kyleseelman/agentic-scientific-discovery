from __future__ import annotations

import traceback
from typing import Any

from src.agent.schemas import ExperimentPlan
from src.tools.data_analysis import ToolContext, run_tool


def execute_plan(plan: ExperimentPlan, ctx: ToolContext) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    aggregated: dict[str, Any] = {"steps": []}

    for i, step in enumerate(plan.steps):
        entry: dict[str, Any] = {
            "index": i,
            "tool": step.tool,
            "params": step.params,
            "description": step.description,
            "ok": False,
            "output": {},
            "error": None,
        }
        try:
            out = run_tool(step.tool, ctx, step.params)
            entry["ok"] = True
            entry["output"] = _make_serializable(out)
        except Exception as e:
            entry["error"] = str(e)
            entry["traceback"] = traceback.format_exc()
        trace.append(entry)
        aggregated["steps"].append({"tool": step.tool, "output": entry["output"], "ok": entry["ok"]})

    aggregated["plan"] = {
        "hypothesis_id": plan.hypothesis_id,
        "success_criteria": plan.success_criteria,
        "failure_criteria": plan.failure_criteria,
    }
    return trace, aggregated


def _make_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)
