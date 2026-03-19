from __future__ import annotations

from typing import Any


def build_terminal_context(
    *,
    task_meta: dict[str, Any],
    env_server_url: str | None = None,
    task_timeouts: dict[str, Any] | None = None,
    run_ctx: dict[str, Any] | None = None,
    success_threshold: float = 1.0,
) -> dict[str, Any]:
    context = {
        "env_type": "terminal",
        "task_meta": task_meta,
        "success_threshold": float(success_threshold),
    }
    if env_server_url:
        context["env_server_url"] = env_server_url
    if task_timeouts:
        context["task_timeouts"] = task_timeouts
    if run_ctx:
        context["run_ctx"] = run_ctx
    return context


def build_gui_context(
    *,
    task_config: dict[str, Any],
    env_server_url: str | None = None,
    episode_id: str | None = None,
    success_threshold: float = 1.0,
    sleep_after_execution: float = 0.0,
    coordinate_type: str = "relative",
    original_width: int | None = None,
    original_height: int | None = None,
    processed_width: int | None = None,
    processed_height: int | None = None,
) -> dict[str, Any]:
    context = {
        "env_type": "gui",
        "task_config": task_config,
        "success_threshold": float(success_threshold),
        "sleep_after_execution": float(sleep_after_execution),
        "coordinate_type": coordinate_type,
    }
    if env_server_url:
        context["env_server_url"] = env_server_url
    if episode_id:
        context["episode_id"] = episode_id
    if original_width is not None:
        context["original_width"] = int(original_width)
    if original_height is not None:
        context["original_height"] = int(original_height)
    if processed_width is not None:
        context["processed_width"] = int(processed_width)
    if processed_height is not None:
        context["processed_height"] = int(processed_height)
    return context


def build_swe_context(
    *,
    image: str,
    eval_script: str,
    env_server_url: str | None = None,
    instance_id: str = "",
    cwd: str = "/testbed",
    exec_timeout: int = 180,
    eval_timeout: int = 3600,
    success_threshold: float = 1.0,
    env: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = {
        "env_type": "swe",
        "image": image,
        "instance_id": instance_id,
        "eval_script": eval_script,
        "cwd": cwd,
        "exec_timeout": int(exec_timeout),
        "eval_timeout": int(eval_timeout),
        "success_threshold": float(success_threshold),
    }
    if env_server_url:
        context["env_server_url"] = env_server_url
    if env:
        context["env"] = env
    return context


def build_toolcall_context(
    *,
    label: str,
    accept_intermediate_tool_success: bool = False,
) -> dict[str, Any]:
    return {
        "env_type": "toolcall",
        "label": label,
        "accept_intermediate_tool_success": bool(accept_intermediate_tool_success),
    }
