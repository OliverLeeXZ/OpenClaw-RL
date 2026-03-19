from __future__ import annotations

import importlib.util
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODULE_CACHE: dict[tuple[str, str], Any] = {}

_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_BASH_BLOCK_RE = re.compile(r"```bash\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)
_ANSWER_BOX_RE = re.compile(r"Answer:\s*\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", re.DOTALL)
_CODE_BLOCK_RE = re.compile(r"<code>(.*?)</code>", re.DOTALL)
_PYTHON_BLOCK_RE = re.compile(r"```python\s*(.*?)\s*```", re.DOTALL)


@dataclass
class ReplayOutcome:
    success: bool
    score: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


class ObjectiveEnvAdapter:
    async def validate_repair(
        self,
        *,
        session_id: str,
        turn_num: int,
        turn_data: dict[str, Any],
        next_state: dict[str, Any],
        hint: str,
        candidate: dict[str, Any],
    ) -> ReplayOutcome:
        raise NotImplementedError


def _load_module(module_name: str, file_path: Path):
    cache_key = (module_name, str(file_path.resolve()))
    cached = _MODULE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(module_name, module)
    spec.loader.exec_module(module)
    _MODULE_CACHE[cache_key] = module
    return module


def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "text" in item:
                    parts.append(str(item.get("text", "")))
        return " ".join(p for p in parts if p)
    return str(content) if content is not None else ""


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _get_replay_context(turn_data: dict[str, Any]) -> dict[str, Any]:
    context = turn_data.get("replay_context")
    return context if isinstance(context, dict) else {}


def _infer_env_type(context: dict[str, Any]) -> str:
    env_type = str(context.get("env_type", "")).strip().lower()
    if env_type:
        return env_type
    if isinstance(context.get("task_meta"), dict):
        return "terminal"
    if isinstance(context.get("task_config"), dict):
        return "gui"
    if context.get("image") or context.get("eval_script") or context.get("instance_id"):
        return "swe"
    if context.get("label") is not None:
        return "toolcall"
    if context.get("url"):
        return "http"
    return ""


def _maybe_load_json(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    try:
        loaded = json.loads(raw)
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _extract_tool_calls_from_message(message: dict[str, Any]) -> list[dict[str, Any]]:
    tool_calls: list[dict[str, Any]] = []

    raw_tool_calls = message.get("tool_calls")
    if isinstance(raw_tool_calls, list):
        for item in raw_tool_calls:
            if not isinstance(item, dict):
                continue
            fn = item.get("function") if isinstance(item.get("function"), dict) else item
            name = fn.get("name")
            arguments = fn.get("arguments", {})
            if isinstance(arguments, str):
                arguments = _maybe_load_json(arguments)
            if isinstance(name, str) and name:
                tool_calls.append({"name": name, "arguments": arguments if isinstance(arguments, dict) else {}})

    content = _flatten_content(message.get("content"))
    for match in _TOOL_CALL_BLOCK_RE.finditer(content):
        parsed = _maybe_load_json(match.group(1))
        name = parsed.get("name")
        arguments = parsed.get("arguments", {})
        if isinstance(name, str) and name:
            tool_calls.append({"name": name, "arguments": arguments if isinstance(arguments, dict) else {}})

    stripped = content.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        parsed = _maybe_load_json(stripped)
        name = parsed.get("name")
        arguments = parsed.get("arguments", {})
        if isinstance(name, str) and name:
            tool_calls.append({"name": name, "arguments": arguments if isinstance(arguments, dict) else {}})

    return tool_calls


def _extract_candidate_tool_calls(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    assistant_message = candidate.get("assistant_message")
    if isinstance(assistant_message, dict):
        return _extract_tool_calls_from_message(assistant_message)
    return _extract_tool_calls_from_message({"role": "assistant", "content": candidate.get("response_text", "")})


def _iter_assistant_messages(messages: list[dict[str, Any]] | None):
    for msg in messages or []:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            yield msg


def _extract_bash_commands(text: str) -> list[str]:
    commands = [m.group(1).strip() for m in _BASH_BLOCK_RE.finditer(text or "")]
    return [cmd for cmd in commands if cmd]


def _extract_swe_commands_from_messages(messages: list[dict[str, Any]] | None) -> list[str]:
    commands: list[str] = []
    for msg in _iter_assistant_messages(messages):
        commands.extend(_extract_bash_commands(_flatten_content(msg.get("content"))))
    return commands


def _extract_patch_from_submission(output: str) -> str:
    text = str(output or "").lstrip("\n")
    sentinel = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
    if text.startswith(sentinel):
        text = text[len(sentinel):].lstrip("\n")
    return text


def _is_valid_git_patch(patch_text: str) -> bool:
    text = str(patch_text or "").strip()
    if not text:
        return False
    if "diff --git " not in text:
        return False
    has_old = ("--- a/" in text) or ("--- /dev/null" in text)
    has_new = "+++ b/" in text
    return has_old and has_new


def _parse_toolcall_prediction(text: str) -> tuple[str | None, str]:
    if not text:
        return None, ""

    answer_match = _ANSWER_BOX_RE.search(text)
    if answer_match:
        return "answer", answer_match.group(1).strip()

    for match in _TOOL_CALL_BLOCK_RE.finditer(text):
        payload = _maybe_load_json(match.group(1))
        if payload.get("name") == "code_interpreter":
            args = payload.get("arguments", {})
            if isinstance(args, dict):
                code = str(args.get("code", "")).strip()
                if code:
                    return "code", code

    code_match = _CODE_BLOCK_RE.search(text)
    if code_match:
        code = code_match.group(1).strip()
        if code:
            return "code", code

    python_match = _PYTHON_BLOCK_RE.search(text)
    if python_match:
        code = python_match.group(1).strip()
        if code:
            return "code", code

    return None, ""


def _default_screen_size() -> tuple[int, int]:
    width = int(os.getenv("SCREEN_WIDTH", "1920"))
    height = int(os.getenv("SCREEN_HEIGHT", "1080"))
    return width, height


def _gui_adjust_coordinates(x: float, y: float, context: dict[str, Any]) -> tuple[int, int]:
    coordinate_type = str(context.get("coordinate_type", "relative")).strip().lower() or "relative"
    original_width = int(context.get("original_width") or _default_screen_size()[0])
    original_height = int(context.get("original_height") or _default_screen_size()[1])
    processed_width = int(context.get("processed_width") or original_width)
    processed_height = int(context.get("processed_height") or original_height)

    if coordinate_type in {"absolute", "qwen25"} and processed_width > 0 and processed_height > 0:
        return int(x * original_width / processed_width), int(y * original_height / processed_height)

    return int(x * original_width / 999), int(y * original_height / 999)


def _gui_tool_call_to_action(tool_call: dict[str, Any], context: dict[str, Any]) -> str | None:
    if tool_call.get("name") != "computer_use":
        return None

    args = _as_dict(tool_call.get("arguments"))
    action = str(args.get("action", "")).strip().lower()
    if not action:
        return None

    def _coord() -> tuple[int, int] | None:
        coord = args.get("coordinate")
        if not isinstance(coord, (list, tuple)) or len(coord) < 2:
            return None
        return _gui_adjust_coordinates(float(coord[0]), float(coord[1]), context)

    if action == "left_click":
        xy = _coord()
        return f"pyautogui.click({xy[0]}, {xy[1]})" if xy is not None else "pyautogui.click()"
    if action == "right_click":
        xy = _coord()
        return f"pyautogui.rightClick({xy[0]}, {xy[1]})" if xy is not None else "pyautogui.rightClick()"
    if action == "middle_click":
        xy = _coord()
        return f"pyautogui.middleClick({xy[0]}, {xy[1]})" if xy is not None else "pyautogui.middleClick()"
    if action == "double_click":
        xy = _coord()
        return f"pyautogui.doubleClick({xy[0]}, {xy[1]})" if xy is not None else "pyautogui.doubleClick()"
    if action == "triple_click":
        xy = _coord()
        return f"pyautogui.click({xy[0]}, {xy[1]}, clicks=3)" if xy is not None else "pyautogui.click(clicks=3)"
    if action == "mouse_move":
        xy = _coord()
        return f"pyautogui.moveTo({xy[0]}, {xy[1]})" if xy is not None else "pyautogui.moveTo(0, 0)"
    if action == "left_click_drag":
        xy = _coord()
        duration = float(args.get("duration", 0.5))
        return (
            f"pyautogui.dragTo({xy[0]}, {xy[1]}, duration={duration})"
            if xy is not None
            else "pyautogui.dragTo(0, 0)"
        )
    if action == "type":
        text = json.dumps(str(args.get("text", "")))
        return f"pyautogui.typewrite({text})"
    if action == "key":
        keys = args.get("keys", [])
        if not isinstance(keys, list):
            keys = [keys]
        keys = [str(k).strip() for k in keys if k is not None]
        if len(keys) > 1:
            rendered = ", ".join(json.dumps(k) for k in keys)
            return f"pyautogui.hotkey({rendered})"
        if len(keys) == 1:
            return f"pyautogui.press({json.dumps(keys[0])})"
        return None
    if action == "scroll":
        return f"pyautogui.scroll({int(args.get('pixels', 0))})"
    if action == "hscroll":
        return f"pyautogui.hscroll({int(args.get('pixels', 0))})"
    if action == "wait":
        return "WAIT"
    if action == "terminate":
        status = str(args.get("status", "success")).strip().lower()
        return "DONE" if status == "success" else "FAIL"
    return None


class NoopObjectiveEnvAdapter(ObjectiveEnvAdapter):
    _warned = False

    async def validate_repair(
        self,
        *,
        session_id: str,
        turn_num: int,
        turn_data: dict[str, Any],
        next_state: dict[str, Any],
        hint: str,
        candidate: dict[str, Any],
    ) -> ReplayOutcome:
        if not self._warned:
            logger.warning(
                "[Self-OPD] no replay backend configured for session=%s turn=%d; no training samples will be produced",
                session_id,
                turn_num,
            )
            self._warned = True
        return ReplayOutcome(success=False, score=0.0, details={"reason": "no_replay_backend"})


class HttpObjectiveEnvAdapter(ObjectiveEnvAdapter):
    def __init__(self, context: dict[str, Any]):
        self.url = str(context.get("url") or os.environ.get("OPENCLAW_SELF_OPD_ENV_URL", "")).strip()
        self.api_key = str(context.get("api_key") or os.environ.get("OPENCLAW_SELF_OPD_ENV_API_KEY", "")).strip()
        self.timeout = float(context.get("timeout") or os.environ.get("OPENCLAW_SELF_OPD_ENV_TIMEOUT", "300"))

    async def validate_repair(
        self,
        *,
        session_id: str,
        turn_num: int,
        turn_data: dict[str, Any],
        next_state: dict[str, Any],
        hint: str,
        candidate: dict[str, Any],
    ) -> ReplayOutcome:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "session_id": session_id,
            "turn_num": turn_num,
            "messages": turn_data.get("messages"),
            "tools": turn_data.get("tools"),
            "original_response_text": turn_data.get("response_text", ""),
            "next_state": next_state,
            "hint": hint,
            "candidate_response_text": candidate.get("response_text", ""),
            "candidate_message": candidate.get("assistant_message"),
            "replay_context": turn_data.get("replay_context"),
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(self.url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        success = bool(
            data.get("success")
            or data.get("passed")
            or data.get("ok")
            or float(data.get("score", 0.0)) > 0.0
        )
        score = float(data.get("score", 1.0 if success else 0.0))
        return ReplayOutcome(success=success, score=score, details=data if isinstance(data, dict) else {})


class TerminalObjectiveEnvAdapter(ObjectiveEnvAdapter):
    def __init__(self, context: dict[str, Any]):
        self.context = context

    async def validate_repair(
        self,
        *,
        session_id: str,
        turn_num: int,
        turn_data: dict[str, Any],
        next_state: dict[str, Any],
        hint: str,
        candidate: dict[str, Any],
    ) -> ReplayOutcome:
        module = _load_module(
            "_openclaw_self_opd_terminal_env_client",
            _REPO_ROOT / "terminal-rl" / "env_client.py",
        )
        TerminalEnvClient = module.TerminalEnvClient

        task_meta = _as_dict(self.context.get("task_meta"))
        task_name = str(task_meta.get("task_name", "unknown"))
        task_path = str(task_meta.get("task_path", ""))
        if not task_name or not task_path:
            return ReplayOutcome(False, 0.0, {"reason": "missing_terminal_task_meta"})

        env_url = str(
            self.context.get("env_server_url")
            or os.getenv("OPENCLAW_SELF_OPD_TERMINAL_ENV_URL")
            or os.getenv("ENV_SERVER_URL", "")
        ).strip()
        if not env_url:
            return ReplayOutcome(False, 0.0, {"reason": "missing_terminal_env_url"})

        success_threshold = float(
            self.context.get("success_threshold")
            or os.getenv("OPENCLAW_SELF_OPD_TERMINAL_SUCCESS_THRESHOLD", "1.0")
        )
        run_ctx = _as_dict(self.context.get("run_ctx"))
        run_ctx.setdefault("uid", f"{session_id}-{turn_num}")
        run_ctx.setdefault("group_index", int(turn_data.get("group_index", 0) or 0))
        run_ctx.setdefault("sample_index", int(turn_data.get("index", 0) or 0))
        run_ctx.setdefault("log_dir", str(_REPO_ROOT / "openclaw-self-opd" / "results" / "terminal_replay"))
        task_timeouts = _as_dict(self.context.get("task_timeouts"))

        client = TerminalEnvClient(env_url)
        task_key = f"{task_name}:{task_path}"
        request_id = f"self-opd:{session_id}:{turn_num}"
        lease = await client.allocate(task_key=task_key, request_id=request_id)
        lease_id = str(lease["lease_id"])

        try:
            await client.reset(
                lease_id=lease_id,
                task_meta=task_meta,
                run_ctx=run_ctx,
                task_timeouts=task_timeouts or None,
            )

            prior_tool_calls = self.context.get("prior_tool_calls")
            if isinstance(prior_tool_calls, list):
                replay_calls = [
                    {"name": str(item.get("name", "")), "arguments": _as_dict(item.get("arguments"))}
                    for item in prior_tool_calls
                    if isinstance(item, dict)
                ]
            else:
                replay_calls = []
                for message in _iter_assistant_messages(turn_data.get("messages")):
                    replay_calls.extend(_extract_tool_calls_from_message(message))

            for call in replay_calls:
                if call["name"]:
                    await client.exec_tool(lease_id, call["name"], call["arguments"])

            candidate_calls = _extract_candidate_tool_calls(candidate)
            for call in candidate_calls:
                if call["name"]:
                    await client.exec_tool(lease_id, call["name"], call["arguments"])

            score = float(await client.evaluate(lease_id))
            return ReplayOutcome(
                success=score >= success_threshold,
                score=score,
                details={
                    "env_type": "terminal",
                    "task_name": task_name,
                    "candidate_tool_calls": len(candidate_calls),
                },
            )
        finally:
            try:
                await client.close(lease_id)
            except Exception:
                logger.exception("[Self-OPD] terminal replay close failed lease=%s", lease_id)


class GuiObjectiveEnvAdapter(ObjectiveEnvAdapter):
    def __init__(self, context: dict[str, Any]):
        self.context = context

    async def validate_repair(
        self,
        *,
        session_id: str,
        turn_num: int,
        turn_data: dict[str, Any],
        next_state: dict[str, Any],
        hint: str,
        candidate: dict[str, Any],
    ) -> ReplayOutcome:
        module = _load_module(
            "_openclaw_self_opd_gui_env_client",
            _REPO_ROOT / "gui-rl" / "env_client.py",
        )
        GuiEnvClient = module.GuiEnvClient

        task_config = _as_dict(self.context.get("task_config"))
        if not task_config:
            return ReplayOutcome(False, 0.0, {"reason": "missing_gui_task_config"})

        env_url = str(
            self.context.get("env_server_url")
            or os.getenv("OPENCLAW_SELF_OPD_GUI_ENV_URL")
            or os.getenv("GUI_ENV_SERVER_URL", "")
        ).strip()
        if not env_url:
            return ReplayOutcome(False, 0.0, {"reason": "missing_gui_env_url"})

        success_threshold = float(
            self.context.get("success_threshold")
            or os.getenv("OPENCLAW_SELF_OPD_GUI_SUCCESS_THRESHOLD", "1.0")
        )
        sleep_after_execution = float(self.context.get("sleep_after_execution", 0.0))
        episode_id = str(self.context.get("episode_id") or f"{session_id}:{turn_num}")

        client = GuiEnvClient(env_url)
        lease = await client.allocate(episode_id=episode_id)
        lease_id = str(lease["lease_id"])

        try:
            await client.reset(lease_id=lease_id, task_config=task_config)

            prior_actions = self.context.get("prior_actions")
            if isinstance(prior_actions, list):
                replay_actions = [str(action) for action in prior_actions if action is not None]
            else:
                replay_actions = []
                for message in _iter_assistant_messages(turn_data.get("messages")):
                    for call in _extract_tool_calls_from_message(message):
                        action = _gui_tool_call_to_action(call, self.context)
                        if action:
                            replay_actions.append(action)

            for action in replay_actions:
                _, _, done, _ = await client.step(
                    lease_id=lease_id,
                    action=action,
                    sleep_after_execution=sleep_after_execution,
                )
                if done:
                    break

            candidate_actions = []
            for call in _extract_candidate_tool_calls(candidate):
                action = _gui_tool_call_to_action(call, self.context)
                if action:
                    candidate_actions.append(action)

            done = False
            for action in candidate_actions:
                _, _, done, _ = await client.step(
                    lease_id=lease_id,
                    action=action,
                    sleep_after_execution=sleep_after_execution,
                )
                if done:
                    break

            score = float(await client.evaluate(lease_id))
            return ReplayOutcome(
                success=score >= success_threshold,
                score=score,
                details={
                    "env_type": "gui",
                    "candidate_actions": candidate_actions,
                    "done_after_candidate": done,
                },
            )
        finally:
            try:
                await client.close(lease_id=lease_id)
            except Exception:
                logger.exception("[Self-OPD] gui replay close failed lease=%s", lease_id)


class SweObjectiveEnvAdapter(ObjectiveEnvAdapter):
    def __init__(self, context: dict[str, Any]):
        self.context = context

    async def validate_repair(
        self,
        *,
        session_id: str,
        turn_num: int,
        turn_data: dict[str, Any],
        next_state: dict[str, Any],
        hint: str,
        candidate: dict[str, Any],
    ) -> ReplayOutcome:
        module = _load_module(
            "_openclaw_self_opd_swe_env_client",
            _REPO_ROOT / "swe-rl" / "swe_env_client.py",
        )
        SweEnvClient = module.SweEnvClient

        env_url = str(
            self.context.get("env_server_url")
            or os.getenv("OPENCLAW_SELF_OPD_SWE_ENV_URL")
            or os.getenv("SWE_ENV_SERVER_URL", "")
        ).strip()
        if not env_url:
            return ReplayOutcome(False, 0.0, {"reason": "missing_swe_env_url"})

        image = str(self.context.get("image", "")).strip()
        eval_script = str(self.context.get("eval_script", "")).strip()
        if not image or not eval_script:
            return ReplayOutcome(False, 0.0, {"reason": "missing_swe_image_or_eval_script"})

        instance_id = str(self.context.get("instance_id", "")).strip()
        cwd = str(self.context.get("cwd", "/testbed"))
        exec_timeout = int(self.context.get("exec_timeout", 180))
        eval_timeout = int(self.context.get("eval_timeout", 3600))
        env = _as_dict(self.context.get("env"))
        success_threshold = float(
            self.context.get("success_threshold")
            or os.getenv("OPENCLAW_SELF_OPD_SWE_SUCCESS_THRESHOLD", "1.0")
        )

        client = SweEnvClient(env_url)
        lease = await client.allocate(image=image, instance_id=instance_id)
        lease_id = str(lease["lease_id"])

        try:
            prior_commands = self.context.get("prior_commands")
            if isinstance(prior_commands, list):
                replay_commands = [str(cmd) for cmd in prior_commands if isinstance(cmd, str) and cmd.strip()]
            else:
                replay_commands = _extract_swe_commands_from_messages(turn_data.get("messages"))

            for command in replay_commands:
                await client.exec(lease_id=lease_id, command=command, cwd=cwd, timeout=exec_timeout, env=env)

            candidate_commands = _extract_bash_commands(candidate.get("response_text", ""))
            last_output = ""
            for command in candidate_commands:
                exec_result = await client.exec(
                    lease_id=lease_id,
                    command=command,
                    cwd=cwd,
                    timeout=exec_timeout,
                    env=env,
                )
                last_output = str(exec_result.get("output", ""))

            patch = _extract_patch_from_submission(last_output)
            if not _is_valid_git_patch(patch):
                patch = await client.diff(lease_id=lease_id, cwd=cwd)

            if not _is_valid_git_patch(patch):
                return ReplayOutcome(
                    success=False,
                    score=0.0,
                    details={
                        "env_type": "swe",
                        "reason": "no_valid_patch",
                        "candidate_commands": candidate_commands,
                    },
                )

            eval_result = await client.evaluate(
                lease_id=lease_id,
                patch=patch,
                eval_script=eval_script,
                cwd=cwd,
                timeout=eval_timeout,
            )
            if isinstance(eval_result, dict):
                raw_score = float(eval_result.get("score", 1.0 if eval_result.get("resolved") else 0.0))
                success = bool(eval_result.get("resolved")) or raw_score >= success_threshold
            else:
                raw_score = float(eval_result)
                success = raw_score >= success_threshold
            return ReplayOutcome(
                success=success,
                score=raw_score,
                details={
                    "env_type": "swe",
                    "candidate_commands": candidate_commands,
                    "instance_id": instance_id,
                },
            )
        finally:
            try:
                await client.close(lease_id)
            except Exception:
                logger.exception("[Self-OPD] swe replay close failed lease=%s", lease_id)


class ToolCallObjectiveEnvAdapter(ObjectiveEnvAdapter):
    def __init__(self, context: dict[str, Any]):
        self.context = context

    async def validate_repair(
        self,
        *,
        session_id: str,
        turn_num: int,
        turn_data: dict[str, Any],
        next_state: dict[str, Any],
        hint: str,
        candidate: dict[str, Any],
    ) -> ReplayOutcome:
        label = self.context.get("label")
        if label is None:
            return ReplayOutcome(False, 0.0, {"reason": "missing_toolcall_label"})

        action, content = _parse_toolcall_prediction(candidate.get("response_text", ""))
        tool_executed = False
        tool_success = False
        tool_result_preview = ""

        if action == "code":
            module = _load_module(
                "_openclaw_self_opd_tool_sandbox",
                _REPO_ROOT / "toolcall-rl" / "tool_sandbox.py",
            )
            async with module.SEMAPHORE:
                tool_result = await module.tool_registry.execute_tool("code_interpreter", {"code": content})
            tool_executed = True
            tool_result_preview = str(tool_result)[:500]
            lower_result = str(tool_result).lower()
            tool_success = not lower_result.startswith("error:")

        from slime.rollout.rm_hub.math_dapo_utils import compute_score

        solution_str = f"{turn_data.get('prompt_text', '')}{candidate.get('response_text', '')}"
        result = compute_score(solution_str, str(label), strict_box_verify=True)
        score = float(result.get("score", -1.0))

        if action == "answer":
            success = score > 0.0
        elif bool(self.context.get("accept_intermediate_tool_success", False)):
            success = tool_success
            score = 1.0 if tool_success else 0.0
        else:
            success = False
            score = 0.0

        return ReplayOutcome(
            success=success,
            score=score,
            details={
                "env_type": "toolcall",
                "action_type": action,
                "tool_executed": tool_executed,
                "tool_success": tool_success,
                "tool_result_preview": tool_result_preview,
                "pred": result.get("pred"),
            },
        )


class AutoObjectiveEnvAdapter(ObjectiveEnvAdapter):
    async def validate_repair(
        self,
        *,
        session_id: str,
        turn_num: int,
        turn_data: dict[str, Any],
        next_state: dict[str, Any],
        hint: str,
        candidate: dict[str, Any],
    ) -> ReplayOutcome:
        context = _get_replay_context(turn_data)
        env_type = _infer_env_type(context)

        if env_type == "terminal":
            adapter: ObjectiveEnvAdapter = TerminalObjectiveEnvAdapter(context)
        elif env_type == "gui":
            adapter = GuiObjectiveEnvAdapter(context)
        elif env_type == "swe":
            adapter = SweObjectiveEnvAdapter(context)
        elif env_type == "toolcall":
            adapter = ToolCallObjectiveEnvAdapter(context)
        elif env_type == "http":
            adapter = HttpObjectiveEnvAdapter(context)
        elif os.environ.get("OPENCLAW_SELF_OPD_ENV_URL", "").strip():
            adapter = HttpObjectiveEnvAdapter(context)
        else:
            adapter = NoopObjectiveEnvAdapter()

        return await adapter.validate_repair(
            session_id=session_id,
            turn_num=turn_num,
            turn_data=turn_data,
            next_state=next_state,
            hint=hint,
            candidate=candidate,
        )


def create_env_adapter() -> ObjectiveEnvAdapter:
    return AutoObjectiveEnvAdapter()
