import re
from typing import Any

_HINT_BLOCK_RE = re.compile(r"<hint>\s*(.*?)\s*</hint>", re.IGNORECASE | re.DOTALL)

_FAILURE_PATTERNS = [
    "error",
    "failed",
    "failure",
    "exception",
    "traceback",
    "not found",
    "invalid",
    "permission denied",
    "assertionerror",
    "timeout",
    "redo",
    "retry",
    "try again",
    "wrong",
    "incorrect",
    "fix",
]


def flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return " ".join(parts)
    return str(content) if content is not None else ""


def render_messages(messages: list[dict[str, Any]]) -> str:
    rendered = []
    for msg in messages:
        role = str(msg.get("role", "user")).strip() or "user"
        content = flatten_content(msg.get("content")).strip()
        if content:
            rendered.append(f"[{role}]\n{content}")
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            rendered.append(f"[{role}.tool_calls]\n{tool_calls}")
    return "\n\n".join(rendered).strip()


def build_reflector_messages(
    original_messages: list[dict[str, Any]],
    failed_response_text: str,
    next_state_text: str,
    next_state_role: str = "user",
) -> list[dict[str, str]]:
    system = (
        "You are the reflection module for an agent that just failed.\n"
        "Study the original context, the failed assistant action, and the objective failure feedback.\n"
        "Reason privately, but your final answer must contain exactly one <hint>...</hint> block.\n"
        "The hint must be short, actionable, and reusable.\n"
        "Do not output the full final answer, final code patch, or raw tool arguments.\n"
        "Focus on what the agent should change next."
    )
    user = (
        "Below is the failed interaction.\n\n"
        f"## Original Conversation\n{render_messages(original_messages)}\n\n"
        f"## Failed Assistant Action\n{failed_response_text.strip()}\n\n"
        f"## Objective Failure Feedback [role: {next_state_role}]\n{next_state_text.strip()}\n\n"
        "Return only one concise corrective hint inside <hint>...</hint>."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def append_hint_to_messages(messages: list[dict[str, Any]], hint: str) -> list[dict[str, Any]]:
    cloned = [dict(m) for m in messages]
    if not cloned:
        return [{"role": "user", "content": f"[user hint]\n{hint.strip()}"}]

    target_idx = None
    for i in range(len(cloned) - 1, -1, -1):
        if cloned[i].get("role") == "user":
            target_idx = i
            break
    if target_idx is None:
        target_idx = len(cloned) - 1

    content = flatten_content(cloned[target_idx].get("content"))
    suffix = f"\n\n[user hint]\n{hint.strip()}"
    cloned[target_idx]["content"] = (content + suffix).strip()
    return cloned


def parse_hint_response(text: str) -> str:
    if not text:
        return ""
    match = _HINT_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()
    trimmed = text.strip()
    if trimmed.startswith("<hint>") and trimmed.endswith("</hint>"):
        return trimmed[6:-7].strip()
    return ""


def looks_like_failure_feedback(next_state_text: str, next_state_role: str = "user") -> bool:
    text = (next_state_text or "").strip().lower()
    if not text:
        return False
    if next_state_role == "tool":
        return any(pat in text for pat in _FAILURE_PATTERNS)
    return any(pat in text for pat in _FAILURE_PATTERNS)


def normalize_binary_rewards(rewards: list[float]) -> list[float] | None:
    if len(rewards) < 2:
        return None
    mean = sum(rewards) / len(rewards)
    var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    std = var ** 0.5
    if std <= 1e-6:
        return None
    return [(r - mean) / (std + 1e-6) for r in rewards]


def select_best_successful_trial(trials: list[dict[str, Any]]) -> dict[str, Any] | None:
    successful = [trial for trial in trials if trial.get("outcome") and trial["outcome"].success]
    if not successful:
        return None
    return max(successful, key=lambda item: (item["outcome"].score, len(item.get("hint", ""))))
