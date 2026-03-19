# OpenClaw Self-OPD

Self-reflective training without a separate PRM model:

- Phase 1: the current policy samples multiple hindsight hints and trains the reflector with GRPO-style normalized binary rewards.
- Phase 2: the best successful hint is distilled back into the base policy through OPD on the repaired action.

## Prompt Split

- Reflector stage: a dedicated failure-analysis prompt that outputs exactly one `<hint>...</hint>` block.
- Actor stage: the original task conversation plus the selected hint appended to the latest user message.

The two stages intentionally use different prompts so the model learns a clean reflective interface instead of collapsing reflection and execution into one mode.

## Replay Backends

This method does not use a PRM/judge model. It now supports four in-repo replay backends directly:

- `terminal`: replays prior tool calls with [terminal-rl/env_client.py](/Users/lxxzzz/Documents/Reseach_Map/RL_distillation/OpenClaw-RL/terminal-rl/env_client.py)
- `gui`: replays UI actions with [gui-rl/env_client.py](/Users/lxxzzz/Documents/Reseach_Map/RL_distillation/OpenClaw-RL/gui-rl/env_client.py)
- `swe`: replays bash steps with [swe-rl/swe_env_client.py](/Users/lxxzzz/Documents/Reseach_Map/RL_distillation/OpenClaw-RL/swe-rl/swe_env_client.py)
- `toolcall`: replays python-tool steps with [toolcall-rl/tool_sandbox.py](/Users/lxxzzz/Documents/Reseach_Map/RL_distillation/OpenClaw-RL/toolcall-rl/tool_sandbox.py)

If needed, the old external HTTP replay endpoint is still available via `env_type=http`.

## Request Context

Each chat request may attach a `self_opd_context` field. The server strips it before forwarding to SGLang, but stores it for replay.

Example terminal payload:

```json
{
  "messages": [...],
  "tools": [...],
  "self_opd_context": {
    "env_type": "terminal",
    "task_meta": {
      "task_name": "fix_tests",
      "task_path": "python/foo",
      "instruction": "Make tests pass"
    }
  }
}
```

Example gui payload:

```json
{
  "messages": [...],
  "tools": [...],
  "self_opd_context": {
    "env_type": "gui",
    "task_config": {...},
    "coordinate_type": "relative",
    "processed_width": 1000,
    "processed_height": 1000
  }
}
```

Example swe payload:

```json
{
  "messages": [...],
  "self_opd_context": {
    "env_type": "swe",
    "image": "sweb.eval.x86_64.my_image",
    "instance_id": "astropy__astropy-12907",
    "eval_script": "python -m pytest -q"
  }
}
```

Example toolcall payload:

```json
{
  "messages": [...],
  "self_opd_context": {
    "env_type": "toolcall",
    "label": "42"
  }
}
```

Helper builders live in [context_builders.py](/Users/lxxzzz/Documents/Reseach_Map/RL_distillation/OpenClaw-RL/openclaw-self-opd/context_builders.py).

## Run

```bash
cd slime
bash ../openclaw-self-opd/run_qwen3_4b_openclaw_self_opd.sh
```
