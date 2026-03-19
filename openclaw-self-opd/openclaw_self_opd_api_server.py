from __future__ import annotations

import asyncio
import logging
import os
from itertools import count
from typing import Any

import httpx
import torch
from fastapi import HTTPException

from env_adapter import ReplayOutcome, create_env_adapter
from openclaw_opd_api_server import (
    OpenClawOPDAPIServer,
    _NON_STANDARD_BODY_KEYS,
    _CYAN,
    _GREEN,
    _RED,
    _RESET,
    _YELLOW,
    _extract_logprobs_from_chat_response,
    _flatten_message_content,
    _normalize_messages_for_template,
    generate,
    reward_func,
)
from reflect_utils import (
    append_hint_to_messages,
    build_reflector_messages,
    looks_like_failure_feedback,
    normalize_binary_rewards,
    parse_hint_response,
    select_best_successful_trial,
)
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

_SELF_OPD_BODY_KEYS = {"self_opd_context", "replay_context"}


class OpenClawSelfOPDAPIServer(OpenClawOPDAPIServer):
    """Self-reflective GRPO + OPD training without a separate PRM model."""

    def __init__(self, args, output_queue, submission_enabled):
        super().__init__(args=args, output_queue=output_queue, submission_enabled=submission_enabled)
        self._prm_enabled = False
        self.sglang_generate_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
        self._env_adapter = create_env_adapter()
        self._hint_group_size = int(os.getenv("OPENCLAW_SELF_OPD_GROUP_SIZE", "4"))
        self._hint_temperature = float(os.getenv("OPENCLAW_SELF_OPD_REFLECT_TEMPERATURE", "0.8"))
        self._hint_max_tokens = int(os.getenv("OPENCLAW_SELF_OPD_REFLECT_MAX_NEW_TOKENS", "512"))
        self._repair_temperature = float(os.getenv("OPENCLAW_SELF_OPD_REPAIR_TEMPERATURE", "0.6"))
        self._repair_max_tokens = int(os.getenv("OPENCLAW_SELF_OPD_REPAIR_MAX_NEW_TOKENS", "8192"))
        self._logprob_semaphore = asyncio.Semaphore(
            max(1, int(os.getenv("OPENCLAW_SELF_OPD_LOGPROB_MAX_CONCURRENCY", "3")))
        )
        self._force_all_next_state = os.getenv("OPENCLAW_SELF_OPD_FORCE_ALL_NEXT_STATE", "0") == "1"
        self._sample_group_counter = count(0)
        self._output_group_counter = count(0)

    async def _policy_chat_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        body = {
            "model": self.served_model_name,
            "messages": messages,
            "stream": False,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "logprobs": True,
            "top_logprobs": 1,
        }
        if tools:
            body["tools"] = tools

        async with httpx.AsyncClient(timeout=None) as client:
            resp = await client.post(self.sglang_chat_url, json=body)
            if resp.status_code != 200:
                logger.error("[Self-OPD] policy chat returned %d: %s", resp.status_code, resp.text[:1000])
                resp.raise_for_status()
            output = resp.json()

        choice = output.get("choices", [{}])[0]
        assistant_msg = choice.get("message", {})
        prompt_text, response_text, prompt_ids, response_ids = self._tokenize_turn(
            messages=messages,
            assistant_msg=assistant_msg,
            tools=tools,
        )
        response_logprobs = _extract_logprobs_from_chat_response(choice)
        if len(response_logprobs) > len(response_ids):
            response_logprobs = response_logprobs[: len(response_ids)]
        elif len(response_logprobs) < len(response_ids):
            response_logprobs = response_logprobs + [0.0] * (len(response_ids) - len(response_logprobs))

        return {
            "output": output,
            "choice": choice,
            "assistant_message": assistant_msg,
            "prompt_text": prompt_text,
            "response_text": response_text,
            "prompt_ids": prompt_ids,
            "response_ids": response_ids,
            "response_logprobs": response_logprobs,
            "reasoning_text": assistant_msg.get("reasoning_content", "") or "",
        }

    def _tokenize_turn(
        self,
        *,
        messages: list[dict[str, Any]],
        assistant_msg: dict[str, Any],
        tools: list[dict[str, Any]] | None,
    ) -> tuple[str, str, list[int], list[int]]:
        response_msg = dict(assistant_msg)
        if response_msg.get("content") is None:
            response_msg["content"] = ""

        norm_msgs = _normalize_messages_for_template(messages)
        norm_resp = _normalize_messages_for_template([response_msg])[0]
        full_norm = norm_msgs + [norm_resp]

        prompt_text = self.tokenizer.apply_chat_template(
            norm_msgs,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = self.tokenizer.apply_chat_template(
            full_norm,
            tools=tools,
            tokenize=False,
            add_generation_prompt=False,
        )
        response_text = full_text[len(prompt_text):] if full_text.startswith(prompt_text) else full_text
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        response_ids = self.tokenizer(response_text, add_special_tokens=False)["input_ids"]
        return prompt_text, response_text, prompt_ids, response_ids

    async def _handle_request(
        self,
        body: dict[str, Any],
        session_id: str,
        turn_type: str,
        session_done: bool,
    ) -> dict[str, Any]:
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=400, detail="messages must be a non-empty list")

        tools = body.get("tools")
        replay_context = body.get("self_opd_context")
        if not isinstance(replay_context, dict):
            replay_context = body.get("replay_context")
        if not isinstance(replay_context, dict):
            replay_context = None

        forward_body = {
            k: v
            for k, v in body.items()
            if k not in _NON_STANDARD_BODY_KEYS and k not in _SELF_OPD_BODY_KEYS
        }
        forward_body["stream"] = False
        forward_body.pop("stream_options", None)
        forward_body["logprobs"] = True
        forward_body["top_logprobs"] = 1
        if "model" not in forward_body:
            forward_body["model"] = self.served_model_name

        async with httpx.AsyncClient(timeout=None) as client:
            sglang_resp = await client.post(self.sglang_chat_url, json=forward_body)
            if sglang_resp.status_code != 200:
                logger.error("[Self-OPD] SGLang returned %d: %s", sglang_resp.status_code, sglang_resp.text[:1000])
                sglang_resp.raise_for_status()
            output = sglang_resp.json()

        choice = output.get("choices", [{}])[0]
        assistant_msg = choice.get("message", {})
        tool_calls = assistant_msg.get("tool_calls") or []
        content = assistant_msg.get("content") or ""
        reasoning = assistant_msg.get("reasoning_content") or ""
        logger.info(
            "%s[Self-OPD] [%s] session=%s prompt_msgs=%d%s",
            _YELLOW,
            turn_type,
            session_id,
            len(messages),
            _RESET,
        )
        logger.info(
            "%s[Self-OPD] [%s] session=%s thinking=%d chars, response:\n%s%s",
            _RED,
            turn_type,
            session_id,
            len(reasoning),
            content,
            _RESET,
        )
        if tool_calls:
            logger.info("[Self-OPD] tool_calls: %s", str(tool_calls)[:500])

        if turn_type == "main":
            prev_turn_num = self._turn_counts.get(session_id, 0)
            if prev_turn_num > 0 and messages:
                self._flush_pending_record(session_id, messages[-1])
                prev_turn_data = self._pending_turn_data.get(session_id, {}).get(prev_turn_num)
                if prev_turn_data is not None:
                    self._fire_opd_task(session_id, prev_turn_num, prev_turn_data, messages[-1])

            response_msg = dict(assistant_msg)
            if response_msg.get("content") is None:
                response_msg["content"] = ""
            norm_msgs = _normalize_messages_for_template(messages)
            norm_resp = _normalize_messages_for_template([response_msg])[0]
            full_norm = norm_msgs + [norm_resp]

            prompt_text = self.tokenizer.apply_chat_template(
                norm_msgs,
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = self.tokenizer.apply_chat_template(
                full_norm,
                tools=tools,
                tokenize=False,
                add_generation_prompt=False,
            )
            response_text = full_text[len(prompt_text):] if full_text.startswith(prompt_text) else full_text
            prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            response_ids = self.tokenizer(response_text, add_special_tokens=False)["input_ids"]

            if not response_ids and not response_text.strip():
                logger.info("[Self-OPD] MAIN session=%s -> empty response, skipping", session_id)
                output["session_id"] = session_id
                return {"response": output}

            response_logprobs = _extract_logprobs_from_chat_response(choice)
            if len(response_logprobs) > len(response_ids):
                response_logprobs = response_logprobs[: len(response_ids)]
            elif len(response_logprobs) < len(response_ids):
                response_logprobs = response_logprobs + [0.0] * (len(response_ids) - len(response_logprobs))

            self._turn_counts[session_id] = prev_turn_num + 1
            turn_num = self._turn_counts[session_id]
            turn_data = {
                "prompt_ids": prompt_ids,
                "response_ids": response_ids,
                "response_logprobs": response_logprobs,
                "prompt_text": prompt_text,
                "response_text": response_text,
                "messages": messages,
                "tools": tools,
                "has_next_state": False,
                "replay_context": replay_context,
            }
            self._pending_turn_data.setdefault(session_id, {})[turn_num] = turn_data
            self._buffer_record(session_id, turn_num, messages, prompt_text, response_text, tool_calls)
            logger.info(
                "[Self-OPD] MAIN session=%s turn=%d prompt_tokens=%d response_tokens=%d",
                session_id,
                turn_num,
                len(prompt_ids),
                len(response_ids),
            )
            self._maybe_submit_ready_samples(session_id)
        else:
            logger.info("[Self-OPD] SIDE session=%s -> skipped (no training data)", session_id)

        if session_done:
            self._flush_pending_record(session_id, None)
            self._maybe_submit_ready_samples(session_id, force_drop_without_next_state=True)
            self._turn_counts.pop(session_id, None)
            logger.info("[Self-OPD] session=%s done -> cleaned up", session_id)

        output["session_id"] = session_id
        return {"response": output}

    async def _compute_policy_log_probs(self, prompt_text: str, response_text: str) -> list[float]:
        response_ids = self.tokenizer(response_text, add_special_tokens=False)["input_ids"]
        if not response_ids:
            return []
        full_ids = self.tokenizer(prompt_text + response_text, add_special_tokens=False)["input_ids"]
        start_len = max(0, len(full_ids) - len(response_ids))
        payload = {
            "input_ids": full_ids,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": 0,
                "skip_special_tokens": False,
            },
            "return_logprob": True,
            "logprob_start_len": start_len,
        }
        async with self._logprob_semaphore:
            async with httpx.AsyncClient(timeout=None) as client:
                resp = await client.post(self.sglang_generate_url, json=payload)
                resp.raise_for_status()
                result = resp.json()

        meta = result.get("meta_info", {}) if isinstance(result, dict) else {}
        inp = meta.get("input_token_logprobs")
        if not isinstance(inp, list):
            return [0.0] * len(response_ids)

        all_lp = []
        for item in inp:
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                all_lp.append(float(item[0]) if item[0] is not None else 0.0)
            elif isinstance(item, dict) and "logprob" in item:
                val = item["logprob"]
                all_lp.append(float(val) if val is not None else 0.0)
            else:
                all_lp.append(0.0)
        if len(all_lp) > 1:
            all_lp = all_lp[1:]
        if len(all_lp) >= len(response_ids):
            return all_lp[-len(response_ids):]
        return [0.0] * (len(response_ids) - len(all_lp)) + all_lp

    async def _sample_reflections(
        self,
        *,
        turn_data: dict[str, Any],
        next_state: dict[str, Any],
    ) -> list[dict[str, Any]]:
        next_state_text = _flatten_message_content(next_state.get("content"))
        next_state_role = next_state.get("role", "user")
        reflect_messages = build_reflector_messages(
            original_messages=turn_data["messages"],
            failed_response_text=turn_data["response_text"],
            next_state_text=next_state_text,
            next_state_role=next_state_role,
        )
        outputs = await asyncio.gather(
            *[
                self._policy_chat_completion(
                    messages=reflect_messages,
                    tools=None,
                    temperature=self._hint_temperature,
                    max_tokens=self._hint_max_tokens,
                )
                for _ in range(self._hint_group_size)
            ]
        )
        reflections = []
        for output in outputs:
            hint = parse_hint_response(output["response_text"])
            if not hint:
                continue
            reflections.append(
                {
                    "hint": hint,
                    "reflect_messages": reflect_messages,
                    **output,
                }
            )
        return reflections

    async def _run_actor_with_hint(
        self,
        *,
        turn_data: dict[str, Any],
        hint: str,
    ) -> dict[str, Any]:
        actor_messages = append_hint_to_messages(turn_data["messages"], hint)
        output = await self._policy_chat_completion(
            messages=actor_messages,
            tools=turn_data.get("tools"),
            temperature=self._repair_temperature,
            max_tokens=self._repair_max_tokens,
        )
        output["actor_messages"] = actor_messages
        return output

    async def _validate_trial(
        self,
        *,
        session_id: str,
        turn_num: int,
        turn_data: dict[str, Any],
        next_state: dict[str, Any],
        reflection: dict[str, Any],
    ) -> dict[str, Any]:
        candidate = await self._run_actor_with_hint(turn_data=turn_data, hint=reflection["hint"])
        outcome = await self._env_adapter.validate_repair(
            session_id=session_id,
            turn_num=turn_num,
            turn_data=turn_data,
            next_state=next_state,
            hint=reflection["hint"],
            candidate=candidate,
        )
        return {
            "hint": reflection["hint"],
            "reflection": reflection,
            "candidate": candidate,
            "outcome": outcome,
        }

    def _build_hint_sample(
        self,
        *,
        reflection: dict[str, Any],
        reward_score: float,
        raw_binary_reward: float,
        sample_group_index: int,
    ) -> Sample:
        sample = Sample()
        sample.prompt = reflection["prompt_text"]
        sample.response = reflection["response_text"]
        sample.tokens = reflection["prompt_ids"] + reflection["response_ids"]
        sample.response_length = len(reflection["response_ids"])
        sample.loss_mask = [1] * len(reflection["response_ids"])
        sample.rollout_log_probs = reflection["response_logprobs"]
        sample.teacher_log_probs = torch.tensor(reflection["response_logprobs"], dtype=torch.float32)
        sample.status = Sample.Status.COMPLETED
        sample.index = next(self._index_counter)
        sample.group_index = sample_group_index
        sample.reward = {"score": float(reward_score)}
        sample.metadata = {
            "phase": "hint_grpo",
            "raw_reward": float(raw_binary_reward),
            "hint": reflection["hint"],
        }
        return sample

    async def _build_action_sample(
        self,
        *,
        turn_data: dict[str, Any],
        trial: dict[str, Any],
        sample_group_index: int,
    ) -> Sample | None:
        candidate = trial["candidate"]
        response_ids = candidate["response_ids"]
        if not response_ids:
            return None

        teacher_log_probs = await self._compute_policy_log_probs(
            candidate["prompt_text"],
            candidate["response_text"],
        )
        student_log_probs = await self._compute_policy_log_probs(
            turn_data["prompt_text"],
            candidate["response_text"],
        )

        sample = Sample()
        sample.prompt = turn_data["prompt_text"]
        sample.response = candidate["response_text"]
        sample.tokens = turn_data["prompt_ids"] + response_ids
        sample.response_length = len(response_ids)
        sample.loss_mask = [1] * len(response_ids)
        sample.rollout_log_probs = student_log_probs
        sample.teacher_log_probs = torch.tensor(teacher_log_probs, dtype=torch.float32)
        sample.status = Sample.Status.COMPLETED
        sample.index = next(self._index_counter)
        sample.group_index = sample_group_index
        sample.reward = {"score": 0.0}
        sample.metadata = {
            "phase": "action_opd",
            "hint": trial["hint"],
            "raw_reward": float(trial["outcome"].score),
        }
        return sample

    async def _self_opd_evaluate(
        self,
        session_id: str,
        turn_num: int,
        turn_data: dict[str, Any],
        next_state: dict[str, Any],
    ) -> dict[str, Any]:
        next_state_text = _flatten_message_content(next_state.get("content")) if next_state else ""
        next_state_role = next_state.get("role", "user") if next_state else "user"
        should_attempt = self._force_all_next_state or looks_like_failure_feedback(next_state_text, next_state_role)
        if not should_attempt:
            return {"hint_samples": [], "action_sample": None, "success_rate": None}

        reflections = await self._sample_reflections(turn_data=turn_data, next_state=next_state)
        if not reflections:
            logger.info("[Self-OPD] session=%s turn=%d produced no valid hints", session_id, turn_num)
            return {"hint_samples": [], "action_sample": None, "success_rate": 0.0}

        trials = await asyncio.gather(
            *[
                self._validate_trial(
                    session_id=session_id,
                    turn_num=turn_num,
                    turn_data=turn_data,
                    next_state=next_state,
                    reflection=reflection,
                )
                for reflection in reflections
            ]
        )

        rewards = [1.0 if trial["outcome"].success else 0.0 for trial in trials]
        success_rate = sum(rewards) / len(rewards) if rewards else 0.0
        normalized_rewards = normalize_binary_rewards(rewards)
        hint_group_index = next(self._sample_group_counter)

        hint_samples = []
        if normalized_rewards is not None:
            for trial, reward_score, raw_binary_reward in zip(trials, normalized_rewards, rewards, strict=False):
                hint_samples.append(
                    self._build_hint_sample(
                        reflection=trial["reflection"],
                        reward_score=reward_score,
                        raw_binary_reward=raw_binary_reward,
                        sample_group_index=hint_group_index,
                    )
                )
        else:
            logger.info(
                "[Self-OPD] session=%s turn=%d skipped hint-GRPO because reward variance is zero: %s",
                session_id,
                turn_num,
                rewards,
            )

        best_trial = select_best_successful_trial(trials)
        action_sample = None
        if best_trial is not None:
            action_sample = await self._build_action_sample(
                turn_data=turn_data,
                trial=best_trial,
                sample_group_index=next(self._sample_group_counter),
            )

        logger.info(
            "%s[Self-OPD] session=%s turn=%d hints=%d success_rate=%.2f hint_updates=%d action_update=%s%s",
            _CYAN,
            session_id,
            turn_num,
            len(trials),
            success_rate,
            len(hint_samples),
            action_sample is not None,
            _RESET,
        )
        return {
            "hint_samples": hint_samples,
            "action_sample": action_sample,
            "success_rate": success_rate,
        }

    def _fire_opd_task(self, session_id: str, turn_num: int, turn_data: dict[str, Any], next_state: dict[str, Any]):
        if not next_state:
            return
        task = asyncio.create_task(self._self_opd_evaluate(session_id, turn_num, turn_data, next_state))
        task.add_done_callback(self._task_done_cb)
        task.add_done_callback(lambda _t: self._maybe_submit_ready_samples(session_id))
        self._prm_tasks.setdefault(session_id, {})[turn_num] = task
        turn_data["has_next_state"] = True

    async def _enqueue_sample(self, sample: Sample):
        queue_group_id = next(self._output_group_counter)
        logger.info(
            "[Self-OPD] submitted %s sample session_group=%d index=%d reward=%.3f response_len=%d",
            sample.metadata.get("phase", "sample"),
            sample.group_index,
            sample.index,
            sample.reward.get("score", 0.0) if isinstance(sample.reward, dict) else 0.0,
            sample.response_length,
        )
        await asyncio.to_thread(self.output_queue.put, (queue_group_id, [sample]))

    def _maybe_submit_ready_samples(self, session_id: str, force_drop_without_next_state: bool = False):
        prm_tasks = self._prm_tasks.get(session_id, {})
        pending = self._pending_turn_data.get(session_id, {})
        for turn_num in sorted(list(pending.keys())):
            td = pending[turn_num]
            task = prm_tasks.get(turn_num)

            if task is None:
                if force_drop_without_next_state:
                    pending.pop(turn_num, None)
                    logger.info("[Self-OPD] dropped session=%s turn=%d (no next_state)", session_id, turn_num)
                continue
            if not task.done():
                continue

            pending.pop(turn_num, None)
            prm_tasks.pop(turn_num, None)
            try:
                result = task.result()
            except Exception as e:
                logger.warning("[Self-OPD] task failed session=%s turn=%d: %s", session_id, turn_num, e)
                continue

            success_rate = result.get("success_rate")
            if success_rate is not None:
                with self._eval_scores_lock:
                    self._eval_scores.append(success_rate)

            for sample in result.get("hint_samples") or []:
                self._safe_create_task(self._enqueue_sample(sample))

            action_sample = result.get("action_sample")
            if action_sample is not None:
                self._safe_create_task(self._enqueue_sample(action_sample))
