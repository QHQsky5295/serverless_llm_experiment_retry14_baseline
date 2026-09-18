# ---------------------------------------------------------------------------- #
#  serverlessllm                                                               #
#  copyright (c) serverlessllm team 2024                                       #
#                                                                              #
#  licensed under the apache license, version 2.0 (the "license");             #
#  you may not use this file except in compliance with the license.            #
#                                                                              #
#  you may obtain a copy of the license at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/license-2.0                  #
#                                                                              #
#  unless required by applicable law or agreed to in writing, software         #
#  distributed under the license is distributed on an "as is" basis,           #
#  without warranties or conditions of any kind, either express or implied.    #
#  see the license for the specific language governing permissions and         #
#  limitations under the license.                                              #
# ---------------------------------------------------------------------------- #
import asyncio
import gc
import inspect
import logging
import math
import os
import time
import uuid
from dataclasses import fields
from types import MethodType
from typing import Any, Dict, List, Optional, Sequence, Union, cast

import torch
from transformers import AutoTokenizer
from vllm import (
    AsyncEngineArgs,
    AsyncLLMEngine,
    EmbeddingRequestOutput,
    PoolingParams,
    PromptType,
    RequestOutput,
    SamplingParams,
)
from vllm.inputs import TokensPrompt
from vllm.utils import Counter

from relayserve.protocol_v4_3.canonical_trace_replay import (
    parse_d_profile_replay_xargs,
)
try:
    from vllm.lora.request import LoRARequest
except Exception:  # pragma: no cover - exercised in runtime env
    LoRARequest = None  # type: ignore[assignment]

from sllm.backends.backend_utils import (
    BackendStatus,
    SllmBackend,
)

logger = logging.getLogger("ray")


def _m3_boot_id() -> str:
    with open("/proc/sys/kernel/random/boot_id", "r", encoding="utf-8") as handle:
        return handle.read().strip()


_M3_BOOT_ID = _m3_boot_id()


def _m3_monotonic_seconds_to_ns(timestamp_s: Any) -> Optional[int]:
    """Convert a native EngineCore CLOCK_MONOTONIC event to integer ns."""
    try:
        timestamp = float(timestamp_s)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(timestamp) or timestamp < 0:
        return None
    return int(round(timestamp * 1_000_000_000))


def _as_env_bool(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return "1"
    if text in {"0", "false", "no", "n", "off"}:
        return "0"
    return "1" if bool(value) else "0"


def _apply_vllm_runtime_env(backend_config: Dict[str, Any]) -> Dict[str, str]:
    env_updates: Dict[str, str] = {}
    configured_env = backend_config.get("vllm_runtime_env")
    if isinstance(configured_env, dict):
        for key, value in configured_env.items():
            if key and value is not None:
                env_updates[str(key)] = str(value)

    if backend_config.get("vllm_use_v1") is not None:
        env_updates.setdefault(
            "VLLM_USE_V1",
            _as_env_bool(backend_config.get("vllm_use_v1")),
        )
    if backend_config.get("vllm_attention_backend") not in (None, ""):
        env_updates.setdefault(
            "VLLM_ATTENTION_BACKEND",
            str(backend_config.get("vllm_attention_backend")),
        )
    if backend_config.get("vllm_use_flashinfer_sampler") is not None:
        env_updates.setdefault(
            "VLLM_USE_FLASHINFER_SAMPLER",
            _as_env_bool(backend_config.get("vllm_use_flashinfer_sampler")),
        )
    env_updates.setdefault("VLLM_NO_USAGE_STATS", "1")

    for key, value in env_updates.items():
        os.environ[str(key)] = str(value)
    return env_updates


def _extract_vllm_metrics(output: RequestOutput) -> Optional[Dict[str, Any]]:
    metrics = getattr(output, "metrics", None)
    if metrics is None:
        return None

    arrival_time = getattr(metrics, "arrival_time", None)
    first_scheduled_time = getattr(metrics, "first_scheduled_time", None)
    first_token_time = getattr(metrics, "first_token_time", None)
    last_token_time = getattr(metrics, "last_token_time", None)
    finished_time = getattr(metrics, "finished_time", None)
    if last_token_time is None:
        last_token_time = finished_time

    completion_tokens = sum(len(result.token_ids) for result in output.outputs)
    prompt_tokens = len(output.prompt_token_ids or [])
    num_cached_tokens = getattr(output, "num_cached_tokens", None)
    if num_cached_tokens is not None:
        num_cached_tokens = max(0, int(num_cached_tokens))

    ttft_ms = None
    if (
        arrival_time is not None
        and first_token_time is not None
        and float(first_token_time) >= float(arrival_time)
    ):
        ttft_ms = (float(first_token_time) - float(arrival_time)) * 1000.0

    e2e_ms = None
    terminal_time = finished_time if finished_time is not None else last_token_time
    if (
        arrival_time is not None
        and terminal_time is not None
        and float(terminal_time) >= float(arrival_time)
    ):
        e2e_ms = (float(terminal_time) - float(arrival_time)) * 1000.0

    tpot_ms = None
    tpot_observed = False
    if (
        completion_tokens > 1
        and first_token_time is not None
        and last_token_time is not None
        and float(last_token_time) >= float(first_token_time)
    ):
        tpot_ms = (
            (float(last_token_time) - float(first_token_time))
            * 1000.0
            / max(completion_tokens - 1, 1)
        )
        tpot_observed = True

    return {
        "source": "serverlessllm_vllm_metrics",
        "ttft_ms": ttft_ms,
        "e2e_ms": e2e_ms,
        "tpot_ms": tpot_ms,
        "tpot_observed": tpot_observed,
        "completion_tokens_observed": completion_tokens,
        "prompt_tokens_observed": prompt_tokens,
        "num_cached_tokens": num_cached_tokens,
        "prefix_cache_hit": (
            bool(num_cached_tokens > 0)
            if num_cached_tokens is not None
            else None
        ),
        "cached_token_fraction": (
            float(num_cached_tokens) / float(prompt_tokens)
            if num_cached_tokens is not None and prompt_tokens > 0
            else None
        ),
        "stream_event_count": completion_tokens,
        "arrival_time": arrival_time,
        "first_scheduled_time": first_scheduled_time,
        "first_token_time": first_token_time,
        "last_token_time": last_token_time,
        "finished_time": finished_time,
    }


def process_output(output: RequestOutput, model_name: str) -> Dict[str, Any]:
    choices: List[Dict[str, Any]] = [
        {
            "index": idx,
            "message": {
                "role": "assistant",
                "content": result.text,
            },
            "logprobs": None,
            "finish_reason": result.finish_reason,
        }
        for idx, result in enumerate(output.outputs)
    ]

    api_response = {
        "id": output.request_id,
        "object": "chat.completion",
        "created": (
            int(time.time())
            if output.metrics is None
            else output.metrics.arrival_time
        ),
        "model": model_name,
        "choices": choices,
        "usage": {
            "prompt_tokens": len(output.prompt_token_ids),
            "completion_tokens": sum(
                len(result.token_ids) for result in output.outputs
            ),
            "total_tokens": len(output.prompt_token_ids)
            + sum(len(result.token_ids) for result in output.outputs),
        },
    }
    metrics = dict(_extract_vllm_metrics(output) or {})
    num_cached_tokens = getattr(output, "num_cached_tokens", None)
    if num_cached_tokens is not None:
        num_cached_tokens = max(0, int(num_cached_tokens))
    prompt_tokens = len(output.prompt_token_ids or [])
    metrics.update(
        {
            "prompt_tokens_observed": prompt_tokens,
            "num_cached_tokens": num_cached_tokens,
            "prefix_cache_hit": (
                bool(num_cached_tokens > 0)
                if num_cached_tokens is not None
                else None
            ),
            "cached_token_fraction": (
                float(num_cached_tokens) / float(prompt_tokens)
                if num_cached_tokens is not None and prompt_tokens > 0
                else None
            ),
        }
    )
    api_response["metrics"] = metrics
    return api_response


def process_embedding_output(
    outputs: List[EmbeddingRequestOutput], model_name: str
) -> Dict[str, Any]:
    valid_outputs = [output for output in outputs if output is not None]
    query_tokens = sum(len(output.prompt_token_ids) for output in valid_outputs)
    api_response = {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "index": i,
                "embedding": output.outputs.embedding,
            }
            for i, output in enumerate(outputs)
        ],
        "model": model_name,
        "usage": {
            "query_tokens": query_tokens,
            "total_tokens": query_tokens,
        },
    }
    return api_response


class LLMEngineStatusDict:
    def __init__(self):
        self.status_dict: Dict[str, Union[RequestOutput, str]] = {}
        self.lock = asyncio.Lock()

    async def update_status(
        self, request_id: str, request_output: Union[RequestOutput, str]
    ):
        async with self.lock:
            self.status_dict[request_id] = request_output

    async def delete_request(self, request_id: str):
        async with self.lock:
            del self.status_dict[request_id]

    async def return_all_results(self) -> List[Union[RequestOutput, str]]:
        async with self.lock:
            return list(self.status_dict.values())

    async def return_all_request_ids(self) -> List[str]:
        async with self.lock:
            return list(self.status_dict.keys())

    async def request_count(self) -> int:
        async with self.lock:
            return len(self.status_dict)


# Note the GPU resource will be decided when the backend is created
class VllmBackend(SllmBackend):
    # This class implements every method in vllm.entrypoints.openai.api_server
    # https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/api_server.py
    # except that we use ray.remote instead of @app and we also add a few new methods:
    # - stop: stops every ongoing request and then stops the backend
    # - get_current_tokens: returns a list of all ongoing request tokens
    # - resume_kv_cache: resumes the key-value cache for the given requests
    def __init__(
        self, model: str, backend_config: Optional[Dict[str, Any]] = None
    ) -> None:
        if backend_config is None:
            raise ValueError("Backend config is missing")

        self.status: BackendStatus = BackendStatus.UNINITIALIZED
        self.status_lock = asyncio.Lock()
        self.backend_config = backend_config
        self.request_trace = LLMEngineStatusDict()
        # if trace_debug is True, request trace will not be deleted after completion
        self.trace_debug = backend_config.get("trace_debug", False)
        self.m3_capture_vllm_engine_events = bool(
            backend_config.get("m3_capture_vllm_engine_events", False)
        )
        self._m3_v1_engine_events: Dict[str, Dict[str, Any]] = {}
        self._m3_event_capture_installed = False
        self.vllm_runtime_env = _apply_vllm_runtime_env(backend_config)
        self.enforce_eager = backend_config.get("enforce_eager", False)
        self.enable_lora = bool(backend_config.get("enable_lora", False))
        self.disable_lora_embeddings = bool(
            backend_config.get("disable_lora_embeddings", False)
        )
        self.pretrained_model_name_or_path = backend_config.get(
            "pretrained_model_name_or_path"
        )
        self.tokenizer = None
        self.enable_prefix_caching = backend_config.get(
            "enable_prefix_caching", True
        )
        self.task = backend_config.get("task", "auto")

        async_engine_fields = {f.name for f in fields(AsyncEngineArgs)}
        filtered_engine_config = {
            k: v for k, v in backend_config.items() if k in async_engine_fields
        }

        load_format = backend_config.get("load_format")
        torch_dtype = backend_config.get("torch_dtype")
        if torch_dtype is not None:
            filtered_engine_config["dtype"] = torch_dtype
        if self.enable_lora:
            filtered_engine_config["enable_lora"] = True
            if self.disable_lora_embeddings:
                os.environ["VLLM_DISABLE_LORA_EMBEDDINGS"] = "1"
            else:
                os.environ.pop("VLLM_DISABLE_LORA_EMBEDDINGS", None)

        if load_format is not None:
            filtered_engine_config["load_format"] = load_format
            filtered_engine_config["model"] = backend_config.get(
                "pretrained_model_name_or_path"
            )
        else:
            storage_path = os.getenv(
                "STORAGE_PATH", os.path.expanduser("~/models")
            )
            model_path = os.path.join(storage_path, "vllm", model)
            filtered_engine_config["model"] = model_path
            filtered_engine_config["load_format"] = "serverless_llm"

        # NOTE: Automatic enable prefix cachinging
        filtered_engine_config["enforce_eager"] = self.enforce_eager
        filtered_engine_config["enable_prefix_caching"] = (
            self.enable_prefix_caching
        )
        filtered_engine_config["task"] = self.task

        logger.info(
            f"Creating new VLLM engine with config: {filtered_engine_config}"
        )
        logger.info(f"Applied VLLM runtime env: {self.vllm_runtime_env}")

        self.engine_args = AsyncEngineArgs(**filtered_engine_config)

        self.engine = None

    async def init_backend(self) -> None:
        async with self.status_lock:
            if self.status != BackendStatus.UNINITIALIZED:
                return
            self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)
            self._install_m3_v1_event_capture()
            tokenizer_source = (
                self.pretrained_model_name_or_path
                or getattr(self.engine_args, "model", None)
            )
            if tokenizer_source:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_source,
                    trust_remote_code=True,
                )
            self.status = BackendStatus.RUNNING

    def _install_m3_v1_event_capture(self) -> None:
        """Passively retain native V1 EngineCore request events.

        The V1 scheduler already records QUEUED/SCHEDULED with its own
        CLOCK_MONOTONIC clock.  RequestOutput does not expose those events, so
        M3 captures them at the existing OutputProcessor boundary before the
        RequestState is released.  The original processor is called exactly
        once with unchanged arguments and unchanged outputs.
        """
        if not self.m3_capture_vllm_engine_events:
            return
        if self.engine is None:
            raise RuntimeError("M3 event capture requires an initialized engine")
        output_processor = getattr(self.engine, "output_processor", None)
        if output_processor is None:
            raise RuntimeError(
                "M3 event capture requires the vLLM V1 OutputProcessor"
            )
        if not bool(getattr(output_processor, "log_stats", False)):
            raise RuntimeError(
                "M3 event capture requires native vLLM EngineCore events"
            )
        original_process_outputs = output_processor.process_outputs
        backend = self

        def capture_process_outputs(
            processor,
            engine_core_outputs,
            engine_core_timestamp=None,
            iteration_stats=None,
        ):
            for engine_output in engine_core_outputs:
                request_id = str(engine_output.request_id)
                request_events = backend._m3_v1_engine_events.setdefault(
                    request_id, {}
                )
                for event in engine_output.events or ():
                    event_type = getattr(event, "type", None)
                    event_name = getattr(event_type, "name", str(event_type))
                    if event_name not in {"QUEUED", "SCHEDULED", "PREEMPTED"}:
                        continue
                    event_key = event_name.lower()
                    if event_key in request_events:
                        continue
                    monotonic_ns = _m3_monotonic_seconds_to_ns(
                        getattr(event, "timestamp", None)
                    )
                    if monotonic_ns is None:
                        continue
                    request_events[event_key] = {
                        "event_type": event_name,
                        "monotonic_seconds": float(event.timestamp),
                        "monotonic_ns": monotonic_ns,
                    }
            return original_process_outputs(
                engine_core_outputs,
                engine_core_timestamp,
                iteration_stats,
            )

        output_processor.process_outputs = MethodType(
            capture_process_outputs, output_processor
        )
        self._m3_event_capture_installed = True

    def _resolve_request_output_limit(self, requested_output_tokens: int) -> int:
        desired_tokens = max(1, int(requested_output_tokens or 0))
        cap = int(self.backend_config.get("max_output_tokens_cap", 0) or 0)
        if cap > 0:
            desired_tokens = min(desired_tokens, cap)
        return max(1, desired_tokens)

    def _render_messages_prompt(self, messages: List[Dict[str, Any]]) -> str:
        if self.tokenizer is not None:
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                if isinstance(prompt, str) and prompt.strip():
                    return prompt
            except Exception:
                pass
        return "\n".join(
            f"{str(message.get('role') or 'user').capitalize()}: "
            f"{'' if message.get('content') is None else str(message.get('content'))}"
            for message in messages
        )

    def _prepare_prompt(self, prompt: str, max_tokens: int) -> tuple[str, int, int]:
        max_tokens = self._resolve_request_output_limit(max_tokens)
        max_len = max(
            32, int(self.backend_config.get("max_model_len", 2048) or 2048)
        )
        max_input_len = max(
            0, int(self.backend_config.get("max_input_len", 0) or 0)
        )
        prompt_budget = max(8, max_len - int(max_tokens) - 8)
        if max_input_len > 0:
            prompt_budget = min(prompt_budget, max_input_len)
        if self.tokenizer is not None:
            try:
                token_ids = self.tokenizer.encode(
                    prompt, add_special_tokens=False
                )
                if len(token_ids) > prompt_budget:
                    logger.warning(
                        "Prompt exceeded budget (%s > %s); truncating to align with fair-run prompt guard",
                        len(token_ids),
                        prompt_budget,
                    )
                    token_ids = token_ids[-prompt_budget:]
                    prompt = self.tokenizer.decode(
                        token_ids, skip_special_tokens=False
                    )
                actual_input_tokens = max(1, len(token_ids))
            except Exception:
                token_ids = None
        else:
            token_ids = None
        if token_ids is None:
            max_chars = min(max_len * 4, 8192)
            if len(prompt) > max_chars:
                prompt = prompt[-max_chars:]
            actual_input_tokens = min(max(1, len(prompt)), prompt_budget)
        max_tokens = min(max_tokens, max(1, max_len - actual_input_tokens - 8))
        return prompt, actual_input_tokens, max(1, max_tokens)

    def _build_lora_request(
        self,
        lora_adapter_name: Optional[str],
        lora_adapter_id: Optional[int],
        lora_adapter_path: Optional[str],
    ):
        if lora_adapter_name is None:
            return None
        if not self.enable_lora:
            return {"error": "LoRA requested but vLLM backend was not initialized with enable_lora=true"}
        if LoRARequest is None:
            return {
                "error": "vLLM LoRA support is unavailable in the active ServerlessLLM environment"
            }
        if not lora_adapter_path:
            return {
                "error": f"LoRA adapter path missing for adapter {lora_adapter_name}"
            }
        resolved_lora_path = str(lora_adapter_path)
        if not os.path.isabs(resolved_lora_path):
            storage_path = os.getenv(
                "STORAGE_PATH", os.path.expanduser("~/models")
            )
            resolved_lora_path = os.path.join(storage_path, resolved_lora_path)
        if not os.path.exists(resolved_lora_path):
            return {
                "error": f"LoRA adapter path not found: {resolved_lora_path}"
            }
        return LoRARequest(
            lora_name=str(lora_adapter_name),
            lora_int_id=int(lora_adapter_id or 0) or 1,
            lora_path=resolved_lora_path,
            base_model_name=str(self.pretrained_model_name_or_path or ""),
        )

    async def generate(self, request_data: Dict[str, Any]):
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return {"error": "Engine is not running"}

        assert self.engine is not None

        if request_data is None:
            return {"error": "Request data is missing"}

        backend_arrival_monotonic_ns = time.monotonic_ns()
        internal_metrics = dict(
            request_data.pop("_sllm_internal_metrics", {}) or {}
        )
        internal_metrics["backend_started_at"] = time.time()

        model_name: str = request_data.pop("model", "vllm-model")
        messages: List[Dict[str, Any]] = request_data.pop("messages", [])
        requested_max_tokens = int(request_data.get("max_tokens", 256) or 256)
        request_data.pop("stream", None)
        request_data.pop("stream_options", None)
        prompt = request_data.pop("prompt", None)
        input_token_ids = request_data.pop("input_tokens", None)
        relayserve_frozen_eos_token_ids = request_data.pop(
            "relayserve_frozen_eos_token_ids", None
        )
        if input_token_ids is None:
            if prompt is None:
                prompt = self._render_messages_prompt(messages)
            if not prompt:
                return {"error": "Missing prompt in request data"}
            prompt, actual_input_tokens, max_tokens = self._prepare_prompt(
                str(prompt), requested_max_tokens
            )
            request_data["max_tokens"] = max_tokens
            inputs: Union[str, TokensPrompt] = prompt
        else:
            actual_input_tokens = len(input_token_ids)
            inputs = TokensPrompt(prompt_token_ids=input_token_ids)

        request_id: str = request_data.pop(
            "request_id", f"chatcmpl-{uuid.uuid4()}"
        )
        if self.m3_capture_vllm_engine_events:
            self._m3_v1_engine_events.pop(request_id, None)
        lora_adapter_name = request_data.pop("lora_adapter_name", None)
        lora_adapter_id = request_data.pop("lora_adapter_id", None)
        lora_adapter_path = request_data.pop("lora_adapter_path", None)
        lora_request = self._build_lora_request(
            lora_adapter_name,
            lora_adapter_id,
            lora_adapter_path,
        )
        if isinstance(lora_request, dict) and lora_request.get("error"):
            return lora_request

        try:
            sampling_params = SamplingParams(**request_data)
            relayserve_replay_target = parse_d_profile_replay_xargs(
                sampling_params.extra_args,
                max_tokens=int(sampling_params.max_tokens or 0),
                stop_token_ids=tuple(
                    relayserve_frozen_eos_token_ids or ()
                ),
            )
            if (
                relayserve_replay_target is not None
                and relayserve_frozen_eos_token_ids is None
            ):
                raise ValueError(
                    "canonical replay requires frozen EOS token IDs"
                )
        except Exception as e:
            return {"error": f"Invalid sampling parameters: {e}"}

        results_generator = self.engine.generate(
            inputs,
            sampling_params,
            request_id,
            lora_request=lora_request,
        )

        # TODO stream results

        # Non-stream case
        final_output = None
        completion_tokens_observed = 0
        first_token_at = None
        first_token_monotonic_ns = None
        last_token_at = None
        token_timestamps = []
        relayserve_token_id_events = []
        async for response_output in results_generator:
            final_output = response_output
            current_completion_tokens = sum(
                len(result.token_ids) for result in response_output.outputs
            )
            if relayserve_replay_target is not None:
                if len(response_output.outputs) != 1:
                    raise RuntimeError(
                        "canonical replay requires exactly one output"
                    )
                relayserve_token_id_events.append({
                    "monotonic_ns": time.monotonic_ns(),
                    "cumulative_output_token_ids": list(
                        response_output.outputs[0].token_ids
                    ),
                })
            if current_completion_tokens > completion_tokens_observed:
                delta = current_completion_tokens - completion_tokens_observed
                now = time.time()
                if first_token_at is None:
                    first_token_at = now
                    first_token_monotonic_ns = time.monotonic_ns()
                last_token_at = now
                if delta == 1:
                    token_timestamps.append(now)
                completion_tokens_observed = current_completion_tokens
            await self.request_trace.update_status(request_id, response_output)

        assert final_output is not None
        finished_at = time.time()
        m3_engine_events = self._m3_v1_engine_events.pop(request_id, {})

        if not self.trace_debug:
            await self.request_trace.delete_request(request_id)
        response = process_output(final_output, model_name)
        if relayserve_replay_target is not None:
            if len(final_output.outputs) != 1:
                raise RuntimeError(
                    "canonical replay final output cardinality drift"
                )
            response["relayserve_native_observability"] = {
                "request_id": final_output.request_id,
                "prompt_token_ids": list(final_output.prompt_token_ids),
                "output_token_ids": list(
                    final_output.outputs[0].token_ids
                ),
                "token_id_events": relayserve_token_id_events,
                "finish_reason": final_output.outputs[0].finish_reason,
                "canonical_replay_applied": True,
                "canonical_target_digest": relayserve_replay_target.digest,
            }
        if actual_input_tokens > 0:
            response.setdefault("usage", {})["prompt_tokens"] = actual_input_tokens
            response["usage"]["total_tokens"] = actual_input_tokens + int(
                response["usage"].get("completion_tokens", 0)
            )
        metrics = dict(response.get("metrics", {}) or {})
        request_received_at = internal_metrics.get("request_received_at")
        backend_started_at = internal_metrics.get("backend_started_at")
        runtime_ttft_ms = None
        if first_token_at is not None and backend_started_at is not None:
            runtime_ttft_ms = max(
                0.0, (float(first_token_at) - float(backend_started_at)) * 1000.0
            )
        if (
            metrics.get("first_token_time") is not None
            and request_received_at is not None
        ):
            metrics["ttft_ms"] = max(
                0.0,
                (float(metrics["first_token_time"]) - float(request_received_at))
                * 1000.0,
            )
        if request_received_at is not None:
            terminal_time = metrics.get("finished_time") or metrics.get(
                "last_token_time"
            )
            if terminal_time is not None:
                metrics["e2e_ms"] = max(
                    0.0,
                    (float(terminal_time) - float(request_received_at)) * 1000.0,
                )
        metrics["runtime_ttft_ms"] = runtime_ttft_ms
        metrics["serverless_overhead_ms"] = (
            max(0.0, float(metrics["ttft_ms"]) - float(runtime_ttft_ms))
            if metrics.get("ttft_ms") is not None and runtime_ttft_ms is not None
            else None
        )
        if not metrics.get("tpot_observed") and (
            completion_tokens_observed > 1
            and len(token_timestamps) == completion_tokens_observed
        ):
            metrics["tpot_ms"] = (
                (float(token_timestamps[-1]) - float(token_timestamps[0])) * 1000.0
                / max(completion_tokens_observed - 1, 1)
            )
            metrics["tpot_observed"] = True
        metrics["request_received_at"] = request_received_at
        metrics["backend_started_at"] = backend_started_at
        metrics["first_token_at"] = first_token_at
        metrics["last_token_at"] = last_token_at
        metrics["finished_at"] = finished_at
        scheduled_event = m3_engine_events.get("scheduled")
        scheduled_monotonic_ns = (
            int(scheduled_event["monotonic_ns"])
            if scheduled_event is not None
            else None
        )
        metrics["m3_ingress_monotonic_ns"] = internal_metrics.get(
            "m3_ingress_monotonic_ns"
        )
        metrics["m3_enqueue_monotonic_ns"] = internal_metrics.get(
            "m3_enqueue_monotonic_ns"
        )
        metrics["m3_runtime_ready_monotonic_ns"] = internal_metrics.get(
            "m3_runtime_ready_monotonic_ns"
        )
        metrics["m3_backend_arrival_monotonic_ns"] = (
            backend_arrival_monotonic_ns
        )
        metrics["m3_scheduled_monotonic_ns"] = scheduled_monotonic_ns
        metrics["m3_first_token_monotonic_ns"] = first_token_monotonic_ns
        metrics["m3_scheduled_event_error_bound_ns"] = (
            0 if scheduled_monotonic_ns is not None else None
        )
        metrics["m3_scheduled_event_source"] = (
            "vllm_v1_engine_core_scheduled_event"
            if scheduled_monotonic_ns is not None
            else None
        )
        metrics["m3_scheduled_event_monotonic_seconds"] = (
            scheduled_event.get("monotonic_seconds")
            if scheduled_event is not None
            else None
        )
        metrics["m3_vllm_engine_events"] = m3_engine_events
        metrics["m3_event_capture_installed"] = (
            self._m3_event_capture_installed
        )
        metrics["m3_ingress_boot_id"] = internal_metrics.get(
            "m3_ingress_boot_id"
        )
        metrics["m3_backend_boot_id"] = _M3_BOOT_ID
        metrics["m3_scheduled_boot_id"] = (
            _M3_BOOT_ID if scheduled_monotonic_ns is not None else None
        )
        metrics["m3_monotonic_resolution_ns"] = internal_metrics.get(
            "m3_monotonic_resolution_ns"
        )
        metrics["queue_wait_ms"] = internal_metrics.get("queue_wait_ms")
        metrics["lora_load_ms"] = internal_metrics.get("lora_load_ms")
        metrics["cache_hit"] = internal_metrics.get("lora_cache_hit")
        metrics["gpu_ready_request"] = internal_metrics.get("lora_cache_hit")
        metrics["scaleup_affected"] = internal_metrics.get("scaleup_affected")
        metrics["scaleup_first_service"] = internal_metrics.get("scaleup_first_service")
        metrics["cold_start_latency_ms"] = internal_metrics.get("cold_start_latency_ms")
        metrics["instance_id"] = internal_metrics.get("instance_id")
        metrics["instance_created_at"] = internal_metrics.get("instance_created_at")
        metrics["instance_ready_at"] = internal_metrics.get("instance_ready_at")
        cached_tokens = metrics.get("num_cached_tokens")
        metrics["cached_token_fraction"] = (
            float(cached_tokens) / float(actual_input_tokens)
            if cached_tokens is not None and actual_input_tokens > 0
            else None
        )
        response["metrics"] = metrics
        return response

    async def load_lora_adapter(self, lora_name: str, lora_path: str):
        return {"cache_hit": None, "load_ms": None}

    async def shutdown(self):
        """Abort all requests and shutdown the backend."""
        async with self.status_lock:
            if self.status == BackendStatus.DELETING:
                return
            self.status = BackendStatus.DELETING

        # Abort all requests
        requests = await self.request_trace.return_all_request_ids()
        tasks = [self.engine.abort(request_id) for request_id in requests]
        await asyncio.gather(*tasks)
        if hasattr(self, "engine"):
            del self.engine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    async def stop(self) -> None:
        """Wait for all requests to finish and shutdown the backend."""
        async with self.status_lock:
            if self.status.value >= BackendStatus.STOPPING.value:
                return
            self.status = BackendStatus.STOPPING
        while await self.request_trace.request_count() > 0:
            logger.info("Waiting for all requests to finish")
            await asyncio.sleep(1)
        logger.info("All requests finished. Shutting down the backend.")
        await self.shutdown()

    async def get_current_tokens(self) -> List[List[int]]:
        """Return a list of all ongoing request tokens."""
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return []
        results = await self.request_trace.return_all_results()
        ongoing_results: List[RequestOutput] = [
            result for result in results if isinstance(result, RequestOutput)
        ]
        tokens: List[List[int]] = [
            result.prompt_token_ids + result.outputs[0].token_ids
            for result in ongoing_results
        ]
        return tokens

    async def resume_kv_cache(self, request_datas: List[List[int]]) -> None:
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return
        constructed_inputs = [
            {
                "input_tokens": request_data,
                "max_tokens": 1,
            }
            for request_data in request_datas
        ]
        tasks = [self.generate(inputs) for inputs in constructed_inputs]
        await asyncio.gather(*tasks)

    async def encode(self, request_data: Dict[str, Any]):
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return {"error": "Engine is not running"}

        assert self.engine is not None

        if not request_data:
            return {"error": "Request data is missing"}

        request_counter: Counter = Counter()
        pooling_params: PoolingParams = PoolingParams()
        model_name = request_data.get("model", "vllm-model")
        query = request_data.get("input", [])

        if not query:
            return {"error": "No inputs provided"}

        inputs = cast(Union[PromptType, Sequence[PromptType]], query)

        async def process_input(input_data) -> List[EmbeddingRequestOutput]:
            request_id = str(next(request_counter))
            res = self.engine.encode(input_data, pooling_params, request_id)
            return [result async for result in res]

        raw_outputs = await asyncio.gather(
            *[process_input(input_data) for input_data in inputs],
            return_exceptions=True,
        )

        valid_outputs = []
        for output in raw_outputs:
            if isinstance(output, Exception):
                logger.error(f"Error encountered: {output}")
            else:
                valid_outputs.extend(output)

        if not valid_outputs:
            return {"error": "All inputs failed"}

        return process_embedding_output(valid_outputs, model_name)
