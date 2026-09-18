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
import json
import os
import threading
import time
import uuid
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional

import peft
import torch
import torch.nn.functional as F
import transformers
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.generation.streamers import BaseStreamer

from sllm.backends.backend_utils import BackendStatus, SllmBackend
from sllm.logger import init_logger
from sllm_store.transformers import load_lora, load_model, save_lora

logger = init_logger(__name__)


class DeletingException(Exception):
    pass


class InferenceStatus(BaseStreamer):
    def __init__(
        self,
        status_getter: Callable[[], BackendStatus],
        prompt_tokens: int = 0,
        request_id: Optional[str] = None,
        internal_metrics: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self._status_getter = status_getter
        self.prompt_tokens = max(int(prompt_tokens), 0)
        self.request_id = request_id or "unknown"
        self.internal_metrics = dict(internal_metrics or {})
        self.intermediate = []
        self.started_at = float(
            self.internal_metrics.get("backend_started_at", time.time())
        )
        self.finished_at: Optional[float] = None
        self.first_token_at: Optional[float] = None
        self.last_token_at: Optional[float] = None
        self.generated_tokens_observed = 0
        self.stream_event_count = 0
        self.multi_token_event_detected = False
        self.event_sizes: List[int] = []
        self.token_timestamps: List[float] = []
        self._saw_prompt_snapshot: List[bool] = []
        self.first_token_exact = False

    def _normalize_value(self, value):
        data = value.tolist() if hasattr(value, "tolist") else value
        if isinstance(data, tuple):
            data = list(data)
        if not isinstance(data, list):
            return [[data]]
        if not data:
            return []
        if isinstance(data[0], list):
            return [list(seq) for seq in data]
        return [list(data)]

    def _is_deleting(self) -> bool:
        return self._status_getter() == BackendStatus.DELETING

    def _record_new_tokens(self, new_tokens: int, now: float):
        if new_tokens <= 0:
            return
        if self.first_token_at is None:
            self.first_token_at = now
            self.first_token_exact = int(new_tokens) == 1
        self.last_token_at = now
        self.generated_tokens_observed += int(new_tokens)
        self.stream_event_count += 1
        self.event_sizes.append(int(new_tokens))
        if int(new_tokens) == 1:
            self.token_timestamps.append(now)
        else:
            self.multi_token_event_detected = True

    def put(self, value):
        raw_value = value
        value = self._normalize_value(value)
        if not value:
            return
        now = time.time()
        if not self.intermediate or len(self.intermediate) != len(value):
            self.intermediate = [[] for _ in value]
            self._saw_prompt_snapshot = [False for _ in value]

        for i, chunk in enumerate(value):
            chunk = list(chunk)
            current = self.intermediate[i]
            saw_prompt_snapshot = self._saw_prompt_snapshot[i]
            is_prompt_snapshot = (
                not saw_prompt_snapshot
                and self.prompt_tokens > 0
                and len(chunk) == self.prompt_tokens
            )
            is_initial_sequence_with_generation = (
                not saw_prompt_snapshot
                and self.prompt_tokens > 0
                and len(chunk) > self.prompt_tokens
            )
            is_sequence_snapshot = (
                saw_prompt_snapshot
                and len(chunk) > len(current)
                and current == chunk[: len(current)]
            )
            if is_prompt_snapshot:
                logger.debug(
                    "InferenceStatus[%s|%s] prompt snapshot prompt_tokens=%s raw_type=%s raw_shape=%s normalized_len=%s",
                    self.request_id,
                    id(self),
                    self.prompt_tokens,
                    type(raw_value).__name__,
                    getattr(raw_value, "shape", None),
                    len(chunk),
                )
                self.intermediate[i] = list(chunk)
                self._saw_prompt_snapshot[i] = True
            elif is_initial_sequence_with_generation:
                generated_total = max(0, len(chunk) - self.prompt_tokens)
                logger.debug(
                    "InferenceStatus[%s|%s] prompt+generation snapshot prompt_tokens=%s generated_total=%s raw_type=%s raw_shape=%s normalized_len=%s",
                    self.request_id,
                    id(self),
                    self.prompt_tokens,
                    generated_total,
                    type(raw_value).__name__,
                    getattr(raw_value, "shape", None),
                    len(chunk),
                )
                self.intermediate[i] = list(chunk)
                self._saw_prompt_snapshot[i] = True
                self._record_new_tokens(generated_total, now)
            elif is_sequence_snapshot:
                prev_generated = (
                    max(0, len(current) - self.prompt_tokens) if current else 0
                )
                self.intermediate[i] = list(chunk)
                generated_total = max(0, len(self.intermediate[i]) - self.prompt_tokens)
                self._record_new_tokens(generated_total - prev_generated, now)
            else:
                current.extend(chunk)
                self._record_new_tokens(len(chunk), now)

        logger.debug(
            "InferenceStatus[%s|%s] events=%s observed=%s first=%s last=%s output=%s",
            self.request_id,
            id(self),
            self.stream_event_count,
            self.generated_tokens_observed,
            self.first_token_at,
            self.last_token_at,
            self.intermediate,
        )
        if self._is_deleting():
            raise DeletingException("Backend is deleting")

    def end(self):
        logger.error("Inference completed")
        self.finished_at = time.time()

    def get(self):
        return deepcopy(self.intermediate)

    def delete(self):
        logger.info("Deleting intermediate output")
        self.intermediate = []
        self.finished_at = time.time()

    def build_metrics(self, completion_tokens: int) -> Dict[str, Any]:
        completion_tokens = max(int(completion_tokens or 0), 0)
        finished_at = self.finished_at if self.finished_at is not None else time.time()
        request_received_at = self.internal_metrics.get("request_received_at")
        ttft_ms = None
        if self.first_token_at is not None:
            if request_received_at is not None:
                ttft_ms = (self.first_token_at - float(request_received_at)) * 1000.0
            else:
                ttft_ms = (self.first_token_at - self.started_at) * 1000.0
        if request_received_at is not None:
            e2e_ms = max(0.0, (finished_at - float(request_received_at)) * 1000.0)
        else:
            e2e_ms = max(0.0, (finished_at - self.started_at) * 1000.0)
        runtime_ttft_ms = None
        if self.first_token_at is not None:
            runtime_ttft_ms = max(0.0, (self.first_token_at - self.started_at) * 1000.0)
        tpot_ms = None
        tpot_observed = False
        if (
            completion_tokens > 1
            and self.first_token_at is not None
            and self.last_token_at is not None
            and self.first_token_exact
        ):
            tpot_ms = (
                (self.last_token_at - self.first_token_at) * 1000.0
                / max(completion_tokens - 1, 1)
            )
            tpot_observed = True
        logger.debug(
            "InferenceStatus[%s|%s] build_metrics completion=%s observed=%s events=%s ttft_ms=%s tpot_ms=%s",
            self.request_id,
            id(self),
            completion_tokens,
            self.generated_tokens_observed,
            self.stream_event_count,
            ttft_ms,
            tpot_ms,
        )
        return {
            "source": "serverlessllm_transformers_streamer",
            "ttft_ms": ttft_ms,
            "e2e_ms": e2e_ms,
            "runtime_ttft_ms": runtime_ttft_ms,
            "serverless_overhead_ms": (
                max(0.0, float(ttft_ms) - float(runtime_ttft_ms))
                if ttft_ms is not None and runtime_ttft_ms is not None
                else None
            ),
            "tpot_ms": tpot_ms,
            "tpot_observed": tpot_observed,
            "completion_tokens_observed": self.generated_tokens_observed,
            "stream_event_count": self.stream_event_count,
            "multi_token_event_detected": self.multi_token_event_detected,
            "event_sizes": list(self.event_sizes),
            "request_received_at": request_received_at,
            "backend_started_at": self.started_at,
            "first_token_at": self.first_token_at,
            "last_token_at": self.last_token_at,
            "finished_at": finished_at,
            "queue_wait_ms": self.internal_metrics.get("queue_wait_ms"),
            "lora_load_ms": self.internal_metrics.get("lora_load_ms"),
            "lora_cache_hit": self.internal_metrics.get("lora_cache_hit"),
            "cache_hit": self.internal_metrics.get("lora_cache_hit"),
            "gpu_ready_request": self.internal_metrics.get("lora_cache_hit"),
            "scaleup_affected": self.internal_metrics.get("scaleup_affected"),
            "scaleup_first_service": self.internal_metrics.get("scaleup_first_service"),
            "cold_start_latency_ms": self.internal_metrics.get("cold_start_latency_ms"),
            "instance_id": self.internal_metrics.get("instance_id"),
            "instance_created_at": self.internal_metrics.get("instance_created_at"),
            "instance_ready_at": self.internal_metrics.get("instance_ready_at"),
        }


class TransformersBackend(SllmBackend):
    def __init__(
        self, model_name: str, backend_config: Optional[Dict[str, Any]] = None
    ) -> None:
        self.backend_config = backend_config
        logger.info(
            f"Initializing TransformersBackend for {model_name} with config: {backend_config}"
        )
        self.model_name = model_name
        self.pretrained_model_name_or_path = backend_config.get(
            "pretrained_model_name_or_path"
        )
        self.status: BackendStatus = BackendStatus.UNINITIALIZED
        self.inf_status = InferenceStatus(lambda: self.status)
        self.status_lock = threading.Lock()
        self.generate_lock = threading.Lock()
        self.active_streamers: Dict[str, InferenceStatus] = {}
        self.active_streamers_lock = threading.Lock()
        self.model = None
        self.tokenizer = None
        self.past_key_values = None

    def _build_chat_completion_response(
        self,
        *,
        request_id: str,
        model_name: str,
        output_text: str,
        prompt_tokens: int,
        total_tokens: int,
        max_tokens: int,
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        completion_tokens = total_tokens - prompt_tokens
        finish_reason = "stop" if completion_tokens < max_tokens else "length"
        response = {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": output_text,
                    },
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            "metrics": metrics,
        }
        return response

    def convert_str_to_json(self, json_str):
        try:
            # Parse the JSON string and return the corresponding Python object
            json_obj = json.loads(json_str)
            return json_obj
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON string: {e}")
            return None

    def init_backend(self) -> None:
        with self.status_lock:
            if self.status != BackendStatus.UNINITIALIZED:
                return
            device_map = self.backend_config.get("device_map", "auto")
            torch_dtype = self.backend_config.get("torch_dtype", torch.float16)
            torch_dtype = getattr(torch, torch_dtype)
            hf_model_class = self.backend_config.get("hf_model_class", None)
            if torch_dtype is None:
                logger.warning(
                    f"Invalid torch_dtype: {torch_dtype}. Using torch.float16"
                )
                torch_dtype = torch.float16
            if hf_model_class is None:
                logger.error(
                    f"hf_model_class cannot be None. Please provide a valid model class"
                )
                raise ValueError(
                    "hf_model_class cannot be None. Please provide a valid model class"
                )
            quantization_config = self.backend_config.get(
                "quantization_config", None
            )

            storage_path = os.getenv(
                "STORAGE_PATH", os.path.expanduser("~/models")
            )
            model_path = os.path.join("transformers", self.model_name)
            self.model = load_model(
                model_path,
                device_map=device_map,
                torch_dtype=torch_dtype,
                storage_path=storage_path,
                hf_model_class=hf_model_class,
                quantization_config=quantization_config,
            )
            tokenizer_path = os.path.join(
                storage_path, "transformers", self.model_name, "tokenizer"
            )
            if os.path.exists(tokenizer_path):
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            else:
                # Fall back to load from system's cache
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.pretrained_model_name_or_path
                )
            self.status = BackendStatus.RUNNING

    def _tokenize(self, prompt: str):
        return self.tokenizer(prompt, return_tensors="pt").to("cuda:0")

    def _resolve_request_output_limit(self, requested_output_tokens: int) -> int:
        desired_tokens = max(1, int(requested_output_tokens or 0))
        cap = int(self.backend_config.get("max_output_tokens_cap", 0) or 0)
        if cap > 0:
            desired_tokens = min(desired_tokens, cap)
        return max(1, desired_tokens)

    def _render_messages_prompt(self, messages: List[Dict[str, Any]]) -> str:
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
        max_len = max(32, int(self.backend_config.get("max_model_len", 2048) or 2048))
        max_input_len = max(0, int(self.backend_config.get("max_input_len", 0) or 0))
        prompt_budget = max(8, max_len - int(max_tokens) - 8)
        if max_input_len > 0:
            prompt_budget = min(prompt_budget, max_input_len)
        try:
            token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            if len(token_ids) > prompt_budget:
                logger.warning(
                    "Prompt exceeded budget (%s > %s); truncating to align with fair-run prompt guard",
                    len(token_ids),
                    prompt_budget,
                )
                token_ids = token_ids[-prompt_budget:]
                prompt = self.tokenizer.decode(token_ids, skip_special_tokens=False)
            actual_input_tokens = max(1, len(token_ids))
        except Exception:
            max_chars = min(max_len * 4, 8192)
            if len(prompt) > max_chars:
                prompt = prompt[-max_chars:]
            actual_input_tokens = min(max(1, len(prompt)), prompt_budget)
        max_tokens = min(max_tokens, max(1, max_len - actual_input_tokens - 8))
        return prompt, actual_input_tokens, max(1, max_tokens)

    def _encoder_tokenize(self, query: str, max_length: int):
        return self.tokenizer(
            query,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to("cuda:0")

    def encode(self, request_data: Optional[Dict[str, Any]]):
        with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return {"error": "Model not initialized"}

        def last_token_pool(
            last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
        ) -> torch.Tensor:
            left_padding = (
                attention_mask[:, -1].sum() == attention_mask.shape[0]
            )
            if left_padding:
                return last_hidden_states[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_states.shape[0]
                return last_hidden_states[
                    torch.arange(batch_size, device=last_hidden_states.device),
                    sequence_lengths,
                ]

        def get_detailed_instruct(task_description: str, query: str) -> str:
            return f"Instruct: {task_description}\nQuery: {query}"

        model_name = request_data.get("model", "dummy-model")
        task_instruct = request_data.get("task_instruct", "")
        max_length = request_data.get("max_length", 4096)
        query = request_data.get("input", [])

        if not query:
            return {"error": "Missing query in request data"}

        query = [get_detailed_instruct(task_instruct, q) for q in query]

        batch_dict = self._encoder_tokenize(query, max_length)
        with torch.no_grad():
            output = self.model(**batch_dict, output_hidden_states=True)
        embeddings = last_token_pool(
            output.hidden_states[-1], batch_dict["attention_mask"]
        )

        embeddings = F.normalize(embeddings, p=2, dim=1)

        query_tokens = sum([len(self.tokenizer.tokenize(q)) for q in query])
        response = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": i,
                    "embedding": embeddings[i].tolist(),
                }
                for i in range(len(embeddings))
            ],
            "model": model_name,
            "usage": {
                "query_tokens": query_tokens,
                "total_tokens": query_tokens,
            },
        }

        return response

    def generate(self, request_data: Optional[Dict[str, Any]]):
        with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return {"error": "Model not initialized"}

        assert self.model is not None

        internal_metrics = dict(
            (request_data or {}).get("_sllm_internal_metrics", {}) or {}
        )
        internal_metrics["backend_started_at"] = time.time()

        model_name = request_data.get("model", "dummy-model")
        messages = request_data.get("messages", [])
        temperature = request_data.get("temperature", 0.7)
        max_tokens = request_data.get("max_tokens", 10)
        lora_adapter_name = request_data.get("lora_adapter_name", None)
        request_id = request_data.get("request_id", f"chatcmpl-{uuid.uuid4()}")

        prompt = self._render_messages_prompt(messages)

        if not prompt:
            return {"error": "Missing prompt in request data"}

        prompt, prompt_tokens, max_tokens = self._prepare_prompt(prompt, max_tokens)

        generate_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
        }

        if lora_adapter_name:
            if (
                not hasattr(self.model, "peft_config")
                or lora_adapter_name not in self.model.peft_config
            ):
                return {"error": f"LoRA adapter {lora_adapter_name} not found"}

        inputs = self._tokenize(prompt)
        prompt_tokens = min(prompt_tokens, inputs.input_ids.shape[1])
        logger.debug(
            "generate request_id=%s prompt_shape=%s prompt_tokens=%s adapter=%s",
            request_id,
            tuple(inputs.input_ids.shape),
            prompt_tokens,
            lora_adapter_name,
        )
        streamer = InferenceStatus(
            lambda: self.status,
            prompt_tokens=prompt_tokens,
            request_id=request_id,
            internal_metrics=internal_metrics,
        )
        generate_kwargs["streamer"] = streamer
        with self.active_streamers_lock:
            self.active_streamers[request_id] = streamer

        # Generate response
        try:
            with self.generate_lock:
                if lora_adapter_name and hasattr(self.model, "set_adapter"):
                    self.model.set_adapter(lora_adapter_name)
                elif lora_adapter_name:
                    # Older PEFT releases route per-request adapter selection
                    # through generate(adapter_names=...). Newer releases reject
                    # that kwarg, so prefer set_adapter() when available.
                    generate_kwargs["adapter_names"] = [lora_adapter_name]
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        **generate_kwargs,
                    )
        except DeletingException:
            logger.info("Backend is shutting down. Aborting request")
            output_tokens = streamer.get()
            streamer.delete()
            return {
                "preempted": "True",
                "current_output": output_tokens,
                "completed_tokens": len(output_tokens[0]) - prompt_tokens,
            }
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            raise e
        else:
            output_text = self.tokenizer.decode(
                outputs[0][prompt_tokens:], skip_special_tokens=True
            )
            total_tokens = len(outputs[0])
            metrics = streamer.build_metrics(total_tokens - prompt_tokens)
            streamer.delete()
            return self._build_chat_completion_response(
                request_id=request_id,
                model_name=model_name,
                output_text=output_text,
                prompt_tokens=prompt_tokens,
                total_tokens=total_tokens,
                max_tokens=max_tokens,
                metrics=metrics,
            )
        finally:
            with self.active_streamers_lock:
                self.active_streamers.pop(request_id, None)

    def load_lora_adapter(self, lora_name: str, lora_path: str):
        with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return {"error": "Model not initialized"}

        load_started_at = time.time()

        if (
            hasattr(self.model, "peft_config")
            and lora_name in self.model.peft_config
        ):
            logger.info(f"LoRA adapter {lora_name} already loaded")
            return {
                "cache_hit": True,
                "load_ms": 0.0,
            }

        storage_path = os.getenv("STORAGE_PATH", os.path.expanduser("~/models"))
        if os.path.isabs(lora_path):
            try:
                lora_path = os.path.relpath(lora_path, storage_path)
            except ValueError:
                pass
        elif not lora_path.startswith("transformers" + os.sep):
            lora_path = os.path.join("transformers", lora_path)
        device_map = self.backend_config.get("device_map", "auto")
        torch_dtype = self.backend_config.get("torch_dtype", torch.float16)
        torch_dtype = getattr(torch, torch_dtype)
        if torch_dtype is None:
            logger.warning(
                f"Invalid torch_dtype: {torch_dtype}. Using torch.float16"
            )
            torch_dtype = torch.float16
        self.model = load_lora(
            self.model,
            lora_name,
            lora_path,
            device_map=device_map,
            storage_path=storage_path,
            torch_dtype=torch_dtype,
        )
        logger.info(f"Loaded LoRA adapter {lora_name} from {lora_path}")
        return {
            "cache_hit": False,
            "load_ms": max(0.0, (time.time() - load_started_at) * 1000.0),
        }

    def shutdown(self):
        """Abort all requests and shutdown the backend."""
        with self.status_lock:
            if self.status == BackendStatus.DELETING:
                return
            self.status = BackendStatus.DELETING

        while True:
            with self.active_streamers_lock:
                active_count = len(self.active_streamers)
            if active_count <= 0:
                break
            logger.info("Waiting for all requests to finish")
            time.sleep(1)

        if self.model is not None:
            del self.model

    def stop(self) -> None:
        """Wait for all requests to finish and shutdown the backend."""
        with self.status_lock:
            if self.status.value >= BackendStatus.STOPPING.value:
                return
            self.status = BackendStatus.STOPPING
        while True:
            with self.active_streamers_lock:
                active_count = len(self.active_streamers)
            if active_count <= 0:
                break
            logger.info("Waiting for all requests to finish")
            time.sleep(1)
        logger.info("All requests finished. Shutting down the backend.")
        self.shutdown()

    def get_current_tokens(self) -> List[List[int]]:
        """Return a list of all ongoing request tokens."""
        with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return []

        with self.active_streamers_lock:
            status = [streamer.get()[0] for streamer in self.active_streamers.values() if streamer.get()]
        logger.info(f"Current tokens: {status}")
        return status

    def resume_kv_cache(self, request_datas):
        logger.info(f"Resuming cache for {request_datas}")
        with torch.no_grad():
            device = self.model.device
            input_ids = torch.tensor(request_datas).to(device)
            logger.info(input_ids)
            output = self.model.generate(
                input_ids,
                past_key_values=self.past_key_values,
                max_new_tokens=1,
                return_dict_in_generate=True,
                return_legacy_cache=True,
            )
            self.past_key_values = output.past_key_values
            self.current_tokens = output.sequences
        logger.info(f"Resumed {len(self.past_key_values[0][0][0][0])} tokens")

    def resume_generate(
        self, request_data: Optional[Dict[str, Any]], current_output
    ):
        with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return {"error": "Model not initialized"}

        assert self.model is not None

        model_name = request_data.get("model", "dummy-model")
        messages = request_data.get("messages", [])
        temperature = request_data.get("temperature", 0.7)
        max_tokens = request_data.get("max_tokens", 10)

        prompt = self._render_messages_prompt(messages)

        if not prompt:
            return {"error": "Missing prompt in request data"}

        prompt, prompt_tokens, max_tokens = self._prepare_prompt(prompt, max_tokens)
        inputs = self._tokenize(prompt)
        prompt_tokens = min(prompt_tokens, inputs.input_ids.shape[1])

        # Generate response
        try:
            with torch.no_grad():
                device = self.model.device
                current_output = torch.tensor(current_output).to(device)
                if len(current_output[0]) < len(self.current_tokens[0]):
                    current_output = self.current_tokens
                outputs = self.model.generate(
                    current_output,
                    past_key_values=self.past_key_values,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    streamer=self.inf_status,
                )
        except DeletingException:
            logger.error("Backend is shutting down. Aborting request")
            raise DeletingException("Backend is shutting down")
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            raise e
        else:
            output_text = self.tokenizer.decode(
                outputs[0][prompt_tokens:], skip_special_tokens=True
            )
            total_tokens = len(outputs[0])
            completion_tokens = total_tokens - prompt_tokens
            # FIXME: consider corner case when max_tokens is reached
            finish_reason = (
                "stop" if completion_tokens < max_tokens else "length"
            )

            # Generate response compatible with OpenAI's API
            response = {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": output_text,
                        },
                        "logprobs": None,
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
            }

            self.inf_status.delete()

            return response
