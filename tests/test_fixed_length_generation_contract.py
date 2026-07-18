from __future__ import annotations

import ast
import asyncio
import copy
import json
import sys
import time
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SLORA_HTTP_MANAGER = (
    ROOT / "repos" / "S-LoRA" / "slora" / "server" / "httpserver" / "manager.py"
)
sys.path.insert(0, str(ROOT / "scripts"))

import replay_openai_trace as replay  # noqa: E402
import validate_replay_results as validate  # noqa: E402


class FixedLengthGenerationContractTest(unittest.TestCase):
    def test_prompt_guard_extreme_branch_returns_final_decoded_prompt(self) -> None:
        class ExpandingTokenizer:
            def encode(self, prompt, *, add_special_tokens=False):
                del add_special_tokens
                if prompt == "original":
                    return list(range(40))
                if prompt == "wide":
                    return list(range(100))
                if prompt == "single":
                    return [7, 8]
                if prompt == "final":
                    return [7, 8]
                raise AssertionError(f"unexpected prompt: {prompt!r}")

            def decode(self, token_ids, *, skip_special_tokens=False):
                del skip_special_tokens
                if len(token_ids) == 32:
                    return "wide"
                if len(token_ids) == 1:
                    return "single"
                if list(token_ids) == [7, 8]:
                    return "final"
                raise AssertionError(f"unexpected token ids: {token_ids!r}")

        with mock.patch.object(
            replay,
            "_get_prompt_guard_tokenizer",
            return_value=ExpandingTokenizer(),
        ):
            prompt, max_tokens, input_tokens = replay._apply_faaslora_style_prompt_guard(
                prompt="original",
                requested_output_tokens=16,
                tokenizer_model="fake",
                max_model_len=64,
                max_input_len=32,
                max_output_tokens_cap=16,
            )

        self.assertEqual(prompt, "final")
        self.assertEqual(input_tokens, 2)
        self.assertEqual(max_tokens, 16)

    def test_target_uses_source_expected_tokens_and_cap(self) -> None:
        self.assertEqual(replay._fixed_length_target(17, 256), 17)
        self.assertEqual(replay._fixed_length_target(400, 256), 256)
        with self.assertRaises(RuntimeError):
            replay._fixed_length_target(0, 256)
        with self.assertRaises(RuntimeError):
            replay._fixed_length_target(None, 256)

    def test_slora_sse_counts_only_integer_token_ids(self) -> None:
        generated_text: list[str] = []
        generated_ids: list[int] = []
        stats: dict[str, int] = {}
        server_metrics: dict[str, object] = {}

        replay._apply_response_payload(
            {"token": {"id": 7, "text": "x"}},
            generated_text_parts=generated_text,
            generated_token_ids=generated_ids,
            native_token_stats=stats,
            server_metrics=server_metrics,
        )
        replay._apply_response_payload(
            {"token": {"id": True, "text": "y"}},
            generated_text_parts=generated_text,
            generated_token_ids=generated_ids,
            native_token_stats=stats,
            server_metrics=server_metrics,
        )

        self.assertEqual(generated_text, ["x", "y"])
        self.assertEqual(generated_ids, [7])
        self.assertEqual(stats["token_events"], 2)
        self.assertEqual(stats["integer_token_ids"], 1)
        self.assertEqual(stats["invalid_token_ids"], 1)

    def test_replay_uses_sse_ids_as_primary_fixed_length_count(self) -> None:
        class FakeResponse:
            status_code = 200
            encoding = "utf-8"

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def iter_content(self, **_kwargs):
                for token_id, text in ((11, "a"), (12, "b"), (13, "c")):
                    payload = {
                        "token": {"id": token_id, "text": text},
                        "generated_text": None,
                        "finished": token_id == 13,
                    }
                    yield f"data:{json.dumps(payload)}\n\n"

        item = {
            "request_id": "req_00000",
            "arrival_time_s": 0.0,
            "adapter_id": "adapter_0000",
            "expected_input_tokens": 2,
            "expected_output_tokens": 3,
            "body": {
                "messages": [{"role": "user", "content": "hello"}],
                "stop": ["bad"],
                "stop_sequences": ["also bad"],
            },
        }
        with (
            mock.patch.object(replay.requests, "post", return_value=FakeResponse()) as post,
            mock.patch.object(replay, "_render_chat_messages_prompt", return_value="prompt"),
            mock.patch.object(
                replay,
                "_apply_faaslora_style_prompt_guard",
                return_value=("prompt", 3, 2),
            ),
            mock.patch.object(replay, "_count_text_tokens", return_value=99),
        ):
            result = replay._replay_one(
                base_url="http://example.test",
                item=item,
                request_index=0,
                timeout_s=1.0,
                start_time=time.perf_counter(),
                base_cost_usd=0.0,
                input_token_cost_usd=0.0,
                output_token_cost_usd=0.0,
                require_server_metrics=False,
                model_override=None,
                adapter_source_field="adapter_id",
                adapter_target_field="lora_dir",
                adapter_value_map={"adapter_0000": "/tmp/adapter_0000"},
                drop_body_fields=[],
                endpoint_path="/generate_stream",
                convert_chat_to_prompt=True,
                prompt_guard_tokenizer_model="fake-tokenizer",
                prompt_guard_max_model_len=1024,
                prompt_guard_max_input_len=0,
                prompt_guard_max_output_tokens_cap=0,
                sglang_native_generate=False,
                slora_native_generate=True,
                generation_seed=42,
                empty_success_retries=0,
                empty_success_retry_delay_s=0.0,
                min_output_tokens=0,
                include_stream_usage=False,
                force_stream=False,
                generation_contract=replay.GENERATION_CONTRACT_FIXED_LENGTH_GREEDY_V1,
                fixed_output_max_tokens=256,
                fixed_prompt_max_tokens=759,
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["requested_completion_tokens"], 3)
        self.assertEqual(result["completion_tokens"], 3)
        self.assertEqual(result["completion_tokens_text_audit"], 99)
        self.assertEqual(result["completion_token_source"], "slora_native_sse_token_id")
        self.assertTrue(result["output_contract_match"])
        self.assertEqual(result["native_sse_integer_token_id_count"], 3)
        request_body = post.call_args.kwargs["json"]
        params = request_body["parameters"]
        self.assertFalse(params["do_sample"])
        self.assertTrue(params["ignore_eos"])
        self.assertEqual(params["max_new_tokens"], 3)
        self.assertEqual(params["temperature"], 0.0)
        self.assertEqual(params["top_p"], 1.0)
        self.assertNotIn("stop_sequences", params)

    def _valid_payload(self) -> dict[str, object]:
        result = {
            "request_id": "req_00001",
            "adapter_id": "adapter_0001",
            "arrival_time_s": 0.25,
            "success": True,
            "generation_contract": validate.FIXED_LENGTH_GREEDY_V1,
            "source_expected_output_tokens": 3,
            "requested_completion_tokens": 3,
            "completion_tokens": 3,
            "output_contract_match": True,
            "completion_token_source": "slora_native_sse_token_id",
            "native_sse_token_event_count": 3,
            "native_sse_integer_token_id_count": 3,
            "native_sse_invalid_token_id_count": 0,
            "canonical_prompt_sha256": "a" * 64,
            "completion_token_ids_sha256": "b" * 64,
            "guard_prompt_tokens": 64,
            "e2e_ms": 110.0,
            "dispatch_admission_wait_ms": 10.0,
            "service_e2e_ms": 100.0,
            "service_ttft_ms": 4.0,
            "tpot_ms": 48.0,
        }
        request_map = [
            {
                "request_id": result["request_id"],
                "adapter_id": result["adapter_id"],
                "arrival_time_s": result["arrival_time_s"],
                "source_expected_output_tokens": result[
                    "source_expected_output_tokens"
                ],
                "requested_completion_tokens": result[
                    "requested_completion_tokens"
                ],
                "canonical_prompt_sha256": result["canonical_prompt_sha256"],
                "canonical_prompt_tokens": result["guard_prompt_tokens"],
            }
        ]
        return {
            "generation_contract": validate.FIXED_LENGTH_GREEDY_V1,
            "generation_contract_policy": {
                "target_formula": (
                    "min(source_expected_output_tokens, fixed_output_max_tokens)"
                ),
                "fixed_output_max_tokens": 256,
                "fixed_prompt_max_tokens": 759,
                "temperature": 0.0,
                "top_p": 1.0,
                "ignore_eos": True,
                "stop_sequences": [],
                "completion_token_source": "slora_native_sse_token_id",
            },
            "generation_contract_request_map_sha256": (
                validate._sha256_canonical_json(request_map)
            ),
            "results": [result],
        }

    def test_validator_accepts_complete_contract(self) -> None:
        payload = self._valid_payload()
        errors = validate._validate_fixed_length_greedy_v1(
            system="S-LoRA",
            payload=payload,
            results=payload["results"],
            fixed_output_max_tokens=256,
            fixed_prompt_max_tokens=759,
        )
        self.assertEqual(errors, [])

    def test_validator_rejects_token_mismatch(self) -> None:
        payload = copy.deepcopy(self._valid_payload())
        payload["results"][0]["completion_tokens"] = 2
        payload["results"][0]["output_contract_match"] = False
        errors = validate._validate_fixed_length_greedy_v1(
            system="S-LoRA",
            payload=payload,
            results=payload["results"],
            fixed_output_max_tokens=256,
            fixed_prompt_max_tokens=759,
        )
        self.assertTrue(any("completion_tokens=2" in error for error in errors))
        self.assertTrue(any("output_contract_match" in error for error in errors))


class SloraStreamFifoRegressionTest(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _load_http_manager_class(batch_str_out, batch_abort_req):
        """Load only HttpServerManager so this test needs no S-LoRA runtime deps."""

        module = ast.parse(SLORA_HTTP_MANAGER.read_text(encoding="utf-8"))
        class_node = next(
            node
            for node in module.body
            if isinstance(node, ast.ClassDef) and node.name == "HttpServerManager"
        )
        isolated = ast.fix_missing_locations(
            ast.Module(body=[class_node], type_ignores=[])
        )
        namespace = {
            "asyncio": asyncio,
            "AbortReq": lambda req_id: {"req_id": req_id},
            "BatchStrOut": batch_str_out,
            "BatchAbortReq": batch_abort_req,
            "Union": lambda *_args: object,
        }
        exec(compile(isolated, str(SLORA_HTTP_MANAGER), "exec"), namespace)
        return namespace["HttpServerManager"]

    async def test_detokenizer_burst_preserves_concurrent_sse_token_events(self) -> None:
        class BatchStrOut:
            def __init__(self, reqs_infs):
                self.reqs_infs = reqs_infs

        class BatchAbortReq:
            def __init__(self, reqs):
                self.reqs = reqs

        class Tokenizer:
            @staticmethod
            def encode(_prompt):
                return [1, 2]

        class SamplingParams:
            def __init__(self, max_new_tokens):
                self.max_new_tokens = max_new_tokens

            @staticmethod
            def stop_sentences_to_token_ids(_tokenizer):
                return None

        class Sender:
            def __init__(self):
                self.messages = []
                self.manager = None

            def send_pyobj(self, message):
                if isinstance(message, tuple):
                    request_id = message[-1]
                    if request_id not in self.manager.req_id_to_out_inf:
                        raise AssertionError("request queue must be registered before send")
                self.messages.append(message)

        class Receiver:
            def __init__(self, first_message):
                self.first_message = first_message
                self.sent = False
                self.block = asyncio.Event()

            async def recv_pyobj(self):
                if not self.sent:
                    self.sent = True
                    return self.first_message
                await self.block.wait()

        manager_class = self._load_http_manager_class(BatchStrOut, BatchAbortReq)
        manager = object.__new__(manager_class)
        manager.tokenizer = Tokenizer()
        manager.total_token_num = 512
        manager.max_req_input_len = 32
        manager.max_req_total_len = 512
        manager.req_id_to_out_inf = {}
        manager.send_to_router = Sender()
        manager.send_to_router.manager = manager
        request_lengths = {"one": 1, "medium": 87, "large": 256, "peer": 16}
        token_ids = {
            request_id: [2] + [1000 + index for index in range(1, length)]
            for request_id, length in request_lengths.items()
        }
        # One detokenizer message can contain a burst for several requests.
        # Token id 2 is intentionally non-final for lengths > 1: ignore_eos is
        # enforced upstream and this transport must not coalesce that event.
        burst = []
        for request_id, length in request_lengths.items():
            burst.extend(
                (
                    request_id,
                    f"token-{index}",
                    {"id": token_id},
                    index == length - 1,
                    False,
                )
                for index, token_id in enumerate(token_ids[request_id])
            )
        manager.recv_from_detokenization = Receiver(
            BatchStrOut(burst)
        )

        outputs = {
            request_id: manager.generate(
                "adapter",
                "prompt",
                SamplingParams(length),
                request_id,
            )
            for request_id, length in request_lengths.items()
        }
        first_outputs = {
                request_id: asyncio.create_task(output.__anext__())
            for request_id, output in outputs.items()
        }
        for _ in range(100):
            if set(manager.req_id_to_out_inf) == set(request_lengths):
                break
            await asyncio.sleep(0)
        self.assertEqual(set(manager.req_id_to_out_inf), set(request_lengths))

        handle_task = asyncio.create_task(manager.handle_loop())
        try:
            for request_id, length in request_lengths.items():
                observed = [
                    await asyncio.wait_for(first_outputs[request_id], timeout=1)
                ]
                for _ in range(length - 1):
                    observed.append(
                        await asyncio.wait_for(
                            outputs[request_id].__anext__(), timeout=1
                        )
                    )
                with self.assertRaises(StopAsyncIteration):
                    await asyncio.wait_for(outputs[request_id].__anext__(), timeout=1)
                self.assertEqual(
                    [item[1]["id"] for item in observed],
                    token_ids[request_id],
                )
                self.assertEqual(
                    [item[2] for item in observed],
                    [False] * (length - 1) + [True],
                )
        finally:
            handle_task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await handle_task

        self.assertEqual(manager.req_id_to_out_inf, {})

    async def test_abort_wakes_generator_and_cleans_request_queue(self) -> None:
        class BatchStrOut:
            pass

        class BatchAbortReq:
            pass

        class Tokenizer:
            @staticmethod
            def encode(_prompt):
                return [1, 2]

        class SamplingParams:
            max_new_tokens = 8

            @staticmethod
            def stop_sentences_to_token_ids(_tokenizer):
                return None

        class Sender:
            def __init__(self):
                self.messages = []

            def send_pyobj(self, message):
                self.messages.append(message)

        manager_class = self._load_http_manager_class(BatchStrOut, BatchAbortReq)
        manager = object.__new__(manager_class)
        manager.tokenizer = Tokenizer()
        manager.total_token_num = 64
        manager.max_req_input_len = 32
        manager.max_req_total_len = 64
        manager.req_id_to_out_inf = {}
        manager.send_to_router = Sender()

        output = manager.generate("adapter", "prompt", SamplingParams(), "abort-me")
        first_output = asyncio.create_task(output.__anext__())
        for _ in range(100):
            if "abort-me" in manager.req_id_to_out_inf:
                break
            await asyncio.sleep(0)
        self.assertIn("abort-me", manager.req_id_to_out_inf)

        await manager.abort("abort-me")
        self.assertEqual(await asyncio.wait_for(first_output, timeout=1), ("", {}, -1))
        with self.assertRaises(StopAsyncIteration):
            await asyncio.wait_for(output.__anext__(), timeout=1)
        self.assertNotIn("abort-me", manager.req_id_to_out_inf)
        self.assertEqual(manager.send_to_router.messages[-1], {"req_id": "abort-me"})


if __name__ == "__main__":
    unittest.main()
