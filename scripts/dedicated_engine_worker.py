#!/usr/bin/env python3
"""
Dedicated engine worker for scale-out.

This process must be launched as a fresh Python interpreter so the runtime's
CUDA-visible GPU set can be pinned before vLLM / torch initialize any CUDA
context.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

# 抑制 Mistral tokenizer 反复打印的已知弃用提示；保留真正的 worker 异常输出。
warnings.filterwarnings(
    "ignore",
    message=r"`get_control_token` is deprecated\. Use `get_special_token` instead\.",
    category=FutureWarning,
    module=r"mistral_common\.tokens\.tokenizers\.tekken",
)
_mistral_warning_filter = "ignore::FutureWarning:mistral_common.tokens.tokenizers.tekken"
_pythonwarnings = os.environ.get("PYTHONWARNINGS", "").strip()
if _mistral_warning_filter not in _pythonwarnings.split(","):
    os.environ["PYTHONWARNINGS"] = (
        f"{_pythonwarnings},{_mistral_warning_filter}" if _pythonwarnings else _mistral_warning_filter
    )


def _write_ready(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _push_env_updates(updates: Dict[str, str]) -> Dict[str, Optional[str]]:
    previous: Dict[str, Optional[str]] = {}
    for key, value in updates.items():
        previous[key] = os.environ.get(key)
        os.environ[key] = value
    return previous


def _restore_env_updates(previous: Dict[str, Optional[str]]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


async def _run_worker(payload_path: Path, ready_path: Path) -> None:
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    repo_root = Path(payload["repo_root"]).resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    model_cfg = dict(payload["model_cfg"])
    cost_model = dict(payload["cost_model"])
    device_id = payload.get("device_id")
    tp = max(1, int(model_cfg.get("tensor_parallel_size", 1) or 1))

    if device_id is not None:
        model_cfg["device_id"] = int(device_id)

    # The parent process has already pinned this worker's exact CUDA-visible GPU
    # set before launching the child. Do not re-derive a second mask here.
    env_updates: Dict[str, str] = {}

    previous_env = _push_env_updates(env_updates)
    try:
        from scripts.run_all_experiments import (
            InferenceEngine,
            _build_local_tp_runtime_env_updates,
        )

        executor_backend = model_cfg.get("distributed_executor_backend")
        previous_env.update(
            _push_env_updates(
                _build_local_tp_runtime_env_updates(
                    tp=tp,
                    executor_backend=str(executor_backend) if executor_backend is not None else None,
                )
            )
        )

        engine = InferenceEngine(model_cfg, cost_model)
        stop_event = asyncio.Event()

        async def _handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            response: Dict[str, Any]
            try:
                line = await reader.readline()
                if not line:
                    return
                rpc = json.loads(line.decode("utf-8"))
                cmd = rpc.get("cmd")
                kwargs = rpc.get("kwargs", {}) or {}
                if cmd == "generate":
                    ttft_ms, tpot_ms, out_tokens = await engine.generate(**kwargs)
                    response = {
                        "ok": True,
                        "result": {
                            "ttft_ms": ttft_ms,
                            "tpot_ms": tpot_ms,
                            "output_tokens": out_tokens,
                        },
                    }
                elif cmd == "load_lora_to_gpu_and_measure":
                    load_ms, ok = await engine.load_lora_to_gpu_and_measure(**kwargs)
                    response = {"ok": True, "result": {"load_ms": load_ms, "ok": ok}}
                elif cmd == "shutdown":
                    response = {"ok": True}
                    stop_event.set()
                else:
                    response = {"ok": False, "error": f"unknown_cmd:{cmd}"}
            except Exception as exc:  # pragma: no cover - exercised via parent integration
                response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

            try:
                writer.write((json.dumps(response, ensure_ascii=True) + "\n").encode("utf-8"))
                await writer.drain()
            finally:
                writer.close()
                try:
                    await writer.wait_closed()
                except Exception:
                    pass

        try:
            await engine.initialize()
            server = await asyncio.start_server(_handle_client, host="127.0.0.1", port=0)
            sock = server.sockets[0]
            host, port = sock.getsockname()[:2]
            _write_ready(ready_path, {"status": "ready", "host": host, "port": int(port)})
            async with server:
                await stop_event.wait()
        finally:
            try:
                await engine.shutdown()
            except Exception:
                pass
    finally:
        _restore_env_updates(previous_env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True)
    parser.add_argument("--ready-file", required=True)
    args = parser.parse_args()

    payload_path = Path(args.payload).resolve()
    ready_path = Path(args.ready_file).resolve()

    try:
        asyncio.run(_run_worker(payload_path, ready_path))
    except Exception as exc:
        try:
            _write_ready(ready_path, {"status": "error", "error": f"{type(exc).__name__}: {exc}"})
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
