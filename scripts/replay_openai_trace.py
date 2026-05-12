#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import threading
import tarfile
import tempfile
import time
import urllib.parse
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

LIVE_PRINT_INTERVAL_S = 2.0
_PROMPT_GUARD_TOKENIZER_CACHE: Dict[str, Any] = {}
_PROMPT_GUARD_TOKENIZER_LOCK = threading.Lock()
METRIC_DEF_PRIMARY_TTFT = (
    "scheduled trace arrival to system-observed first generated output token/chunk"
)
METRIC_DEF_PRIMARY_E2E = (
    "scheduled trace arrival to system-observed response completion when available; "
    "otherwise client-observed completion"
)
METRIC_DEF_SERVICE_TTFT = (
    "service-path start to first generated output token/chunk; service-path start "
    "is the system-specific admitted/backend-execution start and excludes scheduled-arrival "
    "and upstream queue/admission wait"
)
METRIC_DEF_SERVICE_E2E = (
    "service-path start to response completion; service-path start is the system-specific "
    "admitted/backend-execution start and excludes scheduled-arrival and upstream "
    "queue/admission wait"
)


class DynamicLoRALoader:
    """Thread-safe per-endpoint LoRA loader for runtime LoRA APIs."""

    def __init__(
        self,
        *,
        modules: Dict[str, str],
        timeout_s: float,
        max_loaded_per_endpoint: int,
        load_path: str = "/v1/load_lora_adapter",
        unload_path: str = "/v1/unload_lora_adapter",
        remote_fetcher: Optional["RemoteArtifactFetcher"] = None,
    ) -> None:
        self._modules = dict(modules)
        self._timeout_s = max(1.0, float(timeout_s))
        self._max_loaded_per_endpoint = max(0, int(max_loaded_per_endpoint or 0))
        self._load_path = "/" + str(load_path or "/v1/load_lora_adapter").lstrip("/")
        self._unload_path = "/" + str(unload_path or "/v1/unload_lora_adapter").lstrip("/")
        self._remote_fetcher = remote_fetcher
        self._loaded: Dict[str, OrderedDict[str, None]] = {}
        self._active: Dict[str, Dict[str, int]] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

    def _endpoint_lock(self, base_url: str) -> threading.Lock:
        key = base_url.rstrip("/")
        with self._global_lock:
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._locks[key] = lock
            return lock

    def _unload_adapter(self, key: str, adapter: str) -> float:
        endpoint = f"{key}{self._unload_path}"
        payload = {"lora_name": adapter}
        t0 = time.perf_counter()
        try:
            resp = requests.post(endpoint, json=payload, timeout=self._timeout_s)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"dynamic LoRA unload failed for {adapter} on {key}: {type(exc).__name__}: {exc}"
            ) from exc
        if resp.status_code not in (200, 201, 204):
            body = resp.text[:800]
            lowered = body.lower()
            if resp.status_code in (400, 404) and (
                "not" in lowered and ("loaded" in lowered or "found" in lowered)
            ):
                return elapsed_ms
            raise RuntimeError(
                f"dynamic LoRA unload failed for {adapter} on {key}: HTTP {resp.status_code}: {body}"
            )
        return elapsed_ms

    def _evict_inactive(self, key: str, loaded: OrderedDict[str, None]) -> Dict[str, Any]:
        if self._max_loaded_per_endpoint <= 0:
            return {"dynamic_lora_unloaded_count": 0, "dynamic_lora_unload_ms": 0.0}
        active = self._active.setdefault(key, {})
        unloaded_count = 0
        unload_ms = 0.0
        while len(loaded) >= self._max_loaded_per_endpoint:
            victim = None
            for candidate in loaded.keys():
                if active.get(candidate, 0) <= 0:
                    victim = candidate
                    break
            if victim is None:
                break
            unload_ms += self._unload_adapter(key, victim)
            loaded.pop(victim, None)
            active.pop(victim, None)
            unloaded_count += 1
        return {
            "dynamic_lora_unloaded_count": unloaded_count,
            "dynamic_lora_unload_ms": unload_ms,
        }

    def acquire(self, base_url: str, adapter_id: Optional[str]) -> Dict[str, Any]:
        adapter = str(adapter_id or "").strip()
        if not adapter:
            return {
                "dynamic_lora_loaded": False,
                "dynamic_lora_load_ms": 0.0,
                "dynamic_lora_unloaded_count": 0,
                "dynamic_lora_unload_ms": 0.0,
            }
        path = self._modules.get(adapter)
        if not path:
            raise RuntimeError(f"adapter {adapter!r} is absent from dynamic LoRA module map")
        key = base_url.rstrip("/")
        lock = self._endpoint_lock(key)
        with lock:
            remote_metrics: Dict[str, Any] = {}
            if self._remote_fetcher is not None:
                remote_metrics = self._remote_fetcher.ensure(adapter, path)
            loaded = self._loaded.setdefault(key, OrderedDict())
            active = self._active.setdefault(key, {})
            if adapter in loaded:
                loaded.move_to_end(adapter)
                active[adapter] = active.get(adapter, 0) + 1
                return {
                    "dynamic_lora_loaded": False,
                    "dynamic_lora_load_ms": 0.0,
                    "dynamic_lora_unloaded_count": 0,
                    "dynamic_lora_unload_ms": 0.0,
                    "dynamic_lora_resident_count": len(loaded),
                    "dynamic_lora_max_resident": self._max_loaded_per_endpoint,
                    **remote_metrics,
                }
            eviction_metrics = self._evict_inactive(key, loaded)
            endpoint = f"{key}{self._load_path}"
            payload = {"lora_name": adapter, "lora_path": path}
            t0 = time.perf_counter()
            try:
                resp = requests.post(endpoint, json=payload, timeout=self._timeout_s)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"dynamic LoRA load failed for {adapter} on {key}: {type(exc).__name__}: {exc}"
                ) from exc
            if resp.status_code not in (200, 201, 204):
                body = resp.text[:800]
                if resp.status_code == 400 and "already" in body.lower() and "loaded" in body.lower():
                    loaded[adapter] = None
                    loaded.move_to_end(adapter)
                    active[adapter] = active.get(adapter, 0) + 1
                    return {
                        "dynamic_lora_loaded": False,
                        "dynamic_lora_load_ms": elapsed_ms,
                        "dynamic_lora_already_loaded": True,
                        "dynamic_lora_resident_count": len(loaded),
                        "dynamic_lora_max_resident": self._max_loaded_per_endpoint,
                        **eviction_metrics,
                        **remote_metrics,
                    }
                raise RuntimeError(
                    f"dynamic LoRA load failed for {adapter} on {key}: HTTP {resp.status_code}: {body}"
                )
            loaded[adapter] = None
            loaded.move_to_end(adapter)
            active[adapter] = active.get(adapter, 0) + 1
            return {
                "dynamic_lora_loaded": True,
                "dynamic_lora_load_ms": elapsed_ms,
                "dynamic_lora_resident_count": len(loaded),
                "dynamic_lora_max_resident": self._max_loaded_per_endpoint,
                **eviction_metrics,
                **remote_metrics,
            }

    def release(self, base_url: str, adapter_id: Optional[str]) -> None:
        adapter = str(adapter_id or "").strip()
        if not adapter:
            return
        key = base_url.rstrip("/")
        lock = self._endpoint_lock(key)
        with lock:
            active = self._active.setdefault(key, {})
            count = active.get(adapter, 0)
            if count <= 1:
                active.pop(adapter, None)
            else:
                active[adapter] = count - 1


def _load_adapter_path_map(path: Path) -> Dict[str, str]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise RuntimeError(f"adapter path map is empty: {path}")
    mapping: Dict[str, str] = {}
    if raw.startswith("{"):
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise RuntimeError(f"adapter path map JSON must be an object: {path}")
        for key, value in payload.items():
            mapping[str(key)] = str(value)
    elif raw.startswith("["):
        payload = json.loads(raw)
        if not isinstance(payload, list):
            raise RuntimeError(f"adapter path map JSON must be a list: {path}")
        for item in payload:
            if isinstance(item, str) and "=" in item:
                name, value = item.split("=", 1)
            elif isinstance(item, dict):
                name = item.get("name") or item.get("id") or item.get("adapter_id")
                value = item.get("path") or item.get("lora_path") or item.get("lora_dir")
            else:
                name = value = None
            if not name or not value:
                raise RuntimeError(f"invalid adapter path map entry in {path}: {item!r}")
            mapping[str(name)] = str(value)
    else:
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                raise RuntimeError(f"invalid adapter path map line in {path}: {line!r}")
            name, value = line.split("=", 1)
            mapping[str(name)] = str(value)
    if not mapping:
        raise RuntimeError(f"adapter path map has no entries: {path}")
    return mapping


class RemoteArtifactFetcher:
    """Materialize LoRA adapters from a remote source on demand.

    The endpoint may be an HTTP artifact node (``http://host:port``) or a
    local simulated remote root (``file:///path/to/frozen_pool``).  The latter
    is intentionally supported for remote-fair local diagnostics: it starts
    from an empty local cache, copies the adapter from the configured remote
    pool on first touch, and optionally sleeps according to the same
    MiB-per-second bandwidth model used by the PrimeLoRA runner.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        timeout_s: float,
        token_env: str,
        bandwidth_mbps: float = 0.0,
    ) -> None:
        endpoint = str(endpoint or "").strip().rstrip("/")
        if not endpoint:
            raise RuntimeError("remote artifact endpoint must be non-empty")
        self._endpoint = endpoint
        self._parsed_endpoint = urllib.parse.urlparse(endpoint)
        self._timeout_s = max(1.0, float(timeout_s or 300.0))
        self._token = os.getenv(str(token_env or "PRIME_REMOTE_TOKEN"), "")
        self._bandwidth_mbps = max(0.0, float(bandwidth_mbps or 0.0))
        self._session = requests.Session()
        self._session.trust_env = False
        self._locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

    def _lock_for(self, adapter_id: str) -> threading.Lock:
        with self._global_lock:
            lock = self._locks.get(adapter_id)
            if lock is None:
                lock = threading.Lock()
                self._locks[adapter_id] = lock
            return lock

    def ensure(self, adapter_id: str, target_path: str) -> Dict[str, Any]:
        adapter = str(adapter_id or "").strip()
        target = Path(target_path).expanduser().resolve()
        if _looks_like_materialized_adapter(target):
            return {
                "remote_lora_fetched": False,
                "remote_lora_fetch_ms": 0.0,
                "remote_lora_bytes": _path_size(target),
                "remote_lora_endpoint": self._endpoint,
            }
        lock = self._lock_for(adapter)
        with lock:
            if _looks_like_materialized_adapter(target):
                return {
                    "remote_lora_fetched": False,
                    "remote_lora_fetch_ms": 0.0,
                    "remote_lora_bytes": _path_size(target),
                    "remote_lora_endpoint": self._endpoint,
                }
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp_root = Path(tempfile.mkdtemp(prefix="remote-lora-fetch-"))
            archive = tmp_root / f"{adapter}.tar.gz"
            extract_dir = tmp_root / "extract"
            extract_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.perf_counter()
            try:
                parsed = self._parsed_endpoint
                size_bytes = 0
                local_sim_mode = ""
                if parsed.scheme == "file":
                    src_root = Path(urllib.parse.unquote(parsed.path)).expanduser().resolve()
                    src = src_root / adapter
                    if not _looks_like_materialized_adapter(src):
                        raise RuntimeError(f"missing simulated remote adapter: {src}")
                    size_bytes = _path_size(src)
                    if self._bandwidth_mbps > 0 and size_bytes > 0:
                        # Match the existing PrimeLoRA local-remote simulation:
                        # bandwidth_mbps is treated as MiB/s despite the legacy
                        # name, so an adapter of X MiB sleeps X / bandwidth.
                        time.sleep((size_bytes / (1024.0 * 1024.0)) / self._bandwidth_mbps)
                    local_sim_mode = _linktree(src, extract_dir)
                else:
                    url = f"{self._endpoint}/artifacts/{urllib.parse.quote(adapter, safe='')}.tar.gz"
                    headers = {"Accept": "application/gzip"}
                    if self._token:
                        headers["Authorization"] = f"Bearer {self._token}"
                    with self._session.get(url, headers=headers, timeout=self._timeout_s, stream=True) as resp:
                        resp.raise_for_status()
                        with archive.open("wb") as fh:
                            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                                if chunk:
                                    fh.write(chunk)
                    with tarfile.open(archive, "r:gz") as tar:
                        _safe_extract_tar(tar, extract_dir)
                    size_bytes = _path_size(extract_dir)
                if target.exists():
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                shutil.move(str(extract_dir), str(target))
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                return {
                    "remote_lora_fetched": True,
                    "remote_lora_fetch_ms": elapsed_ms,
                    "remote_lora_bytes": size_bytes or _path_size(target),
                    "remote_lora_endpoint": self._endpoint,
                    "remote_lora_local_sim_mode": local_sim_mode,
                }
            finally:
                shutil.rmtree(tmp_root, ignore_errors=True)


def _looks_like_materialized_adapter(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return (path / "adapter_config.json").exists() and any(
        child.name.startswith("adapter_") or child.suffix in {".safetensors", ".bin"}
        for child in path.iterdir()
    )


def _path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return int(path.stat().st_size)
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += int(child.stat().st_size)
    return total


def _linktree(src: Path, dst: Path) -> str:
    """Mirror a local simulated remote tree without charging disk copy as I/O."""
    if dst.exists():
        shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
    dst.mkdir(parents=True, exist_ok=True)
    mode = "hardlink"
    for item in src.rglob("*"):
        rel = item.relative_to(src)
        out = dst / rel
        if item.is_dir():
            out.mkdir(parents=True, exist_ok=True)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        if item.is_symlink():
            os.symlink(os.readlink(item), out)
            mode = "symlink" if mode == "hardlink" else mode
            continue
        try:
            os.link(item, out)
        except OSError:
            try:
                os.symlink(item, out)
                mode = "symlink"
            except OSError:
                shutil.copy2(item, out)
                mode = "copy"
    return mode


def _safe_extract_tar(tar: tarfile.TarFile, target: Path) -> None:
    root = target.resolve()
    for member in tar.getmembers():
        dest = (target / member.name).resolve()
        if dest != root and not str(dest).startswith(str(root) + os.sep):
            raise RuntimeError(f"unsafe archive member path: {member.name}")
    try:
        tar.extractall(target)
    except TypeError:
        tar.extractall(target)


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = max(0.0, min(1.0, float(q) / 100.0)) * (len(ordered) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    frac = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def _render_progress_bar(completed: int, total: int, width: int = 28) -> str:
    total = max(1, int(total))
    completed = max(0, min(int(completed), total))
    filled = int(round((completed / total) * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def _fmt_optional_ms(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.0f}ms"


def _fmt_optional_triplet(avg: Optional[float], p95: Optional[float], p99: Optional[float]) -> str:
    if avg is None or p95 is None or p99 is None:
        return "n/a/n/a/n/a"
    return f"{float(avg):.0f}/{float(p95):.0f}/{float(p99):.0f}ms"


def _fmt_optional_pair(avg: Optional[float], p95: Optional[float]) -> str:
    if avg is None or p95 is None:
        return "n/a/n/a"
    return f"{float(avg):.0f}/{float(p95):.0f}ms"


def _calc_cost(in_tok: int, out_tok: int, *, base: float, in_cost: float, out_cost: float) -> float:
    return float(base) + float(in_cost) * max(0, int(in_tok)) + float(out_cost) * max(0, int(out_tok))


def _cost_per_million_tokens(total_cost_usd: float, token_count: int) -> float:
    try:
        cost = float(total_cost_usd or 0.0)
        tokens = int(token_count or 0)
    except Exception:
        return 0.0
    if cost <= 0.0 or tokens <= 0:
        return 0.0
    return cost * 1_000_000.0 / float(tokens)


def _parse_base_urls(primary: str, extra_csv: Optional[str]) -> List[str]:
    urls: List[str] = []
    if primary:
        urls.append(str(primary).strip())
    if extra_csv:
        urls.extend(item.strip() for item in str(extra_csv).split(",") if item.strip())
    normalized: List[str] = []
    seen = set()
    for url in urls:
        clean = url.rstrip("/")
        if clean and clean not in seen:
            normalized.append(clean)
            seen.add(clean)
    if not normalized:
        raise RuntimeError("at least one --base-url is required")
    return normalized


def _stable_hash_int(value: str) -> int:
    digest = hashlib.sha1(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _choose_dynamic_lora_base_url(
    *,
    base_urls: List[str],
    request_index: int,
    adapter_id: Optional[Any],
    request_id: Optional[Any],
    routing: str,
) -> str:
    if not base_urls:
        raise RuntimeError("no base URLs available for replay")
    if len(base_urls) == 1:
        return base_urls[0]
    adapter = str(adapter_id or "").strip()
    if routing == "round_robin" or not adapter:
        return base_urls[request_index % len(base_urls)]
    if routing in ("adapter_hash", "adaptive_hot_pair_hash"):
        return base_urls[_stable_hash_int(adapter) % len(base_urls)]
    if routing == "adapter_pair_hash":
        first = _stable_hash_int(adapter) % len(base_urls)
        second_offset = 1 + (_stable_hash_int(f"{adapter}|pair") % (len(base_urls) - 1))
        second = (first + second_offset) % len(base_urls)
        pair = (first, second)
        selector_key = f"{adapter}|{request_id if request_id is not None else request_index}"
        return base_urls[pair[_stable_hash_int(selector_key) % 2]]
    raise RuntimeError(f"unknown dynamic LoRA routing policy: {routing}")


def _derive_request_generation_seed(
    base_seed: Optional[int],
    request_id: Optional[str],
    request_index: int,
) -> Optional[int]:
    if base_seed is None:
        return None
    try:
        seed = int(base_seed)
    except (TypeError, ValueError):
        return None
    rid = str(request_id or "").strip()
    match = re.search(r"(\d+)$", rid)
    if match:
        offset = int(match.group(1))
    elif rid:
        offset = int(hashlib.sha1(rid.encode("utf-8")).hexdigest()[:8], 16)
    else:
        offset = int(request_index)
    return int((seed + offset) % (2**32))


def _known_bool_rate(records: List[Dict[str, Any]], key: str) -> Optional[float]:
    known = [record for record in records if record.get(key) is not None]
    if not known:
        return None
    return sum(1 for record in known if bool(record.get(key))) / len(known)


def _render_messages_fallback(messages: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for message in messages:
        item = dict(message or {})
        role = str(item.get("role") or "user").strip().capitalize() or "User"
        content = item.get("content")
        if not isinstance(content, str):
            content = "" if content is None else str(content)
        lines.append(f"{role}: {content}")
    return "\n".join(lines).strip()


def _render_chat_messages_prompt(
    messages: List[Dict[str, Any]],
    *,
    tokenizer_model: Optional[str],
) -> str:
    if not messages:
        return ""
    if tokenizer_model:
        try:
            tokenizer = _get_prompt_guard_tokenizer(tokenizer_model)
            rendered = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            if isinstance(rendered, str) and rendered.strip():
                return rendered
        except Exception:
            pass
    return _render_messages_fallback(messages)


def _get_prompt_guard_tokenizer(model_name: str):
    cached = _PROMPT_GUARD_TOKENIZER_CACHE.get(model_name)
    if cached is not None:
        return cached
    with _PROMPT_GUARD_TOKENIZER_LOCK:
        cached = _PROMPT_GUARD_TOKENIZER_CACHE.get(model_name)
        if cached is not None:
            return cached
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        _PROMPT_GUARD_TOKENIZER_CACHE[model_name] = tokenizer
        return tokenizer


def _count_text_tokens(text: str, tokenizer_model: Optional[str]) -> Optional[int]:
    if not tokenizer_model:
        return None
    try:
        tokenizer = _get_prompt_guard_tokenizer(tokenizer_model)
        return len(tokenizer.encode(str(text or ""), add_special_tokens=False))
    except Exception:
        return None


def _extract_generated_text_fragment(obj: Dict[str, Any]) -> str:
    """Extract generated text from OpenAI-compatible response chunks.

    Some backends, notably standalone vLLM completions in streaming mode, do not
    include a final usage object. Counting the generated text locally keeps
    token/cost diagnostics tied to the actual guarded request instead of falling
    back to the raw trace budget.
    """
    parts: List[str] = []
    for choice in obj.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        text = choice.get("text")
        if isinstance(text, str):
            parts.append(text)
        delta = choice.get("delta")
        if isinstance(delta, dict):
            content = delta.get("content")
            if isinstance(content, str):
                parts.append(content)
            delta_text = delta.get("text")
            if isinstance(delta_text, str):
                parts.append(delta_text)
        message = choice.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                parts.append(content)
    text = obj.get("text")
    if isinstance(text, str):
        parts.append(text)
    token = obj.get("token")
    if isinstance(token, dict):
        token_text = token.get("text")
        if isinstance(token_text, str):
            parts.append(token_text)
    generated_text = obj.get("generated_text")
    if isinstance(generated_text, str):
        parts.append(generated_text)
    elif isinstance(generated_text, list):
        parts.extend(str(item) for item in generated_text if isinstance(item, str))
    return "".join(parts)


def _apply_response_payload(
    obj: Any,
    *,
    generated_text_parts: List[str],
    server_metrics: Dict[str, Any],
) -> tuple[Optional[str], Dict[str, Any], Optional[str]]:
    """Extract text, usage, and metrics from one OpenAI-compatible payload."""
    if not isinstance(obj, dict):
        return None, {}, None
    fragment = _extract_generated_text_fragment(obj)
    if fragment:
        generated_text_parts.append(fragment)
    error = str(obj.get("error")) if obj.get("error") else None
    usage = obj["usage"] if isinstance(obj.get("usage"), dict) else {}
    metrics_source = None
    meta_info = obj.get("meta_info")
    if isinstance(meta_info, dict):
        server_metrics.update(_sglang_meta_to_metrics(meta_info))
        metrics_source = str(
            server_metrics.get("metrics_source") or "sglang_generate_meta_info"
        )
    metrics = obj.get("metrics")
    if isinstance(metrics, dict):
        server_metrics.clear()
        server_metrics.update(metrics)
        metrics_source = str(
            server_metrics.get("source") or "serverlessllm_api_metrics"
        )
    return error, usage, metrics_source


def _apply_faaslora_style_prompt_guard(
    *,
    prompt: str,
    requested_output_tokens: int,
    tokenizer_model: Optional[str],
    max_model_len: int,
    max_input_len: int,
    max_output_tokens_cap: int,
    tokenizer_add_special_tokens: bool = False,
) -> tuple[str, int, Optional[int]]:
    desired_tokens = max(1, int(requested_output_tokens or 1))
    if max_output_tokens_cap > 0:
        desired_tokens = min(desired_tokens, max_output_tokens_cap)
    safe_max_model_len = max(32, int(max_model_len or 0))
    reserve = max(32, min(desired_tokens, 256))
    prompt_budget = max(32, safe_max_model_len - reserve - 8)
    if max_input_len > 0:
        prompt_budget = min(prompt_budget, max_input_len)
    if not tokenizer_model:
        return prompt, max(1, min(desired_tokens, safe_max_model_len - 8)), None
    try:
        tokenizer = _get_prompt_guard_tokenizer(tokenizer_model)
        token_ids = tokenizer.encode(
            prompt,
            add_special_tokens=tokenizer_add_special_tokens,
        )
        if len(token_ids) > int(prompt_budget):
            token_ids = token_ids[-int(prompt_budget):]
        # Tokenizer decode -> re-encode is not perfectly idempotent near a hard
        # boundary. Re-check the final prompt string against the same tokenizer
        # so the emitted request never exceeds the backend's true input budget.
        while True:
            prompt = tokenizer.decode(token_ids, skip_special_tokens=False)
            reencoded = tokenizer.encode(
                prompt,
                add_special_tokens=tokenizer_add_special_tokens,
            )
            if len(reencoded) <= int(prompt_budget):
                token_ids = reencoded
                break
            overflow = max(1, len(reencoded) - int(prompt_budget))
            if len(token_ids) <= overflow:
                token_ids = token_ids[-1:]
                prompt = tokenizer.decode(token_ids, skip_special_tokens=False)
                reencoded = tokenizer.encode(
                    prompt,
                    add_special_tokens=tokenizer_add_special_tokens,
                )
                token_ids = reencoded[: max(1, min(len(reencoded), int(prompt_budget)))]
                break
            token_ids = token_ids[overflow:]
        actual_input_tokens = max(1, len(token_ids))
        safe_max_tokens = min(
            desired_tokens,
            max(1, safe_max_model_len - actual_input_tokens - 8),
        )
        return prompt, safe_max_tokens, actual_input_tokens
    except Exception:
        max_chars = min(safe_max_model_len * 4, 8192)
        if len(prompt) > max_chars:
            prompt = prompt[-max_chars:]
        return prompt, max(1, min(desired_tokens, safe_max_model_len - 8)), None


def _build_sglang_native_generate_body(
    *,
    prompt: str,
    body: Dict[str, Any],
    tokenizer_model: Optional[str],
    max_tokens: int,
) -> tuple[Dict[str, Any], int]:
    if not tokenizer_model:
        raise RuntimeError("sglang native generate requires --prompt-guard-tokenizer-model")
    tokenizer = _get_prompt_guard_tokenizer(tokenizer_model)
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    sampling_params: Dict[str, Any] = {
        "temperature": float(body.get("temperature", 0.7) or 0.7),
        "top_p": float(body.get("top_p", 0.9) or 0.9),
        "max_new_tokens": int(max(1, max_tokens)),
    }
    if body.get("sampling_seed") is not None:
        sampling_params["sampling_seed"] = int(body["sampling_seed"])
    elif body.get("seed") is not None:
        # This SGLang version names the field sampling_seed, while OpenAI/vLLM
        # compatible endpoints commonly use seed.
        sampling_params["sampling_seed"] = int(body["seed"])
    for source_key in (
        "top_k",
        "min_p",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        "stop",
        "stop_token_ids",
        "ignore_eos",
        "skip_special_tokens",
    ):
        value = body.get(source_key)
        if value is not None:
            sampling_params[source_key] = value
    native_body: Dict[str, Any] = {
        "input_ids": input_ids,
        "sampling_params": sampling_params,
        "stream": bool(body.get("stream", True)),
    }
    return native_body, len(input_ids)


def _build_slora_native_generate_body(
    *,
    prompt: str,
    body: Dict[str, Any],
    max_tokens: int,
    request_id: str,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "do_sample": False,
        "ignore_eos": bool(body.get("ignore_eos", True)),
        "max_new_tokens": int(max(1, max_tokens)),
    }
    for source_key in (
        "presence_penalty",
        "frequency_penalty",
        "temperature",
        "top_p",
        "top_k",
        "stop_sequences",
    ):
        value = body.get(source_key)
        if value is not None:
            params[source_key] = value
    return {
        "inputs": prompt,
        "parameters": params,
        "req_id": request_id,
    }


def _derive_latency_ms(start_ts: Any, end_ts: Any) -> Optional[float]:
    try:
        start = float(start_ts)
        end = float(end_ts)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(start) or not math.isfinite(end):
        return None
    if end < start:
        return None
    return (end - start) * 1000.0


def _latest_finite_ts(*values: Any) -> Optional[float]:
    latest: Optional[float] = None
    for value in values:
        try:
            ts = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(ts):
            continue
        if latest is None or ts > latest:
            latest = ts
    return latest


def _sglang_meta_to_metrics(meta_info: Dict[str, Any]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "metrics_source": "sglang_generate_meta_info",
        "meta_info": meta_info,
    }
    prompt_tokens = meta_info.get("prompt_tokens")
    completion_tokens = meta_info.get("completion_tokens")
    if prompt_tokens is not None:
        metrics["prompt_tokens"] = int(prompt_tokens)
    if completion_tokens is not None:
        metrics["completion_tokens"] = int(completion_tokens)

    request_received_ts = meta_info.get("request_received_ts")
    response_sent_to_client_ts = meta_info.get("response_sent_to_client_ts")
    request_finished_ts = meta_info.get("request_finished_ts")
    decode_throughput = meta_info.get("decode_throughput")

    finished_ts = _latest_finite_ts(request_finished_ts, response_sent_to_client_ts)
    ttft_ms = _derive_latency_ms(request_received_ts, response_sent_to_client_ts)
    e2e_ms = _derive_latency_ms(request_received_ts, finished_ts)
    if ttft_ms is not None:
        metrics["ttft_ms"] = ttft_ms
    if e2e_ms is not None:
        metrics["e2e_ms"] = e2e_ms
    completion_token_count: Optional[int] = None
    if completion_tokens is not None:
        completion_token_count = int(completion_tokens)
    if decode_throughput is not None and completion_token_count is not None and completion_token_count > 1:
        try:
            decode_tps = float(decode_throughput)
            if math.isfinite(decode_tps) and decode_tps > 0.0:
                metrics["tpot_ms"] = 1000.0 / decode_tps
                metrics["tpot_observed"] = True
        except (TypeError, ValueError):
            pass
    return metrics


def _replay_one(
    *,
    base_url: str,
    item: Dict[str, Any],
    request_index: int,
    timeout_s: float,
    start_time: float,
    base_cost_usd: float,
    input_token_cost_usd: float,
    output_token_cost_usd: float,
    require_server_metrics: bool,
    model_override: Optional[str],
    adapter_source_field: Optional[str],
    adapter_target_field: Optional[str],
    adapter_value_map: Dict[str, str],
    drop_body_fields: List[str],
    endpoint_path: str,
    convert_chat_to_prompt: bool,
    prompt_guard_tokenizer_model: Optional[str],
    prompt_guard_max_model_len: int,
    prompt_guard_max_input_len: int,
    prompt_guard_max_output_tokens_cap: int,
    sglang_native_generate: bool,
    slora_native_generate: bool,
    generation_seed: Optional[int],
    empty_success_retries: int,
    empty_success_retry_delay_s: float,
    min_output_tokens: int,
    include_stream_usage: bool,
) -> Dict[str, Any]:
    body = dict(item["body"])
    request_seed = _derive_request_generation_seed(
        generation_seed,
        item.get("request_id"),
        request_index,
    )
    if request_seed is not None:
        body["seed"] = int(request_seed)
    local_prompt_tokens_override: Optional[int] = None
    guard_max_tokens: Optional[int] = None
    for field in drop_body_fields:
        body.pop(field, None)
    if convert_chat_to_prompt:
        messages = body.pop("messages", None)
        if messages:
            prompt = _render_chat_messages_prompt(
                list(messages),
                tokenizer_model=prompt_guard_tokenizer_model,
            )
            requested_output_tokens = int(
                body.get("max_tokens")
                or body.get("max_completion_tokens")
                or item.get("expected_output_tokens")
                or 1
            )
            prompt, safe_max_tokens, guarded_prompt_tokens = _apply_faaslora_style_prompt_guard(
                prompt=prompt,
                requested_output_tokens=requested_output_tokens,
                tokenizer_model=prompt_guard_tokenizer_model,
                max_model_len=prompt_guard_max_model_len,
                max_input_len=prompt_guard_max_input_len,
                max_output_tokens_cap=prompt_guard_max_output_tokens_cap,
                tokenizer_add_special_tokens=bool(slora_native_generate),
            )
            guard_max_tokens = int(safe_max_tokens)
            if guarded_prompt_tokens is not None:
                local_prompt_tokens_override = int(guarded_prompt_tokens)
            if sglang_native_generate:
                body, local_prompt_tokens_override = _build_sglang_native_generate_body(
                    prompt=prompt,
                    body=body,
                    tokenizer_model=prompt_guard_tokenizer_model,
                    max_tokens=safe_max_tokens,
                )
            elif slora_native_generate:
                body = _build_slora_native_generate_body(
                    prompt=prompt,
                    body=body,
                    max_tokens=safe_max_tokens,
                    request_id=str(item.get("request_id") or f"req_{request_index:05d}"),
                )
            else:
                body["prompt"] = prompt
                if "max_tokens" in body or "max_completion_tokens" not in body:
                    body["max_tokens"] = safe_max_tokens
                if "max_completion_tokens" in body:
                    body["max_completion_tokens"] = safe_max_tokens
    if min_output_tokens > 0 and not (sglang_native_generate or slora_native_generate):
        request_max_tokens = (
            body.get("max_tokens")
            or body.get("max_completion_tokens")
            or guard_max_tokens
            or item.get("expected_output_tokens")
            or min_output_tokens
        )
        try:
            min_tokens = min(
                max(1, int(min_output_tokens)),
                max(1, int(request_max_tokens)),
            )
        except (TypeError, ValueError):
            min_tokens = max(1, int(min_output_tokens))
        body["min_tokens"] = max(int(body.get("min_tokens", 0) or 0), min_tokens)
    if include_stream_usage and bool(body.get("stream", False)):
        stream_options = body.get("stream_options")
        if not isinstance(stream_options, dict):
            stream_options = {}
        stream_options["include_usage"] = True
        body["stream_options"] = stream_options
    if model_override:
        body["model"] = model_override
    if adapter_source_field and adapter_target_field:
        adapter_value = body.get(adapter_source_field)
        if adapter_value is None:
            adapter_value = item.get(adapter_source_field)
        if adapter_value is None and adapter_source_field == "adapter_id":
            adapter_value = item.get("adapter_id")
        if adapter_value is not None:
            body[adapter_target_field] = adapter_value_map.get(str(adapter_value), adapter_value)
    endpoint = f"{base_url.rstrip('/')}/{endpoint_path.lstrip('/')}"
    scheduled_offset_s = float(item["arrival_time_s"])
    dispatch_offset_s = time.perf_counter() - start_time
    t0 = time.perf_counter()
    api_ttft_ms: Optional[float] = None
    status_code: Optional[int] = None
    usage: Dict[str, Any] = {}
    error: Optional[str] = None
    generated_text_parts: List[str] = []
    stream_event_count = 0
    server_metrics: Dict[str, Any] = {}
    metrics_source: Optional[str] = None
    request_attempts = 0
    empty_success_retry_count = 0
    final_empty_success = False
    last_raw_text_chars = 0
    last_raw_text_preview = ""

    max_attempts = max(1, 1 + max(0, int(empty_success_retries or 0)))
    for attempt_idx in range(max_attempts):
        request_attempts = attempt_idx + 1
        api_ttft_ms = None
        status_code = None
        usage = {}
        error = None
        generated_text_parts = []
        stream_event_count = 0
        server_metrics = {}
        metrics_source = None
        raw_text = ""
        try:
            with requests.post(endpoint, json=body, stream=True, timeout=timeout_s) as resp:
                status_code = resp.status_code
                raw_chunks: List[str] = []
                for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
                    if not chunk:
                        continue
                    # TTFT should start at the first non-empty response payload,
                    # not at an empty keepalive/whitespace chunk.
                    if api_ttft_ms is None and str(chunk).strip():
                        api_ttft_ms = (time.perf_counter() - t0) * 1000.0
                    raw_chunks.append(chunk)
                raw_text = "".join(raw_chunks).strip()
                last_raw_text_chars = len(raw_text)
                last_raw_text_preview = raw_text[:240]

                if status_code != 200:
                    error = raw_text[:1000] or f"HTTP {status_code}"
                elif raw_text:
                    stripped_raw_text = raw_text.lstrip()
                    if stripped_raw_text.startswith(("{", "[")):
                        stream_event_count = 1
                        try:
                            obj = json.loads(raw_text)
                            payload_error, payload_usage, payload_metrics_source = (
                                _apply_response_payload(
                                    obj,
                                    generated_text_parts=generated_text_parts,
                                    server_metrics=server_metrics,
                                )
                            )
                            if payload_error:
                                error = payload_error
                            if payload_usage:
                                usage = payload_usage
                            if payload_metrics_source:
                                metrics_source = payload_metrics_source
                        except json.JSONDecodeError:
                            pass
                    else:
                        for line in raw_text.splitlines():
                            line = line.strip()
                            if not line.startswith("data:"):
                                continue
                            payload = line[len("data:") :].strip()
                            if payload == "[DONE]":
                                break
                            stream_event_count += 1
                            try:
                                obj = json.loads(payload)
                                payload_error, payload_usage, payload_metrics_source = (
                                    _apply_response_payload(
                                        obj,
                                        generated_text_parts=generated_text_parts,
                                        server_metrics=server_metrics,
                                    )
                                )
                                if payload_error:
                                    error = payload_error
                                if payload_usage:
                                    usage = payload_usage
                                if payload_metrics_source:
                                    metrics_source = payload_metrics_source
                            except json.JSONDecodeError:
                                continue
            api_e2e_ms = (time.perf_counter() - t0) * 1000.0
        except Exception as exc:  # noqa: BLE001
            api_e2e_ms = (time.perf_counter() - t0) * 1000.0
            error = str(exc)

        generated_text_candidate = "".join(generated_text_parts)
        observable_success_payload = bool(
            generated_text_candidate
            or usage
            or server_metrics
        )
        empty_success = (
            error is None
            and status_code == 200
            and not observable_success_payload
        )
        if empty_success and request_attempts < max_attempts:
            empty_success_retry_count += 1
            time.sleep(max(0.0, float(empty_success_retry_delay_s or 0.0)))
            continue
        if empty_success:
            final_empty_success = True
            error = (
                "empty successful response with no generated text, usage, or "
                "server metrics"
            )
        break

    completion_offset_s = time.perf_counter() - start_time
    generated_text = "".join(generated_text_parts)
    observed_empty_generation = (
        not generated_text
        and error is None
        and status_code == 200
        and stream_event_count > 0
    )
    local_completion_tokens_override = _count_text_tokens(
        generated_text,
        prompt_guard_tokenizer_model,
    ) if (generated_text or observed_empty_generation) else None
    prompt_token_source = "usage"
    completion_token_source = "usage"
    usage_prompt_tokens = (usage or {}).get("prompt_tokens")
    usage_completion_tokens = (usage or {}).get("completion_tokens")
    server_prompt_tokens = server_metrics.get("prompt_tokens")
    server_completion_tokens = server_metrics.get("completion_tokens")
    if usage_prompt_tokens is None:
        prompt_token_source = "server_metrics" if server_prompt_tokens is not None else (
            "local_guarded_prompt" if local_prompt_tokens_override is not None else "trace_expected"
        )
    if usage_completion_tokens is None:
        completion_token_source = "server_metrics" if server_completion_tokens is not None else (
            (
                "local_generated_text_empty"
                if observed_empty_generation
                else "local_generated_text"
            )
            if local_completion_tokens_override is not None else "trace_expected"
        )
    prompt_tokens = int(
        (usage or {}).get(
            "prompt_tokens",
            server_metrics.get(
                "prompt_tokens",
                local_prompt_tokens_override
                if local_prompt_tokens_override is not None
                else item.get("expected_input_tokens", 0),
            ),
        )
        or 0
    )
    completion_tokens = int(
        (usage or {}).get(
            "completion_tokens",
            server_metrics.get(
                "completion_tokens",
                local_completion_tokens_override
                if local_completion_tokens_override is not None
                else item.get("expected_output_tokens", 0),
            ),
        )
        or 0
    )
    total_tokens = int((usage or {}).get("total_tokens", prompt_tokens + completion_tokens) or 0)
    cost_usd = _calc_cost(
        prompt_tokens,
        completion_tokens,
        base=base_cost_usd,
        in_cost=input_token_cost_usd,
        out_cost=output_token_cost_usd,
    )
    backend_ttft_ms = server_metrics.get("ttft_ms")
    backend_e2e_ms = server_metrics.get("e2e_ms")
    service_ttft_ms = None
    service_e2e_ms = None
    tpot_ms = server_metrics.get("tpot_ms")
    tpot_observed = bool(server_metrics.get("tpot_observed", False))
    runtime_ttft_ms = server_metrics.get("runtime_ttft_ms")
    serverless_overhead_ms = server_metrics.get("serverless_overhead_ms")
    request_received_at = server_metrics.get("request_received_at")
    backend_started_at = server_metrics.get("backend_started_at")
    first_token_at = server_metrics.get("first_token_at")
    last_token_at = server_metrics.get("last_token_at")
    finished_at = server_metrics.get("finished_at")
    lora_load_ms = server_metrics.get("lora_load_ms")
    cache_hit = (
        bool(server_metrics.get("cache_hit"))
        if server_metrics.get("cache_hit") is not None
        else None
    )
    gpu_ready_request = (
        bool(server_metrics.get("gpu_ready_request"))
        if server_metrics.get("gpu_ready_request") is not None
        else None
    )
    scaleup_affected = (
        bool(server_metrics.get("scaleup_affected"))
        if server_metrics.get("scaleup_affected") is not None
        else None
    )
    scaleup_first_service = (
        bool(server_metrics.get("scaleup_first_service"))
        if server_metrics.get("scaleup_first_service") is not None
        else None
    )
    cold_start_latency_ms = server_metrics.get("cold_start_latency_ms")
    comparable_request = None
    warm_standard_request = None
    if cache_hit is not None and scaleup_affected is not None:
        comparable_request = bool(cache_hit) and not bool(scaleup_affected)
        warm_standard_request = comparable_request

    server_queue_wait_ms = _derive_latency_ms(request_received_at, backend_started_at)
    service_start_at = backend_started_at if server_queue_wait_ms is not None else request_received_at
    server_ttft_available = (
        backend_ttft_ms is not None
        or _derive_latency_ms(service_start_at, first_token_at) is not None
        or _derive_latency_ms(request_received_at, first_token_at) is not None
    )
    server_e2e_available = (
        backend_e2e_ms is not None
        or _derive_latency_ms(service_start_at, finished_at) is not None
        or _derive_latency_ms(service_start_at, last_token_at) is not None
        or _derive_latency_ms(request_received_at, finished_at) is not None
        or _derive_latency_ms(request_received_at, last_token_at) is not None
    )
    server_service_ttft_ms = _derive_latency_ms(service_start_at, first_token_at)
    if server_service_ttft_ms is None:
        server_service_ttft_ms = _derive_latency_ms(request_received_at, first_token_at)
    if server_service_ttft_ms is None:
        server_service_ttft_ms = backend_ttft_ms
    server_service_e2e_ms = _derive_latency_ms(service_start_at, finished_at)
    if server_service_e2e_ms is None:
        server_service_e2e_ms = _derive_latency_ms(service_start_at, last_token_at)
    if server_service_e2e_ms is None:
        server_service_e2e_ms = _derive_latency_ms(request_received_at, finished_at)
    if server_service_e2e_ms is None:
        server_service_e2e_ms = _derive_latency_ms(request_received_at, last_token_at)
    if server_service_e2e_ms is None:
        server_service_e2e_ms = backend_e2e_ms
    if server_service_ttft_ms is not None and server_service_e2e_ms is not None:
        server_service_e2e_ms = max(
            float(server_service_e2e_ms),
            float(server_service_ttft_ms),
        )

    # Prefer server-side first-token timestamps when available. Some baselines
    # generate tokens incrementally but return a single non-streamed JSON object,
    # so client chunk timing collapses TTFT onto full response latency.
    #
    # When the backend exposes backend_started_at, the interval before it is
    # dispatch/admission wait rather than service-path TTFT. This keeps the
    # baseline split aligned with FaaSLoRA: waiting for an executable runtime is
    # queue/admission, while runtime prefill/decode after admission is service.
    if server_service_ttft_ms is not None:
        service_ttft_ms = float(server_service_ttft_ms)
    elif api_ttft_ms is not None:
        service_ttft_ms = float(api_ttft_ms)
    if server_service_e2e_ms is not None:
        service_e2e_ms = float(server_service_e2e_ms)
    elif api_e2e_ms is not None:
        service_e2e_ms = float(api_e2e_ms)
    replay_dispatch_wait_ms = max(0.0, (dispatch_offset_s - scheduled_offset_s) * 1000.0)
    dispatch_admission_wait_ms = replay_dispatch_wait_ms + max(0.0, float(server_queue_wait_ms or 0.0))
    overall_ttft_ms = None
    if service_ttft_ms is not None:
        overall_ttft_ms = dispatch_admission_wait_ms + float(service_ttft_ms)
    elif api_ttft_ms is not None:
        overall_ttft_ms = replay_dispatch_wait_ms + float(api_ttft_ms)
    if service_e2e_ms is not None:
        overall_e2e_ms = dispatch_admission_wait_ms + float(service_e2e_ms)
    else:
        overall_e2e_ms = max(0.0, (completion_offset_s - scheduled_offset_s) * 1000.0)
    if overall_ttft_ms is not None and overall_e2e_ms is not None:
        overall_e2e_ms = max(float(overall_e2e_ms), float(overall_ttft_ms))
    if completion_tokens > 1:
        decode_window_ms = _derive_latency_ms(first_token_at, last_token_at)
        if decode_window_ms is not None:
            tpot_ms = decode_window_ms / max(1, completion_tokens - 1)
            tpot_observed = True
        elif tpot_ms is None and api_ttft_ms is not None and api_e2e_ms is not None:
            decode_window_ms = max(0.0, float(api_e2e_ms) - float(api_ttft_ms))
            tpot_ms = decode_window_ms / max(1, completion_tokens - 1)
            # A single non-streamed response cannot expose the decode window.
            tpot_observed = stream_event_count > 1 or tpot_observed
    service_overhead_ms = serverless_overhead_ms
    if service_overhead_ms is None and service_ttft_ms is not None and runtime_ttft_ms is not None:
        service_overhead_ms = max(0.0, float(service_ttft_ms) - float(runtime_ttft_ms))
    if service_overhead_ms is not None:
        serverless_overhead_ms = dispatch_admission_wait_ms + float(service_overhead_ms)

    metric_warnings: List[str] = []
    if error is None and status_code == 200:
        if prompt_token_source == "trace_expected" or completion_token_source == "trace_expected":
            metric_warnings.append(
                "token usage metrics unavailable; at least one token count fell back to trace expected tokens"
            )
        elif (
            prompt_token_source == "local_guarded_prompt"
            or completion_token_source == "local_generated_text"
            or completion_token_source == "local_generated_text_empty"
        ):
            metric_warnings.append(
                "server usage metrics unavailable; using local tokenizer counts from guarded prompt/response text"
            )
    if error is None and status_code == 200 and require_server_metrics:
        if service_ttft_ms is None or service_e2e_ms is None:
            error = "missing required client-observed service metrics: api_ttft_ms/api_e2e_ms"
        else:
            if not server_ttft_available or not server_e2e_available:
                metric_warnings.append(
                    "server latency metrics unavailable; using client-observed api metrics"
                )
            if completion_tokens > 1 and (tpot_ms is None or not tpot_observed):
                metric_warnings.append("observable decode-window tpot_ms unavailable for this request")
    metric_warning = "; ".join(metric_warnings) if metric_warnings else None

    return {
        "request_id": item["request_id"],
        "generation_seed": request_seed,
        "arrival_time_s": scheduled_offset_s,
        "dispatch_offset_s": dispatch_offset_s,
        "completion_offset_s": completion_offset_s,
        "adapter_id": item.get("adapter_id"),
        "ttft_ms": overall_ttft_ms,
        "e2e_ms": overall_e2e_ms,
        "overall_ttft_ms": overall_ttft_ms,
        "overall_e2e_ms": overall_e2e_ms,
        "service_ttft_ms": service_ttft_ms,
        "service_e2e_ms": service_e2e_ms,
        "backend_ttft_ms": backend_ttft_ms,
        "backend_e2e_ms": backend_e2e_ms,
        "dispatch_admission_wait_ms": dispatch_admission_wait_ms,
        "replay_dispatch_wait_ms": replay_dispatch_wait_ms,
        "server_queue_wait_ms": server_queue_wait_ms,
        "tpot_ms": tpot_ms,
        "tpot_observed": tpot_observed,
        "runtime_ttft_ms": runtime_ttft_ms,
        "serverless_overhead_ms": serverless_overhead_ms,
        "service_overhead_ms": service_overhead_ms,
        "lora_load_ms": lora_load_ms,
        "cache_hit": cache_hit,
        "gpu_ready_request": gpu_ready_request,
        "scaleup_affected": scaleup_affected,
        "scaleup_first_service": scaleup_first_service,
        "cold_start_latency_ms": cold_start_latency_ms,
        "comparable_request": comparable_request,
        "warm_standard_request": warm_standard_request,
        "metrics_source": metrics_source,
        "status_code": status_code,
        "success": error is None and status_code == 200,
        "usage": usage,
        "stream_event_count": stream_event_count,
        "api_ttft_ms": api_ttft_ms,
        "api_e2e_ms": api_e2e_ms,
        "server_metrics": server_metrics,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "prompt_token_source": prompt_token_source,
        "completion_token_source": completion_token_source,
        "guard_prompt_tokens": local_prompt_tokens_override,
        "guard_max_tokens": guard_max_tokens,
        "request_attempts": request_attempts,
        "empty_success_retries": empty_success_retry_count,
        "final_empty_success": final_empty_success,
        "raw_text_chars": last_raw_text_chars,
        "raw_text_preview": last_raw_text_preview,
        "generated_text_chars": len(generated_text),
        "cost_usd": cost_usd,
        "metric_warning": metric_warning,
        "error": error,
    }


def _build_live_stats(results: List[Optional[Dict[str, Any]]], ttft_slo_ms: float) -> Dict[str, Any]:
    done = [r for r in results if r is not None]
    ok = [r for r in done if bool(r.get("success"))]
    if not done:
        return {
            "done": 0,
            "ok": 0,
            "failed": 0,
            "avg_ttft_ms": 0.0,
            "p95_ttft_ms": 0.0,
            "p99_ttft_ms": 0.0,
            "avg_tpot_ms": None,
            "avg_e2e_ms": 0.0,
            "p95_e2e_ms": 0.0,
            "p99_e2e_ms": 0.0,
            "done_rps": 0.0,
            "throughput_tok_per_s": 0.0,
            "slo_goodput_rps": 0.0,
            "slo_goodput_tok_per_s": 0.0,
            "avg_cost_usd": 0.0,
            "token_proxy_avg_cost_usd": 0.0,
            "token_proxy_total_cost_usd": 0.0,
            "token_proxy_cost_per_1m_total_tokens_usd": 0.0,
            "ce": 0.0,
            "token_proxy_ce": 0.0,
            "qpr_legacy": 0.0,
            "slo_attainment": 0.0,
        }

    ttft = [float(r["ttft_ms"]) for r in ok if r.get("ttft_ms") is not None]
    service_ttft = [
        float(r["service_ttft_ms"])
        for r in ok
        if r.get("service_ttft_ms") is not None
    ]
    dispatch_wait = [
        float(r["dispatch_admission_wait_ms"])
        for r in ok
        if r.get("dispatch_admission_wait_ms") is not None
    ]
    tpot = [
        float(r["tpot_ms"])
        for r in ok
        if r.get("tpot_ms") is not None and bool(r.get("tpot_observed", True))
    ]
    e2e = [float(r["e2e_ms"]) for r in ok]
    service_e2e = [
        float(r["service_e2e_ms"])
        for r in ok
        if r.get("service_e2e_ms") is not None
    ]
    cost = [float(r["cost_usd"]) for r in ok]
    prompt_tokens = [int(r.get("prompt_tokens", 0) or 0) for r in ok]
    out_tokens = [int(r.get("completion_tokens", 0) or 0) for r in ok]
    runtime_ttft = [
        float(r["runtime_ttft_ms"])
        for r in ok
        if r.get("runtime_ttft_ms") is not None
    ]
    overhead = [
        float(r["serverless_overhead_ms"])
        for r in ok
        if r.get("serverless_overhead_ms") is not None
    ]
    lora_io = [
        float(r["lora_load_ms"])
        for r in ok
        if r.get("lora_load_ms") is not None
    ]
    gpu_ready_ttft = [
        float(r["ttft_ms"])
        for r in ok
        if bool(r.get("gpu_ready_request"))
    ]
    comparable_ttft = [
        float(r["ttft_ms"])
        for r in ok
        if bool(r.get("comparable_request"))
    ]
    warm_standard_ttft = [
        float(r["ttft_ms"])
        for r in ok
        if bool(r.get("warm_standard_request"))
    ]
    scaleup_ttft = [
        float(r["ttft_ms"])
        for r in ok
        if bool(r.get("scaleup_affected"))
    ]
    cold_start = [
        float(r["cold_start_latency_ms"])
        for r in ok
        if r.get("cold_start_latency_ms") is not None
    ]
    elapsed = max(float(r.get("completion_offset_s", 0.0) or 0.0) for r in done) if done else 0.0
    avg_e2e_ms = sum(e2e) / len(e2e) if e2e else 0.0
    total_cost_usd = sum(cost)
    avg_cost_usd = sum(cost) / len(cost) if cost else 0.0
    total_input_tokens = sum(prompt_tokens)
    total_output_tokens = sum(out_tokens)
    total_tokens = total_input_tokens + total_output_tokens
    denom = avg_cost_usd * (avg_e2e_ms / 1000.0)
    ce = 1.0 / denom if denom > 1e-12 else 0.0
    done_rps = (len(done) / max(elapsed, 1e-6)) if elapsed > 0 else 0.0
    legacy_denom = avg_cost_usd * ((sum(ttft) / len(ttft)) / 1000.0) if ttft and avg_cost_usd > 0 else 0.0
    qpr_legacy = ((sum(out_tokens) / max(elapsed, 1e-6)) / legacy_denom) if legacy_denom > 1e-12 else 0.0
    slo_attainment = (
        sum(1 for r in ok if float(r.get("ttft_ms") or 0.0) <= ttft_slo_ms) / len(ok)
    ) if ok else 0.0

    return {
        "done": len(done),
        "ok": len(ok),
        "failed": len(done) - len(ok),
        "avg_ttft_ms": sum(ttft) / len(ttft) if ttft else 0.0,
        "p95_ttft_ms": _percentile(ttft, 95) if ttft else 0.0,
        "p99_ttft_ms": _percentile(ttft, 99) if ttft else 0.0,
        "avg_service_ttft_ms": sum(service_ttft) / len(service_ttft) if service_ttft else 0.0,
        "p95_service_ttft_ms": _percentile(service_ttft, 95) if service_ttft else 0.0,
        "p99_service_ttft_ms": _percentile(service_ttft, 99) if service_ttft else 0.0,
        "avg_dispatch_admission_wait_ms": (
            sum(dispatch_wait) / len(dispatch_wait) if dispatch_wait else 0.0
        ),
        "p95_dispatch_admission_wait_ms": _percentile(dispatch_wait, 95) if dispatch_wait else 0.0,
        "p99_dispatch_admission_wait_ms": _percentile(dispatch_wait, 99) if dispatch_wait else 0.0,
        "avg_tpot_ms": (sum(tpot) / len(tpot)) if tpot else None,
        "avg_e2e_ms": avg_e2e_ms,
        "p95_e2e_ms": _percentile(e2e, 95) if e2e else 0.0,
        "p99_e2e_ms": _percentile(e2e, 99) if e2e else 0.0,
        "avg_service_e2e_ms": sum(service_e2e) / len(service_e2e) if service_e2e else 0.0,
        "p95_service_e2e_ms": _percentile(service_e2e, 95) if service_e2e else 0.0,
        "p99_service_e2e_ms": _percentile(service_e2e, 99) if service_e2e else 0.0,
        "done_rps": done_rps,
        "throughput_tok_per_s": (sum(out_tokens) / max(elapsed, 1e-6)) if elapsed > 0 else 0.0,
        "slo_goodput_rps": done_rps * slo_attainment,
        "slo_goodput_tok_per_s": ((sum(out_tokens) / max(elapsed, 1e-6)) * slo_attainment) if elapsed > 0 else 0.0,
        "avg_cost_usd": avg_cost_usd,
        "token_proxy_avg_cost_usd": avg_cost_usd,
        "token_proxy_total_cost_usd": total_cost_usd,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "token_proxy_cost_per_1m_total_tokens_usd": _cost_per_million_tokens(total_cost_usd, total_tokens),
        "token_proxy_cost_per_1m_output_tokens_usd": _cost_per_million_tokens(total_cost_usd, total_output_tokens),
        "ce": ce,
        "token_proxy_ce": ce,
        "qpr_legacy": qpr_legacy,
        "slo_attainment": slo_attainment,
        "avg_runtime_ttft_ms": (sum(runtime_ttft) / len(runtime_ttft)) if runtime_ttft else None,
        "avg_serverless_overhead_ms": (sum(overhead) / len(overhead)) if overhead else None,
        "avg_lora_io_ms": (sum(lora_io) / len(lora_io)) if lora_io else None,
        "avg_gpu_ready_ttft_ms": (sum(gpu_ready_ttft) / len(gpu_ready_ttft)) if gpu_ready_ttft else None,
        "avg_comparable_ttft_ms": (sum(comparable_ttft) / len(comparable_ttft)) if comparable_ttft else None,
        "p95_comparable_ttft_ms": _percentile(comparable_ttft, 95) if comparable_ttft else None,
        "p99_comparable_ttft_ms": _percentile(comparable_ttft, 99) if comparable_ttft else None,
        "avg_warm_standard_ttft_ms": (sum(warm_standard_ttft) / len(warm_standard_ttft)) if warm_standard_ttft else None,
        "p95_warm_standard_ttft_ms": _percentile(warm_standard_ttft, 95) if warm_standard_ttft else None,
        "p99_warm_standard_ttft_ms": _percentile(warm_standard_ttft, 99) if warm_standard_ttft else None,
        "avg_scaleup_affected_ttft_ms": (sum(scaleup_ttft) / len(scaleup_ttft)) if scaleup_ttft else None,
        "avg_cold_start_latency_ms": (sum(cold_start) / len(cold_start)) if cold_start else None,
        "p95_cold_start_latency_ms": _percentile(cold_start, 95) if cold_start else None,
        "cache_hit_rate": _known_bool_rate(ok, "cache_hit"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay a shared trace against an OpenAI-compatible endpoint.")
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument(
        "--base-url-list",
        default=None,
        help="Optional comma-separated endpoint list for data-parallel replicas. "
        "Requests are dispatched round-robin by trace index.",
    )
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--sleep-scale", type=float, default=1.0)
    ap.add_argument("--timeout-s", type=float, default=600.0)
    ap.add_argument("--base-cost-usd", type=float, default=0.001)
    ap.add_argument("--input-token-cost-usd", type=float, default=0.0000015)
    ap.add_argument("--output-token-cost-usd", type=float, default=0.000002)
    ap.add_argument("--ttft-slo-ms", type=float, default=5000.0)
    ap.add_argument("--label", default="serverlessllm")
    ap.add_argument("--require-server-metrics", action="store_true")
    ap.add_argument("--model-override", default=None)
    ap.add_argument("--adapter-source-field", default=None)
    ap.add_argument("--adapter-target-field", default=None)
    ap.add_argument("--adapter-value-map", type=Path, default=None)
    ap.add_argument("--drop-body-field", action="append", default=[])
    ap.add_argument("--endpoint-path", default="/v1/chat/completions")
    ap.add_argument("--convert-chat-to-prompt", action="store_true")
    ap.add_argument("--prompt-guard-tokenizer-model", default=None)
    ap.add_argument("--prompt-guard-max-model-len", type=int, default=0)
    ap.add_argument("--prompt-guard-max-input-len", type=int, default=0)
    ap.add_argument("--prompt-guard-max-output-tokens-cap", type=int, default=0)
    ap.add_argument("--sglang-native-generate", action="store_true")
    ap.add_argument("--slora-native-generate", action="store_true")
    ap.add_argument(
        "--empty-success-retries",
        type=int,
        default=0,
        help=(
            "Retry a transient HTTP 200 response that contains no body, no stream "
            "events, no usage, and no server metrics. The retry remains part of "
            "the same request latency window."
        ),
    )
    ap.add_argument("--empty-success-retry-delay-s", type=float, default=1.0)
    ap.add_argument(
        "--min-output-tokens",
        type=int,
        default=0,
        help=(
            "For OpenAI-compatible generation endpoints, request at least this "
            "many generated tokens. This keeps positive-output trace requests "
            "from ending at EOS before the first token, so TTFT remains "
            "observable. Native SGLang/S-LoRA endpoint adapters are not changed."
        ),
    )
    ap.add_argument(
        "--include-stream-usage",
        action="store_true",
        help=(
            "When a request is streamed, add stream_options.include_usage=true "
            "so OpenAI-compatible servers can report observed token usage."
        ),
    )
    ap.add_argument(
        "--dynamic-lora-modules",
        type=Path,
        default=None,
        help=(
            "Optional name=path LoRA module file. When set, the replay loads the "
            "request adapter into the target vLLM endpoint through "
            "/v1/load_lora_adapter before sending the request. The load latency "
            "naturally appears as upstream dispatch/admission wait."
        ),
    )
    ap.add_argument(
        "--dynamic-lora-timeout-s",
        type=float,
        default=180.0,
        help="Timeout for vLLM runtime LoRA load requests.",
    )
    ap.add_argument(
        "--dynamic-lora-load-path",
        default="/v1/load_lora_adapter",
        help="Runtime LoRA load endpoint path. vLLM uses /v1/load_lora_adapter; SGLang uses /load_lora_adapter.",
    )
    ap.add_argument(
        "--dynamic-lora-unload-path",
        default="/v1/unload_lora_adapter",
        help="Runtime LoRA unload endpoint path. vLLM uses /v1/unload_lora_adapter; SGLang uses /unload_lora_adapter.",
    )
    ap.add_argument(
        "--dynamic-lora-remote-endpoint",
        default="",
        help="Optional PrimeLoRA HTTP artifact endpoint used to materialize missing adapter paths before runtime loading.",
    )
    ap.add_argument(
        "--dynamic-lora-remote-timeout-s",
        type=float,
        default=300.0,
        help="Timeout for remote artifact downloads used by --dynamic-lora-remote-endpoint.",
    )
    ap.add_argument(
        "--dynamic-lora-remote-token-env",
        default="PRIME_REMOTE_TOKEN",
        help="Environment variable containing the optional bearer token for the remote artifact endpoint.",
    )
    ap.add_argument(
        "--dynamic-lora-remote-bandwidth-mbps",
        type=float,
        default=0.0,
        help=(
            "Optional MiB/s bandwidth used when --dynamic-lora-remote-endpoint "
            "is file://... for local remote simulation. The name keeps legacy "
            "compatibility with the PrimeLoRA bandwidth_mbps setting."
        ),
    )
    ap.add_argument(
        "--request-remote-adapter-map",
        type=Path,
        default=None,
        help=(
            "Optional adapter_id -> local target path map. When set together "
            "with --request-remote-endpoint, the replay materializes a missing "
            "adapter before sending the request without invoking a runtime "
            "LoRA load API. This is used for systems whose router passes a "
            "backend-visible LoRA path on each request."
        ),
    )
    ap.add_argument(
        "--request-remote-endpoint",
        default="",
        help="Remote artifact endpoint used with --request-remote-adapter-map.",
    )
    ap.add_argument(
        "--request-remote-timeout-s",
        type=float,
        default=300.0,
        help="Timeout for request-path remote adapter materialization.",
    )
    ap.add_argument(
        "--request-remote-token-env",
        default="PRIME_REMOTE_TOKEN",
        help="Environment variable containing the optional bearer token for request-path remote fetches.",
    )
    ap.add_argument(
        "--request-remote-bandwidth-mbps",
        type=float,
        default=0.0,
        help="Optional MiB/s bandwidth for file:// request-path remote simulation.",
    )
    ap.add_argument(
        "--dynamic-lora-routing",
        choices=("round_robin", "adapter_hash", "adapter_pair_hash", "adaptive_hot_pair_hash"),
        default="round_robin",
        help=(
            "Endpoint selection policy used when --dynamic-lora-modules is set. "
            "adaptive_hot_pair_hash keeps cold adapters on one deterministic "
            "replica and expands observed hot adapters to two replicas."
        ),
    )
    ap.add_argument(
        "--dynamic-lora-hot-pair-threshold",
        type=int,
        default=8,
        help=(
            "For adaptive_hot_pair_hash, promote an adapter to two endpoint "
            "choices after this many observed requests."
        ),
    )
    ap.add_argument(
        "--dynamic-lora-hot-pair-max-adapters",
        type=int,
        default=32,
        help=(
            "For adaptive_hot_pair_hash, maximum number of adapters promoted "
            "to two endpoint choices."
        ),
    )
    ap.add_argument(
        "--dynamic-lora-max-loaded-per-endpoint",
        type=int,
        default=0,
        help=(
            "Maximum runtime-registered LoRA adapters kept per vLLM endpoint. "
            "When positive, the replay unloads inactive least-recently-used "
            "adapters before loading a new one. This bounds vLLM OpenAI "
            "runtime-LoRA host memory without changing the adapter universe."
        ),
    )
    ap.add_argument(
        "--generation-seed",
        type=int,
        default=None,
        help="Optional base sampling seed; each request derives a stable per-request seed from request_id.",
    )
    ap.add_argument(
        "--max-requests",
        type=int,
        default=0,
        help="Replay only the first N requests. Intended for bounded preflight probes.",
    )
    ap.add_argument(
        "--abort-after-failures",
        type=int,
        default=0,
        help="Abort the replay process once this many failed requests have completed.",
    )
    ap.add_argument(
        "--abort-failures-min-done",
        type=int,
        default=1,
        help="Minimum completed requests before --abort-after-failures can terminate replay.",
    )
    args = ap.parse_args()

    payload = json.loads(args.trace.read_text(encoding="utf-8"))
    requests_list = list(payload.get("requests", []))
    if int(args.max_requests or 0) > 0:
        requests_list = requests_list[: int(args.max_requests)]
    if not requests_list:
        raise RuntimeError("trace contains no requests")
    if args.sglang_native_generate and args.slora_native_generate:
        raise RuntimeError("--sglang-native-generate and --slora-native-generate are mutually exclusive")
    base_urls = _parse_base_urls(args.base_url, args.base_url_list)
    adapter_value_map: Dict[str, str] = {}
    if args.adapter_value_map:
        raw_map = json.loads(args.adapter_value_map.read_text(encoding="utf-8"))
        if not isinstance(raw_map, dict):
            raise RuntimeError(f"adapter value map must be a JSON object: {args.adapter_value_map}")
        adapter_value_map = {str(key): str(value) for key, value in raw_map.items()}
    dynamic_lora_loader: Optional[DynamicLoRALoader] = None
    request_remote_fetcher: Optional[RemoteArtifactFetcher] = None
    request_remote_map: Dict[str, str] = {}
    if args.request_remote_adapter_map:
        if not str(args.request_remote_endpoint or "").strip():
            raise RuntimeError("--request-remote-adapter-map requires --request-remote-endpoint")
        request_remote_map = _load_adapter_path_map(args.request_remote_adapter_map)
        request_remote_fetcher = RemoteArtifactFetcher(
            endpoint=str(args.request_remote_endpoint),
            timeout_s=float(args.request_remote_timeout_s or 300.0),
            token_env=str(args.request_remote_token_env or "PRIME_REMOTE_TOKEN"),
            bandwidth_mbps=float(args.request_remote_bandwidth_mbps or 0.0),
        )
    if args.dynamic_lora_modules:
        module_map = _load_adapter_path_map(args.dynamic_lora_modules)
        if not module_map:
            raise RuntimeError(f"dynamic LoRA module map is empty: {args.dynamic_lora_modules}")
        remote_fetcher = None
        if str(args.dynamic_lora_remote_endpoint or "").strip():
            remote_fetcher = RemoteArtifactFetcher(
                endpoint=str(args.dynamic_lora_remote_endpoint),
                timeout_s=float(args.dynamic_lora_remote_timeout_s or 300.0),
                token_env=str(args.dynamic_lora_remote_token_env or "PRIME_REMOTE_TOKEN"),
                bandwidth_mbps=float(args.dynamic_lora_remote_bandwidth_mbps or 0.0),
            )
        dynamic_lora_loader = DynamicLoRALoader(
            modules=module_map,
            timeout_s=float(args.dynamic_lora_timeout_s or 180.0),
            max_loaded_per_endpoint=int(args.dynamic_lora_max_loaded_per_endpoint or 0),
            load_path=str(args.dynamic_lora_load_path),
            unload_path=str(args.dynamic_lora_unload_path),
            remote_fetcher=remote_fetcher,
        )

    results: List[Optional[Dict[str, Any]]] = [None] * len(requests_list)
    threads: List[threading.Thread] = []
    lock = threading.Lock()
    dynamic_routing_lock = threading.Lock()
    dynamic_adapter_counts: Dict[str, int] = {}
    dynamic_hot_pair_adapters: set[str] = set()
    start_time = time.perf_counter()
    last_live_print_at = 0.0
    arrival_schedule = [float(item["arrival_time_s"]) * max(args.sleep_scale, 0.0) for item in requests_list]

    def _worker(index: int, item: Dict[str, Any]) -> None:
        nonlocal last_live_print_at
        adapter_id_for_dynamic = item.get("adapter_id")
        if adapter_id_for_dynamic is None and args.adapter_source_field:
            adapter_id_for_dynamic = item.get(str(args.adapter_source_field))
        if dynamic_lora_loader is not None:
            dynamic_routing_policy = str(args.dynamic_lora_routing)
            if dynamic_routing_policy == "adaptive_hot_pair_hash":
                adapter_key = str(adapter_id_for_dynamic or "").strip()
                dynamic_routing_policy = "adapter_hash"
                if adapter_key:
                    with dynamic_routing_lock:
                        seen = dynamic_adapter_counts.get(adapter_key, 0) + 1
                        dynamic_adapter_counts[adapter_key] = seen
                        threshold = max(1, int(args.dynamic_lora_hot_pair_threshold or 1))
                        max_hot = max(0, int(args.dynamic_lora_hot_pair_max_adapters or 0))
                        if (
                            seen >= threshold
                            and max_hot > 0
                            and (
                                adapter_key in dynamic_hot_pair_adapters
                                or len(dynamic_hot_pair_adapters) < max_hot
                            )
                        ):
                            dynamic_hot_pair_adapters.add(adapter_key)
                        if adapter_key in dynamic_hot_pair_adapters:
                            dynamic_routing_policy = "adapter_pair_hash"
            target_base_url = _choose_dynamic_lora_base_url(
                base_urls=base_urls,
                request_index=index,
                adapter_id=adapter_id_for_dynamic,
                request_id=item.get("request_id"),
                routing=dynamic_routing_policy,
            )
        else:
            target_base_url = base_urls[index % len(base_urls)]
        request_remote_metrics: Dict[str, Any] = {}
        if request_remote_fetcher is not None:
            adapter_key = str(adapter_id_for_dynamic or "").strip()
            target_path = request_remote_map.get(adapter_key)
            if not target_path:
                completion_offset_s = time.perf_counter() - start_time
                scheduled_offset_s = float(item["arrival_time_s"])
                failed_e2e_ms = max(0.0, (completion_offset_s - scheduled_offset_s) * 1000.0)
                result = {
                    "request_id": item.get("request_id"),
                    "generation_seed": _derive_request_generation_seed(
                        args.generation_seed,
                        item.get("request_id"),
                        index,
                    ),
                    "arrival_time_s": scheduled_offset_s,
                    "dispatch_offset_s": completion_offset_s,
                    "completion_offset_s": completion_offset_s,
                    "adapter_id": item.get("adapter_id"),
                    "ttft_ms": None,
                    "e2e_ms": failed_e2e_ms,
                    "overall_ttft_ms": None,
                    "overall_e2e_ms": failed_e2e_ms,
                    "service_ttft_ms": None,
                    "service_e2e_ms": None,
                    "dispatch_admission_wait_ms": None,
                    "replay_dispatch_wait_ms": None,
                    "status_code": None,
                    "success": False,
                    "prompt_tokens": int(item.get("expected_input_tokens", 0) or 0),
                    "completion_tokens": 0,
                    "total_tokens": int(item.get("expected_input_tokens", 0) or 0),
                    "cost_usd": 0.0,
                    "error": f"adapter {adapter_key!r} is absent from request remote adapter map",
                    "target_base_url": target_base_url,
                }
                with lock:
                    results[index] = result
                    live = _build_live_stats(results, ttft_slo_ms=float(args.ttft_slo_ms))
                    print(
                        f"[ERROR] request remote adapter map miss for request={result['request_id']} "
                        f"adapter={adapter_key}",
                        file=sys.stderr,
                        flush=True,
                    )
                    if (
                        int(args.abort_after_failures or 0) > 0
                        and live["done"] >= max(1, int(args.abort_failures_min_done or 1))
                        and live["failed"] >= int(args.abort_after_failures)
                    ):
                        os._exit(80)
                return
            try:
                request_remote_metrics = request_remote_fetcher.ensure(adapter_key, target_path)
            except Exception as exc:  # noqa: BLE001
                completion_offset_s = time.perf_counter() - start_time
                scheduled_offset_s = float(item["arrival_time_s"])
                failed_e2e_ms = max(0.0, (completion_offset_s - scheduled_offset_s) * 1000.0)
                result = {
                    "request_id": item.get("request_id"),
                    "generation_seed": _derive_request_generation_seed(
                        args.generation_seed,
                        item.get("request_id"),
                        index,
                    ),
                    "arrival_time_s": scheduled_offset_s,
                    "dispatch_offset_s": completion_offset_s,
                    "completion_offset_s": completion_offset_s,
                    "adapter_id": item.get("adapter_id"),
                    "ttft_ms": None,
                    "e2e_ms": failed_e2e_ms,
                    "overall_ttft_ms": None,
                    "overall_e2e_ms": failed_e2e_ms,
                    "service_ttft_ms": None,
                    "service_e2e_ms": None,
                    "dispatch_admission_wait_ms": None,
                    "replay_dispatch_wait_ms": None,
                    "status_code": None,
                    "success": False,
                    "prompt_tokens": int(item.get("expected_input_tokens", 0) or 0),
                    "completion_tokens": 0,
                    "total_tokens": int(item.get("expected_input_tokens", 0) or 0),
                    "cost_usd": 0.0,
                    "error": str(exc),
                    "target_base_url": target_base_url,
                }
                with lock:
                    results[index] = result
                    live = _build_live_stats(results, ttft_slo_ms=float(args.ttft_slo_ms))
                    print(
                        f"[ERROR] request remote adapter fetch failed for request={result['request_id']} "
                        f"adapter={adapter_key}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    if (
                        int(args.abort_after_failures or 0) > 0
                        and live["done"] >= max(1, int(args.abort_failures_min_done or 1))
                        and live["failed"] >= int(args.abort_after_failures)
                    ):
                        os._exit(80)
                return
        dynamic_lora_metrics: Dict[str, Any] = {}
        dynamic_lora_acquired = False
        if dynamic_lora_loader is not None:
            try:
                dynamic_lora_metrics = dynamic_lora_loader.acquire(
                    target_base_url,
                    str(adapter_id_for_dynamic) if adapter_id_for_dynamic is not None else None,
                )
                dynamic_lora_acquired = bool(str(adapter_id_for_dynamic or "").strip())
            except Exception as exc:  # noqa: BLE001
                completion_offset_s = time.perf_counter() - start_time
                scheduled_offset_s = float(item["arrival_time_s"])
                failed_e2e_ms = max(0.0, (completion_offset_s - scheduled_offset_s) * 1000.0)
                result = {
                    "request_id": item.get("request_id"),
                    "generation_seed": _derive_request_generation_seed(
                        args.generation_seed,
                        item.get("request_id"),
                        index,
                    ),
                    "arrival_time_s": scheduled_offset_s,
                    "dispatch_offset_s": completion_offset_s,
                    "completion_offset_s": completion_offset_s,
                    "adapter_id": item.get("adapter_id"),
                    "ttft_ms": None,
                    "e2e_ms": failed_e2e_ms,
                    "overall_ttft_ms": None,
                    "overall_e2e_ms": failed_e2e_ms,
                    "service_ttft_ms": None,
                    "service_e2e_ms": None,
                    "dispatch_admission_wait_ms": None,
                    "replay_dispatch_wait_ms": None,
                    "status_code": None,
                    "success": False,
                    "prompt_tokens": int(item.get("expected_input_tokens", 0) or 0),
                    "completion_tokens": 0,
                    "total_tokens": int(item.get("expected_input_tokens", 0) or 0),
                    "cost_usd": 0.0,
                    "error": str(exc),
                }
                result["target_base_url"] = target_base_url
                result.update(dynamic_lora_metrics)
                with lock:
                    results[index] = result
                    live = _build_live_stats(results, ttft_slo_ms=float(args.ttft_slo_ms))
                    print(
                        f"[ERROR] dynamic LoRA load failed for request={result['request_id']} "
                        f"adapter={result.get('adapter_id')} endpoint={target_base_url}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    if (
                        int(args.abort_after_failures or 0) > 0
                        and live["done"] >= max(1, int(args.abort_failures_min_done or 1))
                        and live["failed"] >= int(args.abort_after_failures)
                    ):
                        print(
                            f"[ERROR] aborting replay after {live['failed']} failed requests "
                            f"among {live['done']} completed requests. This run is invalid.",
                            file=sys.stderr,
                            flush=True,
                        )
                        os._exit(80)
                return
        try:
            result = _replay_one(
                base_url=target_base_url,
                item=item,
                request_index=index,
                timeout_s=args.timeout_s,
                start_time=start_time,
                base_cost_usd=float(args.base_cost_usd),
                input_token_cost_usd=float(args.input_token_cost_usd),
                output_token_cost_usd=float(args.output_token_cost_usd),
                require_server_metrics=bool(args.require_server_metrics),
                model_override=(str(args.model_override) if args.model_override else None),
                adapter_source_field=(str(args.adapter_source_field) if args.adapter_source_field else None),
                adapter_target_field=(str(args.adapter_target_field) if args.adapter_target_field else None),
                adapter_value_map=adapter_value_map,
                drop_body_fields=[str(x) for x in (args.drop_body_field or [])],
                endpoint_path=str(args.endpoint_path),
                convert_chat_to_prompt=bool(args.convert_chat_to_prompt),
                prompt_guard_tokenizer_model=(
                    str(args.prompt_guard_tokenizer_model)
                    if args.prompt_guard_tokenizer_model
                    else None
                ),
                prompt_guard_max_model_len=int(args.prompt_guard_max_model_len or 0),
                prompt_guard_max_input_len=int(args.prompt_guard_max_input_len or 0),
                prompt_guard_max_output_tokens_cap=int(
                    args.prompt_guard_max_output_tokens_cap or 0
                ),
                sglang_native_generate=bool(args.sglang_native_generate),
                slora_native_generate=bool(args.slora_native_generate),
                generation_seed=args.generation_seed,
                empty_success_retries=int(args.empty_success_retries or 0),
                empty_success_retry_delay_s=float(args.empty_success_retry_delay_s or 0.0),
                min_output_tokens=int(args.min_output_tokens or 0),
                include_stream_usage=bool(args.include_stream_usage),
            )
        finally:
            if dynamic_lora_loader is not None and dynamic_lora_acquired:
                dynamic_lora_loader.release(
                    target_base_url,
                    str(adapter_id_for_dynamic) if adapter_id_for_dynamic is not None else None,
                )
        result.update(request_remote_metrics)
        result.update(dynamic_lora_metrics)
        result["target_base_url"] = target_base_url
        with lock:
            results[index] = result
            live = _build_live_stats(results, ttft_slo_ms=float(args.ttft_slo_ms))
            now = time.perf_counter()
            should_print = (
                live["done"] >= len(results)
                or (now - last_live_print_at) >= LIVE_PRINT_INTERVAL_S
            )
            if should_print:
                last_live_print_at = now
                elapsed = max(0.0, now - start_time)
                arrived = sum(1 for offset in arrival_schedule if offset <= elapsed + 1e-9)
                remaining = max(0, len(results) - live["done"])
                eta = (remaining / max(live["done_rps"], 1e-9)) if live["done_rps"] > 1e-9 else 0.0
                backlog = max(0, arrived - live["done"])
                tpot_display = (
                    f"{live['avg_tpot_ms']:.1f}ms"
                    if live["avg_tpot_ms"] is not None
                    else "n/a"
                )
                cache_hit_display = (
                    f"{live['cache_hit_rate']:.0%}"
                    if live["cache_hit_rate"] is not None
                    else "n/a"
                )
                print(
                    f"[live:{args.label}] {_render_progress_bar(live['done'], len(results))} "
                    f"arrived={arrived}/{len(results)} done={live['done']} "
                    f"ok={live['ok']} fail={live['failed']} "
                    f"elapsed={_format_eta(elapsed)} eta={_format_eta(eta)} "
                    f"req/s={live['done_rps']:.2f} tok/s={live['throughput_tok_per_s']:.2f} "
                    f"backlog={backlog} "
                    f"slo@{float(args.ttft_slo_ms):.0f}ms={live['slo_attainment']:.0%} "
                    f"slo_goodput={live['slo_goodput_rps']:.2f}rps/{live['slo_goodput_tok_per_s']:.2f}tok/s",
                    flush=True,
                )
                print(
                    f"[live:{args.label}] "
                    f"ttft_e2e(avg/p95/p99)={live['avg_ttft_ms']:.0f}/{live['p95_ttft_ms']:.0f}/{live['p99_ttft_ms']:.0f}ms "
                    f"ttft_service(avg/p95/p99)={live['avg_service_ttft_ms']:.0f}/{live['p95_service_ttft_ms']:.0f}/{live['p99_service_ttft_ms']:.0f}ms "
                    f"dispatch_wait(avg/p95/p99)={live['avg_dispatch_admission_wait_ms']:.0f}/{live['p95_dispatch_admission_wait_ms']:.0f}/{live['p99_dispatch_admission_wait_ms']:.0f}ms "
                    f"tpot={tpot_display} "
                    f"e2e_e2e(avg/p95/p99)={live['avg_e2e_ms']:.0f}/{live['p95_e2e_ms']:.0f}/{live['p99_e2e_ms']:.0f}ms "
                    f"e2e_service(avg/p95/p99)={live['avg_service_e2e_ms']:.0f}/{live['p95_service_e2e_ms']:.0f}/{live['p99_service_e2e_ms']:.0f}ms",
                    flush=True,
                )
                print(
                    f"[live:{args.label}] "
                    f"tokenproxy/req=${live['token_proxy_avg_cost_usd']:.6f} "
                    f"tokenproxy/1MTok=${live['token_proxy_cost_per_1m_total_tokens_usd']:.2f} "
                    f"tokenproxy_ce={live['token_proxy_ce']:.3f} "
                    f"qpr_legacy={live['qpr_legacy']:.1f} "
                    f"cold_start(avg/p95)="
                    f"{_fmt_optional_pair(live['avg_cold_start_latency_ms'], live['p95_cold_start_latency_ms'])}",
                    flush=True,
                )
                print(
                    f"[live:{args.label}] "
                    f"ttft_comp(avg/p95/p99)={_fmt_optional_triplet(live['avg_comparable_ttft_ms'], live['p95_comparable_ttft_ms'], live['p99_comparable_ttft_ms'])} "
                    f"ttft_warm(avg/p95/p99)={_fmt_optional_triplet(live['avg_warm_standard_ttft_ms'], live['p95_warm_standard_ttft_ms'], live['p99_warm_standard_ttft_ms'])} "
                    f"scaleup_ttft={_fmt_optional_ms(live['avg_scaleup_affected_ttft_ms'])} "
                    f"runtime={_fmt_optional_ms(live['avg_runtime_ttft_ms'])} "
                    f"gpu_ready={_fmt_optional_ms(live['avg_gpu_ready_ttft_ms'])} "
                    f"diag_hit={cache_hit_display} "
                    f"overhead(io+coord)="
                    f"{_fmt_optional_ms(live['avg_serverless_overhead_ms'])}",
                    flush=True,
                )
            if (
                int(args.abort_after_failures or 0) > 0
                and live["done"] >= max(1, int(args.abort_failures_min_done or 1))
                and live["failed"] >= int(args.abort_after_failures)
            ):
                print(
                    f"[ERROR] aborting replay after {live['failed']} failed requests "
                    f"among {live['done']} completed requests. This run is invalid.",
                    file=sys.stderr,
                    flush=True,
                )
                os._exit(80)

    for idx, item in enumerate(requests_list):
        target_offset = float(item["arrival_time_s"]) * max(args.sleep_scale, 0.0)
        while True:
            now_offset = time.perf_counter() - start_time
            wait_s = target_offset - now_offset
            if wait_s <= 0:
                break
            time.sleep(min(wait_s, 0.05))
        thread = threading.Thread(target=_worker, args=(idx, item), daemon=False)
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    final_results = [r for r in results if r is not None]
    output = {
        "metric_schema_version": "e2e_v3",
        "metric_definitions": {
            "primary_ttft": METRIC_DEF_PRIMARY_TTFT,
            "primary_e2e": METRIC_DEF_PRIMARY_E2E,
            "service_ttft": METRIC_DEF_SERVICE_TTFT,
            "service_e2e": METRIC_DEF_SERVICE_E2E,
            "tpot": "(service_completion_time - service_first_token_time) / max(output_tokens - 1, 1)",
        },
        "trace_source": str(args.trace),
        "base_url": args.base_url,
        "base_urls": base_urls,
        "routing_policy": (
            f"dynamic_lora_{args.dynamic_lora_routing}"
            if dynamic_lora_loader is not None and len(base_urls) > 1
            else ("round_robin" if len(base_urls) > 1 else "single_endpoint")
        ),
        "sleep_scale": args.sleep_scale,
        "label": args.label,
        "generation_seed": args.generation_seed,
        "max_requests": int(args.max_requests or 0),
        "abort_after_failures": int(args.abort_after_failures or 0),
        "abort_failures_min_done": int(args.abort_failures_min_done or 1),
        "ttft_slo_ms": float(args.ttft_slo_ms),
        "empty_success_retries": int(args.empty_success_retries or 0),
        "empty_success_retry_delay_s": float(args.empty_success_retry_delay_s or 0.0),
        "min_output_tokens": int(args.min_output_tokens or 0),
        "include_stream_usage": bool(args.include_stream_usage),
        "dynamic_lora_modules": str(args.dynamic_lora_modules) if args.dynamic_lora_modules else None,
        "dynamic_lora_enabled": dynamic_lora_loader is not None,
        "dynamic_lora_routing": str(args.dynamic_lora_routing),
        "dynamic_lora_hot_pair_threshold": int(args.dynamic_lora_hot_pair_threshold or 0),
        "dynamic_lora_hot_pair_max_adapters": int(args.dynamic_lora_hot_pair_max_adapters or 0),
        "dynamic_lora_max_loaded_per_endpoint": int(
            args.dynamic_lora_max_loaded_per_endpoint or 0
        ),
        "dynamic_lora_hot_pair_adapters": sorted(dynamic_hot_pair_adapters),
        "cost_model": {
            "base_cost_usd": float(args.base_cost_usd),
            "input_token_cost_usd": float(args.input_token_cost_usd),
            "output_token_cost_usd": float(args.output_token_cost_usd),
        },
        "results": final_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote replay results -> {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
