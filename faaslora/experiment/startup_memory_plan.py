"""Exact, successfully allocated KV memory reuse via the native worker API.

The vLLM startup-plan proposal is not necessarily the allocation used by its
producer. This opt-in keeps the actual allocation instead. Native profiling,
profile_run, service capture and the native free-memory applicability rule
remain authoritative. No compiler hash or installed backend is modified.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

POLICY = 'exact_native_startup_memory_v1'


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                   separators=(',', ':')).encode()).hexdigest()


def engine_memory_contract(model_config, engine_kwargs):
    """Bind all effective public engine arguments, not just the graph hash."""
    policy = model_config.get('ieee_startup_memory_policy')
    if policy is None:
        return None
    if (policy != POLICY or engine_kwargs.get('tensor_parallel_size') != 1
            or engine_kwargs.get('dtype') != 'float16'
            or engine_kwargs.get('enforce_eager') is not False
            or not engine_kwargs.get('enable_lora')
            or not str(engine_kwargs.get('worker_cls', '')).endswith('.IEEEReachableGraphWorker')
            or engine_kwargs.get('kv_cache_memory_bytes') is not None):
        raise ValueError('startup memory reuse requires the qualified native FP16 LoRA worker')
    # kwargs excludes placement; logical replicas on equivalent devices can
    # reuse memory. Native/profile IDs remain included (a conservative miss).
    contract = dict(policy=POLICY, engine_arguments=engine_kwargs)
    # Round-trip prevents a mutable self-reference when additional_config is
    # subsequently extended, and rejects non-JSON implicit runtime objects.
    return json.loads(json.dumps(contract, sort_keys=True, allow_nan=False))


def runtime_memory_identity(worker, contract):
    """Collect once, before native profiling mutates cache/config state."""
    import torch
    import vllm
    import vllm.envs as envs
    from vllm.compilation.compiler_interface import get_inductor_factors
    from vllm.config.utils import hash_factors
    from vllm.utils.hashing import safe_hash
    from vllm.v1.worker.startup_plan import compute_plan_fingerprint
    from vllm.utils.import_utils import import_pynvml

    nvml = import_pynvml()
    nvml.nvmlInit()
    try:
        driver = nvml.nvmlSystemGetDriverVersion()
    finally:
        nvml.nvmlShutdown()
    if isinstance(driver, bytes):
        driver = driver.decode('ascii')
    model = Path(worker.model_config.model).resolve(strict=True)
    weights = sorted(model.glob('*.safetensors'))
    if not weights or worker.model_config.is_multimodal_model:
        raise ValueError('startup memory reuse is qualified only for local text safetensors models')
    # The campaign already seals model content. Avoid re-reading GBs per
    # activation; bind the sealed local files' metadata, not model name alone.
    files = {}
    for path in [*weights, *sorted(model.glob('*.json'))]:
        stat = path.stat()
        files[path.name] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                               inode=stat.st_ino, device=stat.st_dev)
    sources = {}
    base = Path(vllm.__file__).parent
    for path in [Path(__file__), Path(__file__).with_name('vllm_reachable_graph_worker.py'),
                 base / 'v1/worker/gpu_worker.py', base / 'v1/worker/startup_plan.py',
                 base / 'v1/worker/gpu/model_runner.py']:
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    # Native Inductor hashes str(factors), not JSON: torch_key is bytes and
    # portable configs also contain non-JSON objects. Preserve that encoding.
    compiler_factors = get_inductor_factors()
    return dict(policy=POLICY, engine_contract=contract, driver=driver,
        torch=str(torch.__version__), cuda=torch.version.cuda,
        native_fingerprint=compute_plan_fingerprint(
            worker.vllm_config, worker.rank, worker.parallel_config.world_size),
        model_files=files, sources=sources,
        environment_hash=hash_factors(envs.compile_factors()),
        compiler_hash=safe_hash(str(compiler_factors).encode(), usedforsecurity=False).hexdigest()[:10],
        compiler_component_hashes=[hashlib.sha256(str(v).encode()).hexdigest()
                                   for v in compiler_factors])


class ExactStartupMemoryPlan:
    """Small JSON receipt; first successful native allocation wins per identity."""
    def __init__(self, root, identity, free_memory, requested_memory):
        self.identity = identity
        self.key = digest_json(identity)
        self.path = Path(root) / POLICY / f'{self.key}.json'
        self.free_memory = int(free_memory)
        self.requested_memory = int(requested_memory)
        self.receipt = dict(policy=POLICY, fingerprint=self.key,
            cache_path=str(self.path), free_memory_bytes=self.free_memory,
            requested_memory_bytes=self.requested_memory, status='not_checked')

    def apply(self, worker, native_applicable):
        try:
            plan = json.loads(self.path.read_text())
        except FileNotFoundError:
            self.receipt['status'] = 'miss'
            return
        except (OSError, ValueError):
            self.receipt['status'] = 'unreadable'
            return
        if (not isinstance(plan, dict) or plan.get('schema') != POLICY
                or plan.get('fingerprint') != self.key
                or plan.get('identity') != self.identity
                or type(plan.get('kv_cache_memory_bytes')) is not int
                or type(plan.get('free_memory_baseline')) is not int
                or not isinstance(plan.get('producer'), dict)
                or plan['free_memory_baseline'] <= 0
                or not 0 < plan['kv_cache_memory_bytes'] <= self.requested_memory):
            self.receipt['status'] = 'invalid'
            return
        kv_bytes = native_applicable(plan, self.free_memory)
        if kv_bytes is None:
            self.receipt['status'] = 'insufficient_free_memory'
            return
        if worker.cache_config.kv_cache_memory_bytes is not None:
            raise ValueError('startup memory policy must not replace an explicit KV budget')
        worker.cache_config.kv_cache_memory_bytes = kv_bytes
        self.receipt.update(status='hit', kv_cache_memory_bytes=kv_bytes,
                            producer=plan['producer'])

    def publish(self, kv_bytes):
        """Call only after native KV initialization and service capture succeed."""
        if type(kv_bytes) is not int or not 0 < kv_bytes <= self.requested_memory:
            raise ValueError('native KV allocation is outside its requested budget')
        self.receipt['actual_kv_cache_memory_bytes'] = kv_bytes
        if self.receipt['status'] == 'hit':
            if kv_bytes != self.receipt['kv_cache_memory_bytes']:
                raise RuntimeError('native worker changed the reused KV allocation')
            return
        plan = dict(schema=POLICY, fingerprint=self.key, identity=self.identity,
            kv_cache_memory_bytes=kv_bytes, free_memory_baseline=self.free_memory,
            producer=dict(pid=os.getpid(), time_ns=time.time_ns()))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_name(f'{self.path.name}.tmp.{os.getpid()}')
        try:
            with temp.open('x') as stream:
                json.dump(plan, stream, sort_keys=True, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            # Publish without overwriting a completed concurrent producer.
            try:
                os.link(temp, self.path)
                self.receipt['publication'] = 'created'
            except FileExistsError:
                self.receipt['publication'] = 'existing_preserved'
        finally:
            temp.unlink(missing_ok=True)
