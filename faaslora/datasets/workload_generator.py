"""
FaaSLoRA Workload Generator
===========================

Generates realistic LLM inference request traces for experiments, following the
methodology of comparable serverless LLM inference papers:

  S-LoRA (SOSP 2023)        – ShareGPT prompts, Poisson arrivals, Zipf LoRA
  Punica (MLSys 2024)       – ShareGPT + Alpaca, varied concurrency
  ServerlessLLM (NSDI 2024) – Azure LLM Inference Trace + LLM workload
  MuxServe (SC 2024)        – Poisson arrivals, Zipf popularity

Features
--------
* Real Azure LLM Inference Trace for token length distribution (data/azure_llm/*.csv)
* Poisson arrival process (configurable λ in req/s)
* Zipf-distributed LoRA selection with configurable exponent
* Evolving hotness: piecewise-stationary epochs where top-k LoRAs rotate
* Two-phase burst: models scale-up → quiet → scale-up (tests contribution 3)
* ShareGPT prompt distribution (auto-download or embedded fallback)
"""

import random
import time
import asyncio
import hashlib
import json
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from ..clock import local_monotonic_clock_id

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RequestTrace:
    """A single generated inference request."""
    request_id: str
    arrival_time: float          # seconds since experiment start
    prompt: str
    adapter_id: Optional[str]   # Formal many-LoRA experiments require every request to bind a LoRA
    adapter_domain: Optional[str]
    expected_input_tokens: int
    expected_output_tokens: int
    chat_messages: Optional[List[Dict[str, Any]]] = None
    prompt_input_tokens: Optional[int] = None
    prompt_output_tokens: Optional[int] = None
    is_burst: bool = False       # Part of a simulated burst / scale-up event


@dataclass(frozen=True)
class FrozenReplayEntry:
    request_id: str
    offset_s: float
    source_json: str
    source_sha256: str


@dataclass(frozen=True)
class FrozenReplayPlan:
    """Read-only view of an existing trace; never generates requests or weights.

    W1 changes only the arrival index by the approved formula. Prefix/rate
    options are explicit diagnostic views, not new independent trace seeds.
    """
    path: str
    source_sha256: str
    entries: Tuple[FrozenReplayEntry, ...]
    profile: str
    rate_scale: float
    source_count: int

    @classmethod
    def load(cls, path, *, profile='W0', count=None, rate_scale=1.0):
        source = Path(path).resolve(strict=True)
        raw = source.read_bytes()
        rows = json.loads(raw)['requests']
        if not rows or not math.isfinite(rate_scale) or rate_scale <= 0:
            raise ValueError('replay requires a nonempty frozen trace and positive finite rate')
        if profile not in {'W0', 'W1'}:
            raise ValueError('unqualified replay transform; W2 requires its frozen adapter map')
        indexed = []
        for i, row in enumerate(rows):
            rid = str(row.get('request_id', f'req_{i:05d}'))
            arrival = float(row.get('arrival_time_s', row.get('arrival_time', 0)))
            if not rid or not math.isfinite(arrival) or arrival < 0:
                raise ValueError('invalid frozen request identity/arrival')
            indexed.append((arrival, rid, row))
        indexed.sort(key=lambda x: (x[0], x[1]))
        if len({r[1] for r in indexed}) != len(indexed):
            raise ValueError('duplicate request ID in frozen trace')
        if count is not None:
            if not isinstance(count, int) or not 0 < count <= len(indexed):
                raise ValueError('diagnostic prefix exceeds existing trace')
            indexed = indexed[:count]
        base, phase_base, previous = indexed[0][0], 0., 0.
        entries = []
        for i, (arrival, rid, row) in enumerate(indexed):
            if profile == 'W1' and i % 500 == 0:
                base = arrival
                phase_base = 0. if i == 0 else previous+30.
            offset = ((arrival-base)/2+phase_base) if profile == 'W1' else arrival-base
            previous = offset
            canonical = json.dumps(row, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
            entries.append(FrozenReplayEntry(rid, offset/rate_scale, canonical,
                           hashlib.sha256(canonical.encode()).hexdigest()))
        return cls(str(source), hashlib.sha256(raw).hexdigest(), tuple(entries),
                   profile, rate_scale, len(rows))

    def identity(self):
        view = [[e.request_id, e.offset_s, e.source_sha256] for e in self.entries]
        return {'source_path': self.path, 'source_sha256': self.source_sha256,
                'view_sha256': hashlib.sha256(json.dumps(view, separators=(',', ':')).encode()).hexdigest(),
                'count': len(self.entries), 'source_count': self.source_count,
                'profile': self.profile, 'rate_scale': self.rate_scale}


async def publish_frozen_replay(plan, address, nonce, receive_start, emit):
    """Auxiliary-process open loop; transport never blocks the arrival producer.

    A finite trace bounds the pending queue. No request is released early, no
    connection/admission limit changes its planned time, and no response is
    awaited. Backpressure appears in queued-to-submit spans. Service ingress is
    recorded by the receiver, NOT inferred from successful socket writes.
    """
    queue = asyncio.Queue()
    connected = asyncio.get_running_loop().create_future()
    counts = {'N_plan': len(plan.entries), 'N_arrived': 0, 'N_submitted': 0}
    clock_id = local_monotonic_clock_id()
    context = None
    producer = sender = group = None
    writer = None
    frame_limit = max(len(e.source_json.encode()) for e in plan.entries)+8192

    async def accept(reader, candidate):
        try:
            if connected.done():
                raise ValueError('only one service consumer is permitted per replay')
            hello = json.loads(await asyncio.wait_for(reader.readline(), timeout=10))
            if hello != {'nonce': nonce, 'view_sha256': plan.identity()['view_sha256'], 'clock_id': clock_id}:
                raise ValueError('replay consumer identity/clock differs')
            if connected.done():
                raise ValueError('duplicate replay consumer')
            connected.set_result(candidate)
        except Exception as exc:
            candidate.close()
            if not connected.done():
                connected.set_exception(exc)

    server = await asyncio.start_unix_server(accept, path=address, limit=frame_limit)
    emit({'event': 'replay_ready', 'plan': plan.identity(), 'clock_id': clock_id,
          'frame_limit': frame_limit})
    try:
        context = await receive_start()
        notice, t0 = context['deployment_notice_s'], context['replay_t0_s']
        if (context.get('clock_id') != clock_id or not all(map(math.isfinite, (notice, t0)))
                or t0 < notice or notice > time.perf_counter()):
            raise ValueError('invalid deployment-notice origin/clock')

        async def produce():
            for index, entry in enumerate(plan.entries):
                planned = t0+entry.offset_s
                await asyncio.sleep(max(0., planned-time.perf_counter()))
                created = time.perf_counter()
                event = {'index': index, 'request_id': entry.request_id,
                         'planned_arrival_s': planned, 'task_created_s': created,
                         'source_item_sha256': entry.source_sha256}
                counts['N_arrived'] += 1
                emit({'event': 'request_created', **event})
                queue.put_nowait((event, entry.source_json))
            queue.put_nowait(None)

        async def send():
            nonlocal writer
            writer = await connected
            writer.write((json.dumps({'event': 'replay_header', **context,
                                     'plan': plan.identity(), 'frame_limit': frame_limit})+'\n').encode())
            await writer.drain()
            while True:
                item = await queue.get()
                if item is None:
                    break
                event, payload = item
                submitted = time.perf_counter()
                event = {**event, 'client_submit_s': submitted}
                # Full frozen request is transported, not just a timer/index hint.
                packet = {'event': 'request', **event, 'source_request': json.loads(payload)}
                writer.write((json.dumps(packet, separators=(',', ':'), ensure_ascii=False)+'\n').encode())
                await writer.drain()
                counts['N_submitted'] += 1
                emit({'event': 'request_submitted', **event, 'socket_drain_s': time.perf_counter()})
            writer.write((json.dumps({'event': 'replay_end', **counts})+'\n').encode())
            await writer.drain()

        producer = asyncio.create_task(produce())
        sender = asyncio.create_task(send())
        # A finite launcher/qualification deadline is independent of readiness.
        timeout = max(0., t0-time.perf_counter())+plan.entries[-1].offset_s+1800.
        group = asyncio.gather(producer, sender)
        await asyncio.wait_for(group, timeout)
        emit({'event': 'replay_complete', **counts, 'clock_id': clock_id})
        return counts
    except BaseException as exc:
        emit({'event': 'replay_incomplete', **counts, 'error': type(exc).__name__+': '+str(exc)})
        raise
    finally:
        for task in (producer, sender):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(*(t for t in (producer, sender) if t is not None), return_exceptions=True)
        if group is not None:
            await asyncio.gather(group, return_exceptions=True)
        if connected.done() and not connected.cancelled():
            connected.exception()
        if writer is not None:
            writer.close()
            await writer.wait_closed()
        server.close()
        await server.wait_closed()


class ExternalReplayIngress:
    """Service-side receipt of due requests; no look-ahead arrival-rate oracle."""
    def __init__(self, plan, context):
        self.plan, self.context = plan, context
        if context['plan'] != plan.identity() or context['clock_id'] != local_monotonic_clock_id():
            raise ValueError('external replay trace/view/clock differs from service')
        self.observed_times = []
        self.records = {}
        self.complete = False

    async def receive(self):
        reader, writer = await asyncio.open_unix_connection(
            self.context['address'], limit=self.context['frame_limit'])
        try:
            writer.write((json.dumps({'nonce': self.context['nonce'],
                 'view_sha256': self.plan.identity()['view_sha256'],
                 'clock_id': local_monotonic_clock_id()})+'\n').encode())
            await writer.drain()
            header = json.loads(await reader.readline())
            for name in ('plan', 'clock_id', 'deployment_notice_s', 'replay_t0_s'):
                if header.get(name) != self.context[name]:
                    raise ValueError('publisher origin differs from guarded launch')
            for index, entry in enumerate(self.plan.entries):
                packet = json.loads(await reader.readline())
                received = time.perf_counter()
                payload = json.dumps(packet.get('source_request'), sort_keys=True,
                                     separators=(',', ':'), ensure_ascii=False)
                expected_time = self.context['replay_t0_s']+entry.offset_s
                times = [packet.get(name, math.nan) for name in
                         ('planned_arrival_s', 'task_created_s', 'client_submit_s')]+[received]
                if (packet.get('event') != 'request' or packet.get('index') != index
                        or packet.get('request_id') != entry.request_id
                        or packet.get('source_item_sha256') != entry.source_sha256
                        or hashlib.sha256(payload.encode()).hexdigest() != entry.source_sha256
                        or times[0] != expected_time or not all(map(math.isfinite, times))
                        or times != sorted(times)):
                    raise ValueError('invalid/duplicate/early external request or changed content')
                record = {k: v for k, v in packet.items() if k not in {'source_request', 'event'}}
                record['server_received_s'] = received
                self.observed_times.append(received)
                self.records[entry.request_id] = record
                yield index, record
            end = json.loads(await reader.readline())
            if end != {'event': 'replay_end', 'N_plan': len(self.plan.entries),
                       'N_arrived': len(self.plan.entries), 'N_submitted': len(self.plan.entries)}:
                raise ValueError('incomplete external replay denominator')
            self.complete = True
        finally:
            writer.close()
            await writer.wait_closed()


@dataclass
class WorkloadConfig:
    """Workload generation parameters."""
    # Request arrival
    arrival_rate_rps: float = 4.0   # λ: mean requests/second (Poisson)
    total_requests: int = 60

    # LoRA adapter settings
    num_adapters: int = 6
    lora_request_ratio: float = 1.0   # formal many-LoRA experiments bind a LoRA on every request
    zipf_exponent: float = 1.0        # α: Zipf distribution exponent
    active_adapter_cap: Optional[int] = None
    hotset_rotation_requests: int = 0
    hotset_rotation_mode: str = "legacy"
    hotset_overlap_fraction: float = 0.75

    # Hotness evolution (piecewise-stationary)
    enable_hotness_evolution: bool = True
    epoch_requests: int = 20          # rotate hotness every N requests
    top_k_ratio: float = 0.3          # top-k fraction in each epoch
    rotation_ratio: float = 0.2       # fraction of top-k that rotates each epoch

    # Burst simulation (single-phase)
    enable_burst: bool = True
    burst_start_fraction: float = 0.4  # burst at 40% through experiment
    burst_duration_requests: int = 10  # how many requests in the burst
    burst_multiplier: float = 3.0      # λ multiplied during burst

    # Two-phase burst (for contribution 3 demonstration)
    # Phase 1 burst → quiet → Phase 2 burst
    enable_two_phase_burst: bool = False
    phase1_start_fraction: float = 0.2
    phase1_duration_requests: int = 15
    quiet_duration_requests: int = 20   # low-load period between bursts
    phase2_start_fraction: float = 0.7
    phase2_duration_requests: int = 15
    burst_multiplier_two_phase: float = 4.0

    # Token length distribution (overridden by Azure trace when available)
    use_azure_trace_tokens: bool = True   # use real Azure LLM trace token lengths
    input_mean_tokens: int = 128
    input_std_tokens: int = 95
    output_mean_tokens: int = 185
    output_std_tokens: int = 120

    # Adapter domain to task mapping
    adapter_domain_map: Dict[str, str] = field(default_factory=dict)


def _hotset_rotation_stride(
    active_cap: int,
    *,
    rotation_mode: str,
    overlap_fraction: float,
) -> int:
    """Return the adapter-window stride without changing legacy semantics."""
    active_cap = max(1, int(active_cap or 1))
    mode = str(rotation_mode or "legacy").strip().lower()
    if mode == "legacy":
        return max(1, active_cap // 4)
    if mode == "stationary":
        return 0
    if mode == "abrupt":
        return active_cap
    if mode == "gradual":
        overlap = min(1.0, max(0.0, float(overlap_fraction)))
        return max(1, int(round(active_cap * (1.0 - overlap))))
    raise ValueError(
        f"Unsupported hotset_rotation_mode={rotation_mode!r}; "
        "expected legacy, stationary, abrupt, or gradual"
    )


# ---------------------------------------------------------------------------
# WorkloadGenerator
# ---------------------------------------------------------------------------

class WorkloadGenerator:
    """
    Generates realistic request traces for FaaSLoRA experiments.

    Uses real Azure LLM traces for token lengths and Poisson + Zipf for
    arrival/adapter selection, exactly matching S-LoRA / ServerlessLLM methodology.

    Usage
    -----
    gen = WorkloadGenerator(adapter_ids, config)
    traces = gen.generate()              # synchronous
    traces = await gen.generate_async()  # async with inter-arrival delays
    """

    def __init__(
        self,
        adapter_ids: List[str],
        config: Optional[WorkloadConfig] = None,
        seed: int = 42,
        dataset: Any = None,   # WorkloadDataset instance (optional)
    ):
        self.adapter_ids = adapter_ids
        self.config = config or WorkloadConfig()
        self.rng = random.Random(seed)
        self._dataset = dataset  # pre-loaded WorkloadDataset
        self._azure_records: Optional[List] = None
        self._hotness_cache: Optional[List[float]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _load_azure_records(self):
        """Lazily load Azure LLM trace for token distribution."""
        if self._azure_records is not None:
            return
        if self._dataset and self._dataset.has_real_azure_data():
            self._azure_records = self._dataset._azure_records
            return
        # Try direct load
        try:
            from .dataset_loader import AzureTraceLoader
            loader = AzureTraceLoader()
            self._azure_records = loader.load()
        except Exception:
            self._azure_records = []

    def generate(self) -> List[RequestTrace]:
        """Generate the full request trace list synchronously."""
        cfg = self.config
        self._load_azure_records()

        traces: List[RequestTrace] = []
        current_time = 0.0
        weights = self._compute_zipf_weights()

        # Build burst schedule
        burst_schedule = self._build_burst_schedule(cfg)

        for i in range(cfg.total_requests):
            if cfg.enable_hotness_evolution and i > 0 and i % cfg.epoch_requests == 0:
                weights = self._rotate_hotness(weights)

            burst_type = burst_schedule.get(i, "normal")
            is_burst   = burst_type != "normal"

            if burst_type == "phase1" or (not cfg.enable_two_phase_burst and is_burst):
                lam = cfg.arrival_rate_rps * cfg.burst_multiplier
            elif burst_type == "phase2":
                lam = cfg.arrival_rate_rps * cfg.burst_multiplier_two_phase
            elif burst_type == "quiet":
                lam = cfg.arrival_rate_rps * 0.2   # very low load
                is_burst = False
            else:
                lam = cfg.arrival_rate_rps

            inter_arrival = self.rng.expovariate(max(lam, 0.01))
            current_time += inter_arrival

            adapter_id = None
            adapter_domain = None
            if self.adapter_ids and self.rng.random() < cfg.lora_request_ratio:
                sample_ids = self.adapter_ids
                sample_weights = weights
                active_cap = None
                if cfg.active_adapter_cap is not None and cfg.active_adapter_cap > 0:
                    active_cap = min(int(cfg.active_adapter_cap), len(self.adapter_ids))
                if active_cap and active_cap < len(self.adapter_ids):
                    rotation_every = max(0, int(cfg.hotset_rotation_requests or 0))
                    rotation_mode = str(cfg.hotset_rotation_mode or "legacy").strip().lower()
                    rotation_stride = _hotset_rotation_stride(
                        active_cap,
                        rotation_mode=rotation_mode,
                        overlap_fraction=cfg.hotset_overlap_fraction,
                    )
                    if rotation_every > 0 and rotation_mode != "stationary":
                        window_index = i // rotation_every
                        start = (window_index * rotation_stride) % len(self.adapter_ids)
                    else:
                        start = 0
                    ordered_ids = self.adapter_ids[start:] + self.adapter_ids[:start]
                    sample_ids = ordered_ids[:active_cap]
                    sample_weights = weights[:len(sample_ids)]
                adapter_id = self.rng.choices(sample_ids, weights=sample_weights, k=1)[0]
                adapter_domain = cfg.adapter_domain_map.get(adapter_id, "general")

            prompt, in_tokens, out_tokens, prompt_in_tokens, prompt_out_tokens = self._sample_request(adapter_domain)

            traces.append(RequestTrace(
                request_id=f"req_{i:05d}",
                arrival_time=current_time,
                prompt=prompt,
                adapter_id=adapter_id,
                adapter_domain=adapter_domain,
                expected_input_tokens=in_tokens,
                expected_output_tokens=out_tokens,
                prompt_input_tokens=prompt_in_tokens,
                prompt_output_tokens=prompt_out_tokens,
                is_burst=is_burst,
            ))

        return traces

    def _build_burst_schedule(self, cfg: WorkloadConfig) -> Dict[int, str]:
        """Map request index → burst type."""
        schedule: Dict[int, str] = {}
        if cfg.enable_two_phase_burst:
            p1_start = int(cfg.total_requests * cfg.phase1_start_fraction)
            p1_end   = p1_start + cfg.phase1_duration_requests
            q_start  = p1_end
            q_end    = q_start + cfg.quiet_duration_requests
            p2_start = int(cfg.total_requests * cfg.phase2_start_fraction)
            p2_end   = p2_start + cfg.phase2_duration_requests
            for i in range(p1_start, min(p1_end, cfg.total_requests)):
                schedule[i] = "phase1"
            for i in range(q_start, min(q_end, cfg.total_requests)):
                schedule[i] = "quiet"
            for i in range(p2_start, min(p2_end, cfg.total_requests)):
                schedule[i] = "phase2"
        elif cfg.enable_burst:
            b_start = int(cfg.total_requests * cfg.burst_start_fraction)
            b_end   = b_start + cfg.burst_duration_requests
            for i in range(b_start, min(b_end, cfg.total_requests)):
                schedule[i] = "phase1"
        return schedule

    async def replay_async(
        self,
        traces: List[RequestTrace],
        callback,
    ):
        """
        Replay a pre-generated trace, calling `callback(trace)` at the correct
        inter-arrival times.  `callback` should be an async function.
        """
        start_real = time.perf_counter()
        for trace in traces:
            # Wait until scheduled arrival time
            elapsed = time.perf_counter() - start_real
            wait = trace.arrival_time - elapsed
            if wait > 0:
                await asyncio.sleep(wait)
            asyncio.create_task(callback(trace))

    # ------------------------------------------------------------------
    # Statistics helpers
    # ------------------------------------------------------------------

    def get_adapter_popularity(self, traces: List[RequestTrace]) -> Dict[str, float]:
        """Return normalized request rate per adapter."""
        counts: Dict[str, int] = {}
        lora_reqs = [t for t in traces if t.adapter_id]
        for t in lora_reqs:
            counts[t.adapter_id] = counts.get(t.adapter_id, 0) + 1
        total = max(sum(counts.values()), 1)
        return {k: v / total for k, v in sorted(counts.items(), key=lambda x: -x[1])}

    def get_burst_stats(self, traces: List[RequestTrace]) -> Dict[str, Any]:
        """Return burst-period statistics."""
        burst = [t for t in traces if t.is_burst]
        non_burst = [t for t in traces if not t.is_burst]
        if not burst or not non_burst:
            return {"burst_enabled": False}

        def rate(ts):
            if len(ts) < 2:
                return 0.0
            span = ts[-1].arrival_time - ts[0].arrival_time
            return len(ts) / span if span > 0 else 0.0

        return {
            "burst_enabled": True,
            "burst_requests": len(burst),
            "non_burst_requests": len(non_burst),
            "burst_rate_rps": rate(burst),
            "baseline_rate_rps": rate(non_burst),
        }

    # ------------------------------------------------------------------
    # Optional: try to load real ShareGPT data from HuggingFace
    # ------------------------------------------------------------------

    @staticmethod
    def try_load_sharegpt(max_samples: int = 500) -> Optional[List[Tuple[str, int, int]]]:
        """
        Attempt to download a small slice of ShareGPT.
        Returns list of (prompt, input_tokens, output_tokens) or None on failure.
        """
        try:
            from datasets import load_dataset
            ds = load_dataset(
                "anon8231489123/ShareGPT_Vicuna_unfiltered",
                data_files="ShareGPT_V3_unfiltered_cleaned_split.json",
                split="train",
                streaming=True,
            )
            results = []
            for item in ds:
                if len(results) >= max_samples:
                    break
                convs = item.get("conversations") or item.get("conversation", [])
                if not convs:
                    continue
                human_turns = [c["value"] for c in convs if c.get("from") == "human"]
                gpt_turns   = [c["value"] for c in convs if c.get("from") == "gpt"]
                if human_turns and gpt_turns:
                    prompt = human_turns[0]
                    in_t   = len(prompt.split())
                    out_t  = sum(len(g.split()) for g in gpt_turns)
                    if 10 <= in_t <= 1500 and 20 <= out_t <= 1500:
                        results.append((prompt, in_t, out_t))
            return results if results else None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_zipf_weights(self) -> List[float]:
        """Initial Zipf weights based on adapter rank."""
        n = len(self.adapter_ids)
        if n == 0:
            return []
        alpha = self.config.zipf_exponent
        # rank-1 gets highest weight
        ranks = list(range(1, n + 1))
        raw = [1.0 / (r ** alpha) for r in ranks]
        total = sum(raw)
        self._hotness_cache = [w / total for w in raw]
        return list(self._hotness_cache)

    def _rotate_hotness(self, weights: List[float]) -> List[float]:
        """
        Rotate top-k adapters to model time-varying hotness (piecewise-stationary).
        A fraction `rotation_ratio` of the top-k adapters are replaced by random
        cold adapters, simulating shifting workloads.
        """
        cfg = self.config
        n = len(weights)
        k = max(1, int(n * cfg.top_k_ratio))
        r = max(1, int(k * cfg.rotation_ratio))

        # Identify current top-k and cold adapters
        ranked = sorted(range(n), key=lambda i: -weights[i])
        top_k  = set(ranked[:k])
        cold   = [i for i in ranked[k:]]

        if not cold:
            return weights

        # Swap `r` top-k adapters with `r` cold ones
        to_demote = self.rng.sample(list(top_k), min(r, len(top_k)))
        to_promote = self.rng.sample(cold, min(r, len(cold)))

        new_weights = list(weights)
        for d, p in zip(to_demote, to_promote):
            new_weights[d], new_weights[p] = new_weights[p], new_weights[d]

        return new_weights

    def _sample_request(
        self, domain: Optional[str] = None
    ) -> Tuple[str, int, int, Optional[int], Optional[int]]:
        """
        Sample (prompt, input_tokens, output_tokens).
        Token lengths come from Azure LLM trace when available.
        Prompt text comes from ShareGPT / embedded dataset.
        """
        # Token lengths: prefer real Azure trace distribution
        if self._azure_records and self.config.use_azure_trace_tokens:
            rec = self.rng.choice(self._azure_records)
            in_tokens  = max(10, rec.context_tokens)
            out_tokens = max(10, rec.generated_tokens)
        else:
            # Fall back to Gaussian with ShareGPT-calibrated parameters
            in_tokens  = max(10, int(self.rng.gauss(
                self.config.input_mean_tokens, self.config.input_std_tokens)))
            out_tokens = max(10, int(self.rng.gauss(
                self.config.output_mean_tokens, self.config.output_std_tokens)))

        prompt_input_tokens: Optional[int] = None
        prompt_output_tokens: Optional[int] = None
        # Prompt text
        if self._dataset and self._dataset._sgpt_records:
            matched = self._dataset._select_sharegpt_record(
                self.rng,
                target_input_tokens=in_tokens,
                target_output_tokens=out_tokens,
            )
            if matched is not None:
                prompt = matched.prompt
                prompt_input_tokens = max(1, int(matched.input_tokens or 1))
                prompt_output_tokens = max(1, int(matched.output_tokens or 1))
            else:
                prompt = self.rng.choice(self._dataset._sgpt_records).prompt
        else:
            from .sharegpt_prompts import PROMPT_DOMAINS, ALL_PROMPTS
            pool = PROMPT_DOMAINS.get(domain, ALL_PROMPTS) if domain else ALL_PROMPTS
            prompt, prompt_input_tokens, prompt_output_tokens = self.rng.choice(pool)

        return prompt, in_tokens, out_tokens, prompt_input_tokens, prompt_output_tokens

    def _sample_prompt(self, domain: Optional[str] = None) -> Tuple[str, int, int, Optional[int], Optional[int]]:
        """Alias for backward compatibility."""
        return self._sample_request(domain)
