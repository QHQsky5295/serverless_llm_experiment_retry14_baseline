"""
FaaSLoRA Dataset Loader
========================

Loads and integrates the real datasets used by comparable papers:

  Azure LLM Inference Trace  (data/azure_llm/)
  ├─ AzureLLMInferenceTrace_conv.csv   (~19K records, conversation workload)
  └─ AzureLLMInferenceTrace_code.csv   (~8K records, code workload)

  ShareGPT  (optional, downloaded from HuggingFace)
  └─ data/sharegpt/sharegpt_cache.json  (cached after first download)

Usage in experiments (consistent with S-LoRA / ServerlessLLM paper methodology):
  - Inter-arrival times from Azure trace timestamps (Poisson fit)
  - Input/output token counts from Azure trace (ContextTokens / GeneratedTokens)
  - Actual prompt text from ShareGPT (or embedded prompts as fallback)
  - LoRA adapter selection via Zipf distribution over the adapters

References
----------
  Azure trace: "Towards Efficient Generative Large Language Model Serving:
                A Survey from Algorithms to Systems", 2024
  S-LoRA: used ShareGPT for workload, Poisson arrivals
  ServerlessLLM: used Azure trace for arrival modeling
"""

import csv
import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("faaslora.datasets")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
AZURE_CONV_CSV  = REPO_ROOT / "data" / "azure_llm" / "AzureLLMInferenceTrace_conv.csv"
AZURE_CODE_CSV  = REPO_ROOT / "data" / "azure_llm" / "AzureLLMInferenceTrace_code.csv"
SHAREGPT_CACHE  = REPO_ROOT / "data" / "sharegpt" / "sharegpt_cache.json"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AzureRecord:
    timestamp_s: float        # seconds since epoch
    context_tokens: int
    generated_tokens: int
    workload_type: str        # "conv" or "code"


@dataclass
class ShareGPTRecord:
    conversation_id: str
    prompt: str
    input_tokens: int
    output_tokens: int


# ---------------------------------------------------------------------------
# Azure LLM Trace Loader
# ---------------------------------------------------------------------------

class AzureTraceLoader:
    """
    Loads the real Azure LLM inference traces.

    Contains ~28K records with timestamps and token counts,
    matching the data used in ServerlessLLM (NSDI'24).
    """

    def __init__(self):
        self._records: List[AzureRecord] = []
        self._loaded = False

    def load(self, max_records: Optional[int] = None) -> List[AzureRecord]:
        if self._loaded and self._records:
            return self._records if not max_records else self._records[:max_records]

        records: List[AzureRecord] = []
        for csv_path, wtype in [(AZURE_CONV_CSV, "conv"), (AZURE_CODE_CSV, "code")]:
            if not csv_path.exists():
                logger.warning(f"Azure trace not found: {csv_path}")
                continue
            try:
                with open(csv_path, newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            ts = _parse_azure_timestamp(row["TIMESTAMP"])
                            ctx = int(float(row["ContextTokens"]))
                            gen = int(float(row["GeneratedTokens"]))
                            if ctx > 0 and gen > 0:
                                records.append(AzureRecord(ts, ctx, gen, wtype))
                        except (ValueError, KeyError):
                            continue
            except Exception as e:
                logger.warning(f"Error reading {csv_path}: {e}")

        # Sort by timestamp
        records.sort(key=lambda r: r.timestamp_s)
        self._records = records
        self._loaded = True
        logger.info(f"Loaded {len(records)} Azure LLM trace records")
        return records if not max_records else records[:max_records]

    def get_stats(self) -> Dict[str, Any]:
        if not self._records:
            self.load()
        ctx  = [r.context_tokens for r in self._records]
        gen  = [r.generated_tokens for r in self._records]
        iats = _compute_inter_arrivals([r.timestamp_s for r in self._records])
        return {
            "total_records": len(self._records),
            "conv_records": sum(1 for r in self._records if r.workload_type == "conv"),
            "code_records": sum(1 for r in self._records if r.workload_type == "code"),
            "context_tokens": {"mean": _mean(ctx), "p50": _pct(ctx, 50), "p95": _pct(ctx, 95), "p99": _pct(ctx, 99)},
            "generated_tokens": {"mean": _mean(gen), "p50": _pct(gen, 50), "p95": _pct(gen, 95), "p99": _pct(gen, 99)},
            "inter_arrival_s": {"mean": _mean(iats), "p50": _pct(iats, 50), "p95": _pct(iats, 95)},
            "implied_rps": 1.0 / max(_mean(iats), 1e-9) if iats else 0.0,
        }

    def sample_token_lengths(self, n: int, rng: Optional[random.Random] = None) -> List[Tuple[int, int]]:
        """Sample n (input_tokens, output_tokens) pairs from the trace."""
        if not self._records:
            self.load()
        rng = rng or random.Random()
        pool = self._records
        return [(r.context_tokens, r.generated_tokens) for r in rng.choices(pool, k=n)]

    def get_arrival_rate_rps(self) -> float:
        """Estimate the mean arrival rate from the trace."""
        if not self._records:
            self.load()
        if len(self._records) < 2:
            return 1.0
        span = self._records[-1].timestamp_s - self._records[0].timestamp_s
        return len(self._records) / max(span, 1.0)


# ---------------------------------------------------------------------------
# ShareGPT Loader
# ---------------------------------------------------------------------------

class ShareGPTLoader:
    """
    Loads ShareGPT conversations for actual prompt text.

    Priority order:
    1. Local cache (data/sharegpt/sharegpt_cache.json)
    2. HuggingFace download via `datasets` library
    3. Embedded prompts fallback (200 prompts)
    """

    def __init__(self):
        self._records: List[ShareGPTRecord] = []
        self._source = "none"

    def load(self, max_samples: int = 5000) -> List[ShareGPTRecord]:
        """Load ShareGPT prompts. Returns cached/downloaded/embedded data."""
        # 1. Check local cache
        if SHAREGPT_CACHE.exists():
            try:
                return self._load_cache(max_samples)
            except Exception as e:
                logger.warning(f"Cache load failed: {e}, trying download")

        # 2. Try HuggingFace download
        downloaded = self._try_download(max_samples)
        if downloaded:
            self._records = downloaded
            self._save_cache(downloaded)
            self._source = "huggingface"
            logger.info(f"Downloaded {len(downloaded)} ShareGPT records from HuggingFace")
            return downloaded

        # 3. Fallback: embedded prompts
        return self._load_embedded()

    def _load_cache(self, max_samples: int) -> List[ShareGPTRecord]:
        with open(SHAREGPT_CACHE, "r", encoding="utf-8") as f:
            data = json.load(f)
        records = [
            ShareGPTRecord(
                conversation_id=d.get("id", f"sg_{i}"),
                prompt=d["prompt"],
                input_tokens=d["input_tokens"],
                output_tokens=d["output_tokens"],
            )
            for i, d in enumerate(data[:max_samples])
        ]
        self._records = records
        self._source = "cache"
        logger.info(f"Loaded {len(records)} ShareGPT records from cache")
        return records

    def _try_download(self, max_samples: int) -> Optional[List[ShareGPTRecord]]:
        try:
            from datasets import load_dataset
            logger.info("Downloading ShareGPT dataset from HuggingFace (this may take a moment)...")
            ds = load_dataset(
                "anon8231489123/ShareGPT_Vicuna_unfiltered",
                data_files="ShareGPT_V3_unfiltered_cleaned_split.json",
                split="train",
            )
            records = []
            for i, item in enumerate(ds):
                if len(records) >= max_samples:
                    break
                convs = item.get("conversations") or []
                human_turns = [c["value"] for c in convs if c.get("from") == "human"]
                gpt_turns   = [c["value"] for c in convs if c.get("from") == "gpt"]
                if not human_turns or not gpt_turns:
                    continue
                prompt = human_turns[0]
                in_t   = len(prompt.split())
                out_t  = sum(len(g.split()) for g in gpt_turns)
                if 10 <= in_t <= 2048 and 10 <= out_t <= 2048:
                    records.append(ShareGPTRecord(
                        conversation_id=str(item.get("id", i)),
                        prompt=prompt,
                        input_tokens=in_t,
                        output_tokens=out_t,
                    ))
            return records if records else None
        except Exception as e:
            logger.debug(f"ShareGPT download failed: {e}")
            return None

    def _save_cache(self, records: List[ShareGPTRecord]):
        SHAREGPT_CACHE.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {"id": r.conversation_id, "prompt": r.prompt,
             "input_tokens": r.input_tokens, "output_tokens": r.output_tokens}
            for r in records
        ]
        with open(SHAREGPT_CACHE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def _load_embedded(self) -> List[ShareGPTRecord]:
        from .sharegpt_prompts import ALL_PROMPTS
        records = [
            ShareGPTRecord(
                conversation_id=f"embed_{i}",
                prompt=p,
                input_tokens=in_t,
                output_tokens=out_t,
            )
            for i, (p, in_t, out_t) in enumerate(ALL_PROMPTS)
        ]
        self._records = records
        self._source = "embedded"
        logger.info(f"Using {len(records)} embedded ShareGPT-style prompts")
        return records

    def get_source(self) -> str:
        return self._source

    def get_stats(self) -> Dict[str, Any]:
        if not self._records:
            return {"loaded": False}
        in_t  = [r.input_tokens for r in self._records]
        out_t = [r.output_tokens for r in self._records]
        return {
            "source": self._source,
            "total_records": len(self._records),
            "input_tokens":  {"mean": _mean(in_t),  "p50": _pct(in_t,  50), "p95": _pct(in_t,  95)},
            "output_tokens": {"mean": _mean(out_t), "p50": _pct(out_t, 50), "p95": _pct(out_t, 95)},
        }


# ---------------------------------------------------------------------------
# Combined Workload Dataset
# ---------------------------------------------------------------------------

class AzureTraceReplay:
    """
    Replays the real Azure LLM inference trace with authentic arrival timing.

    The 1-hour trace (2023-11-16 18:15 → 19:14) naturally contains periods of
    high and low request density — no artificial burst injection needed.

    time_scale_factor
      1.0  = real-time replay (1 hour trace takes 1 hour)
      0.1  = 10× compressed (1 hour → 6 min)    ← recommended for experiments
      0.05 = 20× compressed (1 hour → 3 min)    ← for --quick runs

    Adapter selection uses Zipf distribution as in S-LoRA/Punica papers.
    """

    def __init__(self, loader: "AzureTraceLoader"):
        self.loader = loader

    def replay(
        self,
        adapter_ids: List[str],
        workload_type: str = "mixed",     # "conv", "code", "mixed"
        zipf_exponent: float = 1.0,
        max_requests: int = 500,
        time_scale_factor: float = 0.1,
        lora_request_ratio: float = 0.85,
        active_adapter_cap: Optional[int] = None,
        hotset_rotation_requests: int = 0,
        domain_map: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ) -> List[Any]:    # returns List[RequestTrace] — imported at runtime to avoid circular
        """
        Generate RequestTrace list from the real Azure trace.

        Arrival times use authentic inter-request gaps from production traffic
        (scaled by time_scale_factor). Token lengths come directly from the trace.
        LoRA adapter selection uses Zipf distribution (α = zipf_exponent).
        """
        from .workload_generator import RequestTrace

        rng = random.Random(seed)
        records = self.loader.load()

        # Filter by workload type
        if workload_type == "conv":
            records = [r for r in records if r.workload_type == "conv"]
        elif workload_type == "code":
            records = [r for r in records if r.workload_type == "code"]
        # "mixed" → all records already sorted by timestamp

        if not records:
            logger.warning(f"No Azure trace records found for workload_type={workload_type}")
            return []

        # Sample uniformly across the full trace span to preserve temporal distribution
        if len(records) > max_requests:
            step = len(records) / max_requests
            selected = [records[int(i * step)] for i in range(max_requests)]
        else:
            selected = list(records)

        if not selected:
            return []

        t0 = selected[0].timestamp_s   # wall-clock origin

        # Zipf weights for LoRA adapter selection
        n_adapters = len(adapter_ids)
        if n_adapters > 0:
            zipf_w = [1.0 / (i + 1) ** zipf_exponent for i in range(n_adapters)]
        else:
            zipf_w = []

        active_cap = None
        if active_adapter_cap is not None and active_adapter_cap > 0:
            active_cap = min(int(active_adapter_cap), n_adapters) if n_adapters > 0 else None
        rotation_every = max(0, int(hotset_rotation_requests or 0))
        rotation_stride = 0
        if active_cap and active_cap < n_adapters:
            rotation_stride = max(1, active_cap // 4)

        traces = []
        for i, rec in enumerate(selected):
            # Real arrival time (scaled)
            arrival = (rec.timestamp_s - t0) * time_scale_factor

            # Zipf LoRA selection
            adapter_id = None
            domain = None
            if adapter_ids and rng.random() < lora_request_ratio:
                sample_ids = adapter_ids
                sample_weights = zipf_w
                if active_cap and active_cap < n_adapters:
                    if rotation_every > 0:
                        window_index = i // rotation_every
                        start = (window_index * rotation_stride) % n_adapters
                    else:
                        start = 0
                    ordered_ids = adapter_ids[start:] + adapter_ids[:start]
                    sample_ids = ordered_ids[:active_cap]
                    sample_weights = [1.0 / (j + 1) ** zipf_exponent for j in range(len(sample_ids))]
                adapter_id = rng.choices(sample_ids, weights=sample_weights, k=1)[0]
                domain = (domain_map or {}).get(adapter_id, "general")

            # Prompt: use embedded or ShareGPT text (token lengths from Azure trace)
            prompt = f"[Azure:{rec.workload_type}] Request {i}"

            traces.append(RequestTrace(
                request_id=f"req_{i:05d}",
                arrival_time=arrival,
                prompt=prompt,
                adapter_id=adapter_id,
                adapter_domain=domain,
                expected_input_tokens=max(10, rec.context_tokens),
                expected_output_tokens=max(5, rec.generated_tokens),
                is_burst=False,   # real data has natural variation; no artificial flag
            ))

        logger.info(
            "Azure trace replay: %d requests  type=%s  scale=%.2f  "
            "adapters=%d  span=%.1fs",
            len(traces), workload_type, time_scale_factor,
            n_adapters, (selected[-1].timestamp_s - t0) * time_scale_factor,
        )
        return traces

    def get_stats(self, workload_type: str = "mixed") -> Dict[str, Any]:
        records = self.loader.load()
        if workload_type != "mixed":
            records = [r for r in records if r.workload_type == workload_type]
        if not records:
            return {}
        iats = _compute_inter_arrivals([r.timestamp_s for r in records])
        ctx  = [r.context_tokens for r in records]
        gen  = [r.generated_tokens for r in records]
        span = records[-1].timestamp_s - records[0].timestamp_s
        return {
            "workload_type": workload_type,
            "total_records": len(records),
            "span_hours": span / 3600,
            "avg_arrival_rps": len(records) / max(span, 1.0),
            "inter_arrival_ms": {
                "mean": _mean(iats) * 1000 if iats else 0,
                "p50":  _pct(iats, 50) * 1000 if iats else 0,
                "p95":  _pct(iats, 95) * 1000 if iats else 0,
            },
            "context_tokens":   {"mean": _mean(ctx), "p50": _pct(ctx, 50), "p95": _pct(ctx, 95), "p99": _pct(ctx, 99)},
            "generated_tokens": {"mean": _mean(gen), "p50": _pct(gen, 50), "p95": _pct(gen, 95), "p99": _pct(gen, 99)},
        }


class WorkloadDataset:
    """
    Combines Azure LLM Inference Trace with ShareGPT prompts.

    For publication-quality experiments:
      - arrival timing: AzureTraceReplay (real 1-hour production timestamps)
      - token lengths:  Azure trace ContextTokens / GeneratedTokens
      - prompt text:    ShareGPT (auto-downloaded) or embedded fallback
      - LoRA selection: Zipf distribution (per S-LoRA / Punica methodology)
    """

    def __init__(
        self,
        azure_loader: Optional[AzureTraceLoader] = None,
        sharegpt_loader: Optional[ShareGPTLoader] = None,
    ):
        self.azure   = azure_loader or AzureTraceLoader()
        self.sgpt    = sharegpt_loader or ShareGPTLoader()
        self.replay  = AzureTraceReplay(self.azure)

        self._azure_records: Optional[List[AzureRecord]] = None
        self._sgpt_records:  Optional[List[ShareGPTRecord]] = None
        self._initialized = False

    def initialize(
        self,
        max_azure: Optional[int] = None,
        max_sgpt: int = 5000,
    ) -> Dict[str, Any]:
        """Load all datasets. Returns a stats summary."""
        self._azure_records = self.azure.load(max_records=max_azure)
        self._sgpt_records  = self.sgpt.load(max_samples=max_sgpt)
        self._initialized = True
        return {
            "azure": self.azure.get_stats(),
            "sharegpt": self.sgpt.get_stats(),
        }

    def generate_traces(
        self,
        adapter_ids: List[str],
        workload_type: str = "mixed",
        zipf_exponent: float = 1.0,
        max_requests: int = 500,
        time_scale_factor: float = 0.1,
        lora_request_ratio: float = 0.85,
        active_adapter_cap: Optional[int] = None,
        hotset_rotation_requests: int = 0,
        domain_map: Optional[Dict] = None,
        seed: int = 42,
    ):
        """
        Generate request traces from the real Azure LLM trace.

        Arrival times use authentic inter-request gaps from the 1-hour production
        trace scaled by time_scale_factor. Token lengths are from the trace.
        LoRA adapter selection uses Zipf distribution.

        Returns List[RequestTrace]
        """
        if not self._initialized:
            self.initialize()
        traces = self.replay.replay(
            adapter_ids=adapter_ids,
            workload_type=workload_type,
            zipf_exponent=zipf_exponent,
            max_requests=max_requests,
            time_scale_factor=time_scale_factor,
            lora_request_ratio=lora_request_ratio,
            active_adapter_cap=active_adapter_cap,
            hotset_rotation_requests=hotset_rotation_requests,
            domain_map=domain_map,
            seed=seed,
        )
        # Enrich prompts with real ShareGPT text if available
        if traces and self._sgpt_records:
            rng = random.Random(seed)
            for t in traces:
                t.prompt = rng.choice(self._sgpt_records).prompt
        return traces

    def sample_request(
        self,
        rng: random.Random,
        prefer_domain: Optional[str] = None,
    ) -> Tuple[str, int, int]:
        """Return (prompt_text, input_tokens, output_tokens) for Poisson mode."""
        if not self._initialized:
            self.initialize()
        in_t, out_t = (100, 150)
        if self._azure_records:
            rec = rng.choice(self._azure_records)
            in_t  = max(10, rec.context_tokens)
            out_t = max(10, rec.generated_tokens)
        if self._sgpt_records:
            prompt = rng.choice(self._sgpt_records).prompt
        else:
            from .sharegpt_prompts import ALL_PROMPTS
            prompt = rng.choice(ALL_PROMPTS)[0]
        return prompt, in_t, out_t

    def get_azure_arrival_rate(self) -> float:
        if not self._initialized:
            self.initialize()
        return self.azure.get_arrival_rate_rps()

    def has_real_azure_data(self) -> bool:
        return bool(self._azure_records)

    def has_real_sharegpt_data(self) -> bool:
        return self.sgpt.get_source() in ("cache", "huggingface")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_azure_timestamp(ts_str: str) -> float:
    """Parse Azure trace timestamp to Unix epoch seconds."""
    from datetime import datetime, timezone
    # Format: "2023-11-16 18:15:46.6805900"
    ts_str = ts_str.strip()
    for fmt in (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
    ):
        try:
            dt = datetime.strptime(ts_str[:26], fmt)
            return dt.replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    return 0.0


def _compute_inter_arrivals(timestamps: List[float]) -> List[float]:
    if len(timestamps) < 2:
        return []
    return [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)
            if timestamps[i + 1] > timestamps[i]]


def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _pct(vals: List[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    i = max(0, min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1)))))
    return s[i]
