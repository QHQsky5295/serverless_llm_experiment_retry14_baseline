"""
Helpers for scalable LoRA adapter manifests.

The original project hard-coded six adapters in YAML.  This helper generates
deterministic adapter metadata for larger-scale experiments (100-1000+) while
keeping a simple manifest format that scripts can share.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence


DOMAIN_TEMPLATES: Sequence[Dict[str, Any]] = (
    {"domain": "finance", "task_type": "finance", "size_mb": 32, "lora_rank": 8},
    {"domain": "medical", "task_type": "medical", "size_mb": 28, "lora_rank": 8},
    {"domain": "code", "task_type": "code", "size_mb": 45, "lora_rank": 16},
    {"domain": "support", "task_type": "support", "size_mb": 20, "lora_rank": 8},
    {"domain": "legal", "task_type": "legal", "size_mb": 35, "lora_rank": 8},
    {"domain": "translate", "task_type": "general", "size_mb": 24, "lora_rank": 4},
    {"domain": "education", "task_type": "education", "size_mb": 26, "lora_rank": 8},
    {"domain": "research", "task_type": "research", "size_mb": 30, "lora_rank": 8},
    {"domain": "ecommerce", "task_type": "ecommerce", "size_mb": 22, "lora_rank": 8},
    {"domain": "gaming", "task_type": "gaming", "size_mb": 18, "lora_rank": 4},
    {"domain": "writing", "task_type": "writing", "size_mb": 27, "lora_rank": 8},
    {"domain": "security", "task_type": "security", "size_mb": 34, "lora_rank": 8},
)

LEGACY_IDS: Sequence[str] = (
    "finance_lora",
    "medical_lora",
    "code_lora",
    "support_lora",
    "legal_lora",
    "translate_lora",
)


def _tiered_hotness(index: int, total: int) -> float:
    """Map a rank index to a stable [0, 1] hotness score.

    The thresholds are chosen so existing preload thresholds (0.3 / 0.4 / 0.9)
    still produce a manageable hot set when the adapter count grows to 1000.
    """
    if total <= 0:
        return 0.05
    frac = float(index + 1) / float(total)
    if frac <= 0.005:
        return 0.95
    if frac <= 0.02:
        return 0.75
    if frac <= 0.05:
        return 0.55
    if frac <= 0.10:
        return 0.35
    if frac <= 0.20:
        return 0.20
    return 0.05


def _adapter_id_for(index: int, domain: str, preserve_legacy_names: bool) -> str:
    if preserve_legacy_names and index < len(LEGACY_IDS):
        return str(LEGACY_IDS[index])
    return f"{domain}_lora_{index + 1:04d}"


def build_adapter_entries(
    num_adapters: int,
    *,
    preserve_legacy_names: bool = True,
) -> List[Dict[str, Any]]:
    if num_adapters < 1:
        raise ValueError("num_adapters must be >= 1")

    entries: List[Dict[str, Any]] = []
    templates = list(DOMAIN_TEMPLATES)
    for idx in range(num_adapters):
        tmpl = templates[idx % len(templates)]
        size_mb = float(tmpl["size_mb"])
        entries.append(
            {
                "id": _adapter_id_for(idx, str(tmpl["domain"]), preserve_legacy_names),
                "domain": str(tmpl["domain"]),
                "task_type": str(tmpl["task_type"]),
                "size_mb": size_mb,
                "size_hint": f"{size_mb:.0f} MB",
                "hotness": _tiered_hotness(idx, num_adapters),
                "lora_rank": int(tmpl["lora_rank"]),
                "manifest_index": idx,
            }
        )
    return entries


def build_adapter_manifest(
    num_adapters: int,
    *,
    model_name: str = "",
    preserve_legacy_names: bool = True,
) -> Dict[str, Any]:
    adapters = build_adapter_entries(
        num_adapters, preserve_legacy_names=preserve_legacy_names
    )
    return {
        "version": 1,
        "created_at": int(time.time()),
        "model_name": model_name,
        "num_adapters": num_adapters,
        "preserve_legacy_names": preserve_legacy_names,
        "adapters": adapters,
    }


def write_adapter_manifest(manifest: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_adapter_manifest(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    adapters = data.get("adapters")
    if not isinstance(adapters, list) or not adapters:
        raise ValueError(f"invalid adapter manifest: {path}")
    return data


def select_adapter_entries(manifest: Dict[str, Any], count: int) -> List[Dict[str, Any]]:
    adapters = list(manifest.get("adapters", []))
    if count < 1:
        raise ValueError("adapter selection count must be >= 1")
    if count > len(adapters):
        raise ValueError(
            f"requested {count} adapters, but manifest only contains {len(adapters)}"
        )
    return adapters[:count]
