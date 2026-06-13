#!/usr/bin/env python3
"""Materialize SLINFER's official GPU-only topology for the local 4x3090 testbed."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


MODEL_SPEC = {
    "3b": {
        "model_type": "llama-3.2-3b",
        "workers_per_gpu": 8,
        "hf_dir": "Llama-3.2-3B-Instruct",
    },
    "7b": {
        "model_type": "llama-2-7b",
        "workers_per_gpu": 4,
        "hf_dir": "Llama-2-7b-chat-hf",
    },
}


def _pool_config(model_type: str, workers_per_gpu: int, node_memory_gb: float) -> str:
    lines = [
        "from .models_info_template import models_info_template",
        "",
        "models_info = models_info_template",
        "",
        "pools_info_template = {",
        "    'gpu': {",
    ]
    for gpu in range(4):
        base_port = 8000 + gpu * 100
        lines.extend(
            [
                f"        {gpu}: {{",
                f"            'node_memory_capacity_GB': {node_memory_gb!r},",
                "            'node_ip': '127.0.0.1',",
                "            'gateway_ip': '127.0.0.1',",
                "            'dist_scheduler': False,",
                f"            'base_port': {base_port},",
                "            'node_label': 'gpu',",
                "            'workers': {",
            ]
        )
        for worker in range(workers_per_gpu):
            lines.append(f"                {worker}: models_info['{model_type}'],")
        lines.extend(["            },", "        },"])
    lines.extend(["    },", "    'cpu': {},", "}", ""])
    return "\n".join(lines)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--model-key", choices=sorted(MODEL_SPEC), required=True)
    parser.add_argument("--node-memory-gb", type=float, default=23.0)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    args = parser.parse_args()

    project = args.project_root.resolve()
    spec = MODEL_SPEC[args.model_key]
    template_dir = project / "SLINFER_core/scheduler/config_template"
    target = template_dir / "pools_info_template.py"
    source = template_dir / f"pools_info_template_{args.model_key.upper()}_0C4G.py"
    if not source.exists():
        raise RuntimeError(f"official SLINFER GPU-only template is missing: {source}")

    hf_path = project / "huggingface_models" / spec["hf_dir"]
    gpu_path = project / "gpu_models" / spec["hf_dir"]
    if not hf_path.exists():
        raise RuntimeError(f"SLINFER Hugging Face model path is missing: {hf_path}")
    if not gpu_path.exists():
        raise RuntimeError(f"SLINFER GPU model path is missing: {gpu_path}")
    hf_config = hf_path / "config.json"
    gpu_config = gpu_path / "config.json"
    if not hf_config.is_file() or not gpu_config.is_file():
        raise RuntimeError("SLINFER source or converted model config.json is missing")
    hf_config_sha256 = _sha256(hf_config)
    gpu_config_sha256 = _sha256(gpu_config)
    if hf_config_sha256 != gpu_config_sha256:
        raise RuntimeError(
            "SLINFER converted model config does not match the frozen source model"
        )

    rendered = _pool_config(
        spec["model_type"],
        spec["workers_per_gpu"],
        args.node_memory_gb,
    )
    target.write_text(rendered, encoding="utf-8")

    args.snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot = args.snapshot_dir / "pools_info_template.py"
    snapshot.write_text(rendered, encoding="utf-8")
    shutil.copy2(template_dir / "models_info_template.py", args.snapshot_dir)
    shutil.copy2(template_dir / "models_path_template.py", args.snapshot_dir)
    adaptation = {
        "schema": "slinfer_relayserve_hardware_adaptation_v1",
        "model_key": args.model_key,
        "model_type": spec["model_type"],
        "gpu_count": 4,
        "gpu_type": "RTX 3090 24GB",
        "configured_node_memory_capacity_gb": args.node_memory_gb,
        "workers_per_gpu": spec["workers_per_gpu"],
        "official_template_path": str(source),
        "materialized_template_path": str(target),
        "hf_model_path": str(hf_path.resolve()),
        "gpu_model_path": str(gpu_path.resolve()),
        "model_identity": {
            "source_model_realpath": str(hf_path.resolve()),
            "source_config_sha256": hf_config_sha256,
            "converted_config_sha256": gpu_config_sha256,
            "config_identity_verified": True,
        },
        "algorithm_changes": [],
        "hardware_adaptations": [
            (
                "node_memory_capacity_GB changed from the official A100-80GB "
                "template value 78 to the local RTX 3090 usable budget"
            ),
            (
                "generated pool template uses the package-relative "
                "models_info_template import required by the documented "
                "scheduler launch directory"
            ),
        ],
    }
    (args.snapshot_dir / "hardware_adaptation.json").write_text(
        json.dumps(adaptation, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[slinfer-prepare] model={args.model_key} "
        f"workers/gpu={spec['workers_per_gpu']} node_memory={args.node_memory_gb}GB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
