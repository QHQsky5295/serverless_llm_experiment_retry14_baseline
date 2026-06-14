# SLINFER RelayServe-Workload Reproduction Contract

Date: 2026-06-13

## Upstream identity

- Repository: `https://github.com/BarrinXu/SLINFER.git`
- Upstream commit: `483a276178efcea9c761deae802e82de6443b1`
- Scheduler mode: official `sota`
- Deployment mode: GPU-only
- GPU budget: four RTX 3090 GPUs

## Necessary compatibility adaptations

The reproduction preserves SLINFER scheduling, preemption, migration, model
lifecycle, and worker-count policies. The tracked compatibility patch only:

1. exposes the frozen paper TTFT/TPOT thresholds through the existing
   configuration endpoint;
2. passes exact frozen prompt token IDs instead of SLINFER's synthetic token
   table;
3. retains microsecond timing precision in the returned TTFT/TPOT metrics.

The patch is stored at `patches/slinfer_relayserve_compat.patch` and is applied
idempotently by `scripts/apply_slinfer_relayserve_patch.sh`.

The local hardware adaptation changes each official pool template's node
memory capacity from the A100-80GB value to 23 GB for RTX 3090. The upstream
logical worker counts are not host-memory safe on this shared 125 GiB server:
32 unloaded 3B vLLM processes plus the checkpoint store exhausted host
memory. The frozen topology therefore uses the number of physically possible
model slots plus one cold spare:

| Model | Physical slots/GPU | Workers/GPU | Total logical workers |
|---|---:|---:|---:|
| Llama-3.2 3B | 3 | 4 | 16 |
| Llama-2 7B | 1 | 2 | 8 |

This retains enough workers for every physically possible resident model plus
a cold replacement. No scheduling algorithm, GPU memory capacity, model
memory estimate, or power-model coefficient is changed. Results are labeled
as a hardware-adapted reproduction rather than an exact A100 topology replay.

The materialized pool template also uses the package-relative
`from .models_info_template ...` import. The upstream GPU-only templates use
an unqualified import that fails when `gateway.py` is launched from the
documented `scheduler/` directory, while upstream debug templates already use
the package-relative form.

## Host-memory safety adaptation

The upstream launch recipe reserves a 64 GB pinned checkpoint-store pool.
That setting caused a host-level OOM on this 125 GiB shared testbed on
2026-06-13, so those failed smoke runs are excluded from all reported data.

The frozen local launch uses a 24 GB store pool. This is larger than the
18 GB converted 7B checkpoint and does not alter SLINFER's scheduler, model
placement decisions, or GPU execution. A continuous guard records
`MemAvailable`, total memory, and free swap every two seconds. It terminates
the isolated experiment if available host memory falls below 32 GB. The
memory trace and thresholds are included in every run manifest.

The environment follows the upstream README's `pyairports` instruction:
the incorrect PyPI placeholder is replaced by
`ozeliger/pyairports@f611ee5a5a82b4e98b22641bb99693d862c802e4`.

## Model and workload identity

- Llama-3.2 3B uses the same
  `LLM-Research--Llama-3.2-3B-Instruct` model as RelayServe.
- Llama-2 7B uses the same `meta-llama--Llama-2-7b-hf` model as RelayServe.
- The preparation step requires the source and converted `config.json` files
  to have identical SHA-256 hashes.
- Calibration uses the disjoint 512-request chronological trace.
- Formal evaluation uses the frozen 4000-request chronological trace.
- Both use rate 1.00x and paper-nominal TTFT/TPOT targets.

## Calibration policy

The only selected system parameter is SLINFER's official keep-alive time.
Candidates are `1`, `10`, `30`, and `60` seconds for both models. The winner
minimizes the worst normalized P95 SLO ratio, then maximizes joint attainment
and CE, then prefers the shorter keep-alive.

Formal4000 data cannot select or alter the candidate.

## Evidence and cost

The replay records every request, exact prompt/output token counts,
scheduled-arrival TTFT, service TTFT, TPOT, E2E, and the official gateway
monitor output. The monitor uses SLINFER's one-second cadence.

Lifecycle cost charges startup and active GPU-seconds at full price and
ready-idle GPU-seconds at the frozen serverless idle factor. Monitoring
continues for `keep_alive + 2` seconds after the final request so the selected
retention policy is represented in cost.
