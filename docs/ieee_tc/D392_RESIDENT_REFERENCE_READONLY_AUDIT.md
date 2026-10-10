# D392 — Resident-vLLM reference path read-only audit

Date: 2026-10-10. This audit runs no GPU process, remote service, replay, or
figure. It checks whether the existing ordinary-vLLM runner is ready for a
new protocol-compliant Resident reference after the correctness gate is
resolved.

## Findings

The baseline runner now contains a distinct `VLLM_RESIDENT_PROTOCOL_V1` path,
but this is implementation readiness, not a completed reference measurement.

| V1 requirement | Read-only observation | status |
|---|---|---|
| ordinary vLLM semantics | Resident path launches ordinary vLLM replicas and the replay client; it does not invoke `faaslora_full`, `ieee_confirmed`, Prime routing, handoff, hierarchy, or admission | passes semantic design audit |
| common 60-second notice | `resident_protocol_receipt.py notice` records a monotonic replay origin before model startup; replay receives the same clock ID and origin | ready for qualification |
| remote delivery | Resident mode requires `dynamic_remote` and a non-empty endpoint; the replay uses the approved prepublished gzip endpoint and per-run cold cache | ready for qualification; must verify endpoint/cache state at launch |
| independent service/auxiliary domains | systemd service and auxiliary slices are created before launch; vLLM replicas and replay are assigned to their respective slices | ready for qualification; effective worker limits must be recorded live |
| physical lifecycle | receipt records the selected GPU UUID inventory before startup, holds a per-GPU interval through service shutdown, and closes only after the final NVML census is empty | suitable for static four-GPU Resident accounting; live receipt still required |
| fixed output/native tokens | runner passes `fixed_length_greedy_v1`, vLLM native token mode, and strict replay validation | ready for qualification |
| cleanup | `EXIT` cleanup stops owned units, then finalizes the receipt; syntax and bytecode checks pass | ready for qualification |

The current source identities, retained without staging the dirty baseline
replay script, are:

```text
run_vllm_fair_experiment.sh       21cf3a2e280e5188331a9ef6a017cd35e8a11df6a72c4aa4a89cd9f0d40f30f6
resident_protocol_receipt.py      b13353c2f80182639e6bd3e1495bf4caa310ec74308af18fe16c5cd782b23fcf
replay_openai_trace.py            04f3eb52c64c6d684dec3ecef1e6a450d687e0922430c2d77932917e4a38fc22
baseline repository HEAD           3bd3da067e1843bd8367b9ff7ae77bb47ced5bca
```

The replay script is already user-modified in the baseline worktree. This
audit neither stages nor changes that file; its working-tree status must be
resolved and recorded by the baseline repository before a formal run.

## Remaining qualification work

The prior D213 runs are correctly rejected as Resident references because they
used local static adapters, shifted the business clock after readiness, shared
the replay/service resource scope, lacked a physical allocation ledger, and
used a 7,200-second timeout. The current scaffolding addresses those protocol
categories, but only a fresh, separately identified three-run series can
establish `U_ref` and the G2 budget. The series must first pass a CPU/launch
gate and a short qualification replay, then use the existing 1,800-second
protection and the same frozen trace/subset/generation contract as Prime.

This audit does not authorize that replay. Under D390/D391, the 7B numerical
correctness route remains the earlier gate; no Resident run, 3B run, baseline
ranking, ablation, sensitivity analysis, or figure is started from this audit.

## Checks performed

- `bash -n scripts/run_vllm_fair_experiment.sh` passed.
- `/usr/bin/python3 -m py_compile` passed for the Resident receipt helper and
  replay client.
- Source paths, endpoint mode checks, notice origin, resource-domain setup,
  native-token validation, and shutdown ordering were read directly.
- No result, trace, adapter, cache, remote service, or user-owned manifest was
  modified.

## Provenance

- Frozen metric protocol SHA-256:
  `5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22`
- D213 protocol rejection: `docs/ieee_tc/D213_RESIDENT_REFERENCE_PROTOCOL_AUDIT.md`
- D213 ordinary-vLLM semantic audit:
  `docs/ieee_tc/D213_VLLM_RESIDENT_SEMANTIC_AUDIT.md`
- D391 correctness qualification proposal:
  `docs/ieee_tc/D391_7B_CORRECTNESS_QUALIFICATION_PROPOSAL.md`
