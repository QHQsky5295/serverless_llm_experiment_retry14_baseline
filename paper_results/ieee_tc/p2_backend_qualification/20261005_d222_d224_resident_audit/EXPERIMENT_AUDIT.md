# D222–D224 Resident-vLLM experiment-integrity audit

Date: 2026-10-05  
Audit class: read-only, same-family provisional review  
Scope: the D222–D224 Resident-vLLM prototype attempts and their retained raw
receipts/results

## Overall verdict: FAIL for Resident/G1/G2; retain diagnostic evidence only

The attempts are not eligible for a Resident baseline, lifecycle GPU-s/request,
joint-SLO qualification, G1/G2 selection, or cross-system ranking.  The raw
files remain immutable in the baseline repository.  The Resident entrypoint is
now fail-closed.

## Evidence checks

### Backend and resource contract — FAIL

The launch path resolves the historical `LLM_vllm0102` Python environment and
does not record a vLLM package or source identity.  The D224 receipt records a
96-GiB service maximum, no CPU affinity, and task limits that do not match the
frozen 72/80-GiB service contract.  Its release snapshots show inactive units
with `MemoryMax=infinity`, `TasksMax=153966`, and an empty slice.  Claimed
arguments are not observed effective limits.

### Common deployment clock — FAIL

The receipt does encode a planned 60-second lead, but the notice is published
only after all replicas are ready.  It records no observed replay-start event;
the replay service establishes a separate origin after tokenizer preparation.
The recorded lead therefore proves scheduling intent, not the frozen
deployment-notice timeline.

### Service/auxiliary accounting — FAIL

Replay and remote materialization execute in the auxiliary path in the
prototype, excluding service work that belongs in the serving budget.  Effective
cgroup and CPU settings are not captured as an observed, pre-launch contract.

### Physical GPU lifecycle — FAIL

`resident_protocol_receipt.py init` labels an initial `nvidia-smi` inventory as
acquisition; no owner lock or physical lease proves exclusivity.  `finalize`
closes every GPU interval from a final memory-only census.  The resulting
intervals are therefore not an auditable physical owner-union measurement.

### Remote delivery identity — FAIL

The receipt stores endpoint and mode claims but no joined health/version/content
digest.  Readiness accepts an HTTP response without validating its body.  D224
records remote bytes and endpoint fields but no artifact content identity.  The
launch YAMLs also select `dynamic_remote`; the repository's remote-fair rules
require round-local `static_remote` staging for the ordinary formal vLLM path
unless a separately resource-bounded exception is qualified.  That exception
was not qualified here.

### Output and adapter correctness — WARN (limited PASS for generation contract)

D224 has 100/100 successful records with the frozen output-token contract,
native vLLM token IDs, and prompt hashes.  This validates only token-source and
length accounting for a 100-request prefix.  It does not prove numerical
adapter correctness or a full 4,000-request workload.  D222/D223 have no
usable replay result.

### Completion/status integrity — WARN/FAIL

The replay path performs some post-checks, but receipt finalization does not
require a replay artifact, a successful notice, or a complete planned
population.  A released receipt can therefore coexist with a missing or
failed replay.  Systemd `--collect` status is not sufficient evidence.

## Classification

| Attempt | Evidence | Classification | Eligible result |
|---|---|---|---|
| D222 | syntax/CPU/dry-run receipt path; no formal replay | protocol/launcher diagnostic | none |
| D223 | startup/replay protocol failure; no usable replay | protocol/launcher diagnostic | none |
| D224 | 100/100 fixed-output records, `max_requests=100` of a 4,000-request workload | generation-contract smoke | diagnostic only |

## Required before any future Resident run

1. Reuse the qualified common preflight and resource-domain implementation,
   with limits applied and verified before worker startup.
2. Use the frozen vLLM 0.30.0 environment and record package/source identity.
3. Publish the common notice before deployment work and derive replay time from
   that clock without a second origin.
4. Establish a real, owner-bound physical GPU lease and release journal with
   overlap-union validation.
5. Join remote health, cache/content SHA, transfer ID, and request records.
6. Keep serving-path remote materialization in the service budget.
7. Require complete planned population and numerical adapter correctness before
   any G1/G2 or ranking claim.

No GPU rerun is authorized by this audit.
