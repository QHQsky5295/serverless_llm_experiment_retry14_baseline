# D276: post-D275 7B readiness recheck

Date: 2026-10-06
Status: registered; not yet launched

## Purpose

D265 completed the eight-phase 7B diagnostic but supplied only ten natural
scale-out first-service observations. D266 was authorized as the identical cold
replicate, but stopped at request 198 with an uninstrumented native source
conflict. D275 now converts one demonstrated stale-selection race into the
controller's existing typed re-route while retaining fail-closed identity
checks. D276 is the one permitted same-configuration cold replicate to obtain
the missing first-service observations; it is not a new optimizer or a
performance comparison.

## Frozen identity

- Model: Llama-2-7B, existing vLLM 0.30.0 environment.
- Workload: existing seed-42 W0 trace, 4,000 requests, 500-adapter subset,
  eight 500-request phases, fixed-length greedy output contract.
- Prime configuration: byte-equivalent to D265 after replacing only the
  result/cache/run identity (`d265`/`retry7` → `d276`/`retry8`). A semantic YAML
  comparison was run after normalization and returned `normalized_equal=True`.
- Remote delivery: existing prepublished gzip cache; no repack, weight, trace,
  or second decompressed pool.
- Resource envelope: service 72/80 GiB with 2 GiB swap; auxiliary 3/4 GiB,
  swap disabled, CPUs 2,3,26,27; predicted growth 32 GiB.
- `formal_run=0`, `trace_role=exploratory`; the result cannot enter G1/G2 or
  ranking even if all requests complete.

## Entry gates checked

- D170 7B common-warm measurement is retained with 1,536 measured calls and
  all native timing/output contracts; its thresholds remain provisional until
  the separate Resident reference and cross-model work are complete.
- D236/D248 provide bounded native-token and independent PEFT evidence, with
  the cross-backend near-tie boundary retained rather than rewritten as exact
  equality.
- D265 physical allocation/release sidecar and cleanup are present; D276 will
  be rejected if its own sidecar does not prove the same lifecycle closure.
- Current host check before registration: approximately 112 GiB
  `MemAvailable`, 155 GiB filesystem free, four idle RTX 3090s, and no owned
  active experiment scope. These are launch-time observations, not guarantees;
  the launcher must repeat all gates.

## Acceptance boundary

The diagnostic needs 4,000/4,000 terminal success, complete dispatch-tier
coverage, zero tier invariant conflicts, at least eight natural scale-up
events, and at least twenty first-service observations. It remains descriptive
if the sample threshold is not reached, and any interruption or source
conflict is retained as failure evidence. The first-service observation count
is a readiness diagnostic only; it does not establish numerical adapter
correctness, common SLO qualification, G1/G2, or a baseline result.

The launch must start only after remote health and local cleanup are verified,
and must stop the remote artifact service after local physical release. A
failure does not authorize an OOM retry or a different configuration.

## Files and hashes

- Config: `configs/ieee_tc/20261006_d276_7b_readiness_8phase_full_retry8.yaml`
- Launcher: `scripts/ieee_tc/run_d276_7b_readiness_8phase_full_retry8.sh`
- Config SHA-256: `36880b280f881947ca2a3f68ba5ecb3808945116a9d42e9c4f61e0a6313b102a`
- Launcher SHA-256: `59178f2ca493e8d1a800e43115e3acadc64750074e33d83e4a27d365a57aa10f`
- Plan SHA-256: `0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7`
- Frozen V1 SHA-256: `5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22`
