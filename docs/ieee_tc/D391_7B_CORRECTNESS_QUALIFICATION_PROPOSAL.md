# D391 — 7B numerical-correctness qualification proposal

Date: 2026-10-10. This document is a proposal for review. It launches no GPU
process, remote service, baseline, or figure and does not change the frozen
metric protocol.

## Decision boundary

The existing 7B evidence is sufficient for the native request and resource
contracts, but not for the frozen `N_correct` denominator:

| evidence | observation | admissible claim |
|---|---:|---|
| D384 native witness | 100/100 target-length/native-token records | request, prompt, adapter binding, and native accounting |
| D385 independent HF/PEFT witness | 92/100 exact; 8/8 first native tokens in the reference top-2 | bounded checkpoint/input consistency and near-tie warning |
| D295 same-backend repeats | cross-launch exact pairs 22/27, 24/27, 23/27 | output variation exists under the tested vLLM conditions |
| D379/D386 full replay | 4,000/4,000 terminal/native records | serving/timing/resource contract |

No row proves that every full-workload sequence is the intended adapter
computation. In particular, a length match, adapter tensor SHA, native slot
snapshot, or successful terminal state is a necessary condition, not a
semantic oracle. Therefore D379 and D386 remain development diagnostics and
`n_correct` remains unknown.

## Why the obvious shortcuts are rejected

1. **Promoting native-contract success to correctness.** This changes the
   frozen denominator by assertion and would make a resource number look like a
   correctness-normalized result.
2. **Treating top-2 near ties as exact.** D385 records strict mismatches. A
   numerical explanation is plausible, but the frozen ruler has no tolerance.
3. **Repeating the old 22-request probe.** D295 is the pre-registered repeat
   series and already shows cross-launch variation. Another screening repeat
   would add no pre-declared coverage.
4. **Enabling `VLLM_BATCH_INVARIANT=1` silently.** The installed vLLM 0.30
   environment documents this beta mode as requiring compute capability at
   least 9.0; the test GPUs are RTX 3090/SM86. The official documentation also
   scopes reproducibility to the same hardware and vLLM version. It is not a
   valid hidden switch for the current performance candidate.
5. **Using a same-backend replay as an independent oracle.** It can test
   execution stability or greedy self-consistency, but cannot by itself prove
   the intended checkpoint/adapter semantics.

## Lawful qualification routes

The existing ruler is strict. At least one of the following must be completed
before `N_correct=N` can be assigned to a full workload:

### Route Q1 — full independent exact reference (strict route)

Run the already frozen 7B trace and adapter/content manifest through an
independently implemented reference (HF/PEFT or an equivalent independently
audited implementation), using the same canonical prompt, tokenizer, target
length, and adapter bytes. Capture native token IDs for every request and
require exact sequence equality for all 4,000 requests. Retain first-divergence
position, logits/margins, adapter-content SHA, and resource/clock receipts for
every mismatch. Any mismatch leaves `N_correct` unknown; it is not relabelled
as a near tie. This is the only route in this proposal that directly satisfies
the current strict exact-sequence interpretation without changing the ruler.

The route is expensive and may fail because the existing 3090 FP16 vLLM and
HF/PEFT execution paths already disagree at near ties. That risk is a result,
not a reason to weaken the contract. A successful bounded witness cannot be
promoted to 4,000/4,000; the full denominator is required for G1/G2.

### Route Q2 — native teacher-forced arithmetic audit (supporting evidence)

For each captured native sequence, replay the exact prompt and adapter through
the same frozen backend in teacher-forced mode, record the selected-token
logit/top-1 decision at every position, and independently verify the adapter
transformation from the original safetensors. This can establish that the
sequence is greedy under the measured native implementation and that the
request did not silently use the base model or another adapter. It does not
replace Q1: same-backend arithmetic is not an independent semantic oracle.

Q2 is recommended as a prerequisite to Q1 because it localizes an incorrect
adapter path before an expensive full reference run. It must be run as a
separately identified diagnostic; its synchronization, logits capture, and
teacher-forcing work are not part of latency or GPU-s measurements.

### Route Q3 — deterministic hardware/backend qualification

On hardware that officially supports vLLM batch invariance, a separately
qualified candidate could use that mode and then repeat Q1 under the frozen
contract. The current RTX 3090 host is not such hardware, so this route is
out of scope for the present experiment and cannot justify an H100 or SM90
claim.

## Proposed execution order if the route is approved

1. CPU-only create a new qualification manifest binding the plan SHA, metric
   protocol SHA, backend/runtime hashes, frozen trace/subset/content hashes,
   generation contract, and the independent-reference implementation SHA.
2. Run a pre-registered bounded coverage check for every materialized content
   class and input-length group. This is a diagnostic stop condition, not a
   correctness denominator and not a performance sample.
3. If Q2 shows an adapter/source or teacher-forcing failure, stop and repair
   the implementation before any performance replay. If it passes, execute Q1
   over the complete frozen 7B trace, without changing routing, admission,
   cache, remote-delivery, timeout, or resource settings.
4. Analyze all 4,000 rows before any ranking. Only exact full coverage may set
   `N_correct=4,000`; otherwise retain `unknown` and report the qualification
   failure without running 3B or baselines under a correctness-normalized
   claim.
5. If and only if the correctness gate closes, perform the already approved
   common warm/SLO and Resident reference work, then resume the Prime-first
   optimization loop. Existing D379/D386 data remain preserved and are never
   relabelled.

## Required acceptance record

The qualification result must retain `N_plan`, `N_arrived`, `N_terminal`,
`N_correct`, exact/mismatch counts, per-request adapter and prompt identities,
native token IDs or lossless evidence sufficient to recheck them, independent
reference version and source SHA, first-divergence diagnostics, and complete
GPU/memory/remote cleanup receipts. A failed, incomplete, or externally
interfered run remains in the denominator audit and cannot be repaired by
deleting rows.

Until this proposal is reviewed and a route is executed successfully, no new
controller candidate, 3B replay, baseline comparison, ablation, sensitivity
experiment, or figure is authorized by the frozen plan. This is a qualification
boundary, not evidence that PrimeLoRA is numerically incorrect.

## Provenance

- Plan SHA-256: `0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7`
- Frozen metric protocol SHA-256:
  `5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22`
- D384: `docs/ieee_tc/D384_NATIVE_TOKEN_WITNESS_PROTOCOL.md`
- D385: `docs/ieee_tc/D385_PEFT_SEQUENCE_REFERENCE_RESULT.md`
- D390: `docs/ieee_tc/D390_7B_CORRECTNESS_BOUNDARY_DECISION.md`
- vLLM reproducibility and batch-invariance documentation:
  <https://docs.vllm.ai/en/v0.30.0/usage/reproducibility/> and
  <https://docs.vllm.ai/en/v0.30.0/features/batch_invariance/>
