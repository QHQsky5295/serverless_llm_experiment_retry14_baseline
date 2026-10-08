# D343: bounded7B native capacity/TPOT probe

2026-10-08, P2/P3 development. Full Plan/V1 read; no baseline or figure work.
This follows completed D342, not another dispatch analysis or warm calibration.

## Hypothesis and controls

D342 found a real local scarcity window: at756.640562s,8 definitely held permits,
ready capacity8,37 gate waiters and3 pending activations. D341's long acquire wait
mostly overlaps that startup window. This does not prove unused compute capacity,
nor justify free warm GPUs. D169 established C8 for the prescribed warm batch;
it did not optimize sustainable concurrency. Native D341 startup logs independently
report KV capacity; its compact requests do not retain a live free-block series.

Hypothesis: increasing real native sequence/adapter capacity permits more useful
overlap during low-replica intervals. Counter-risk: TPOT, per-request TTFT, memory,
startup or closed-loop GPU-s can worsen. More accepted calls alone is not success.
Only max_num_seqs/runtime_concurrency_cap/max_loras change8→15; FP16, rank16,
TP1,0.92 memory fraction,1024 context/iteration budget, prefix-off, HOST allowance,
real published remote, original generation and paper equations remain unchanged.
The base is D336/D298 native behavior, NOT the unaccepted D341 on-demand import
candidate. No preload/scale/autoscaler change is bundled into this native probe.

Candidate derivation: D169 minimum measured free1022 KV blocks,16tokens/block;
ceil(1024/16)=64blocks for a full context; floor(1022/64)=15. This is an initial
candidate bound, NOT a claim that the larger runtime retains1022free blocks.
After its LoRA buffers/graphs initialize, every probe wave requires a drained
native pool with at least15×64=960 actually free blocks. Any failure is retained;
no automatic batch shrink, memory increase, fallback, or OOM retry.

## Version-specific basis

- [vLLM0.30 optimizer documentation](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/docs/configuration/optimization.md):
  sequence/token capacity and KV preemption have distinct throughput/latency
  trade-offs. Do not import datacenter batch defaults or reopen the failed512
  iteration-budget candidate.
- [same-version scheduler](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/core/sched/scheduler.py):
  native max sequences and distinct LoRA limit both constrain scheduling;
  raising only the outer gate moves queues without increasing native capacity.
- [LoRA configuration](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/config/lora.py):
  max_loras is a positive integer, not restricted to powers of two; changing it
  changes buffers/compilation identity. No timing-profile relabelling is valid.

## Minimal check and acceptance boundary

Reuse D298 native_source_matrix and append-only launcher mode d343-probe.
Select15distinct existing adapters from D168's recorded request index within
the first1000 development requests, longest declared prompt+target first with
original-index tie break. These are stress cases, not a representative workload.
One15-call warmup, then three alternating8/15 pairs:84calls,69measured.
All source preparation is real and recorded. No new weights/trace or pool copy.

Use explicit fixed-output **completed-length-only** transfer for the coupled
capacity change. It validates old native completions and preserves measured
length priors; no latency, correctness qualification or frozen SLO moves with it.
The old iteration-budget transfer contract retains its original restricted type.

Report native counts, references, fresh KV, actual time overlap/peak concurrency,
native TTFT/TPOT, wave makespan/token throughput, release and resource events.
All84 must satisfy token/reference/terminal contracts; numerical N_correct remains
separately unqualified. Same-runtime8/15 wave differences are diagnostic, not a
causal old-C8/new-C15 full comparison. Warm reference stays batch8 and unchanged.

If safe/useful, refresh affected service/preparation classes on this exact runtime,
then existing100-request Full smoke and original4000W0. Do not feed oldC8 timing
profiles to C15 or claim GPU-s savings from a native-only probe. One hypothesis,
at most two minimal checks if new evidence warrants. Full acceptance still uses
V1 correctness/common SLO/physical lifecycle, not solely warm latency.

Prepared source_spec/prepare.json retain selected indexes, actual child model,
native startup log references, SHA chain, old-result protection and candidate
derivation.27initial CPU tests passed; further regression outcome recorded at
checkpoint. No GPU measurement has occurred at this document's preparation.

Expanded regression:371tests passed (43.046s), covering existing source setup,
reservation/retirement, capacity/length binding and unchanged warm reference.
The first broad CPU driver lacked a multiprocessing main guard; spawned test
children re-entered its completed cleanup, so that CPU attempt was interrupted
and retained. The spawn-safe second driver passes without serving-code changes.
This is a test-launch mistake, not a failed GPU capacity candidate.

## First native outcome: C15 rejected, one evidence-driven followup

The larger native runtime initialized, but its drained pool had942free blocks,
not the earlier C8's1022. Full15-context capacity needs960. No inference request
was submitted.15setup downloads transferred39,670,961wire bytes, no packing.
Startup86.594s and physical allocation101.518437GPU-s remain recorded; they
are not a successful workload result. Service peak5,450,616,832B, hostminimum
107,138,461,696B, high/max/OOM/swap0. Actual native release, owned local
cleanup and three remote services stopped; source/journal bytes/SHA matched.

The new diagnostic guard initially compared native unretired_iterations to[],
but this native API returns an integer count0. Its schema check was corrected
and regression-tested against the retained native snapshot. Corrected guard
still rejects C15 with explicit942<960. Do not call this a pure capacity-only
execution failure or hide the diagnostic error; both facts are retained.

One second minimal check is justified by the fresh measurement:
floor(942/64)=14. It has a separate cap14 configuration/spec/output identity,
not an in-run fallback or overwritten C15 result. It must freshly pass896free
blocks before any inference. The same selection rule yields14warmup plus
three8/14pairs=80calls. Only old completed fixed-output lengths transfer;
no old latency profile or SLO is relabelled.40follow-up CPU tests passed,
including the real integer-zero schema and insufficient-C15 regression.
No additional minimal-capacity sweep is authorized by this hypothesis.
