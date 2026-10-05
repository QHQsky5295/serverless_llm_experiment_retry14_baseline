# D243 — 7B bounded native token-ID capture diagnostic

Date: 2026-10-06 (Asia/Shanghai)

## Purpose and scope

D243 is the first GPU execution step after the D228 CPU qualification.  Its
sole purpose is to obtain exact native completion-token sequences for a bounded
22-request 7B run so that the D225 numerical gate can compare PrimeLoRA with an
independent original-checkpoint PEFT reference.  It is not a performance point,
not a common-SLO result, and not evidence for a PrimeLoRA ranking.

The run reuses the frozen D227/D238 model, request, adapter-content, profile,
and published-delivery contracts.  It does not regenerate weights, LoRA files,
traces, prompts, or gzip objects.  The only semantic diagnostic switch is
`model.ieee_token_id_capture: true`; output, NVMe, shared-memory, and scenario
identities are fresh.  The ordinary digest field remains present alongside the
captured native integer IDs.

Configuration:

- path: `configs/ieee_tc/20261006_d243_7b_token_capture_22req.yaml`
- SHA-256: `b57acd9c6e2e61b20535afac697be6ad81fc39767cd672262221d91cd61cb213`
- guarded launcher: `scripts/ieee_tc/run_d243_7b_token_capture_22req.sh`
- launcher SHA-256: `e3966b7663f6d3c3016a232aad4b4d3606d99705a0fe06ac221d42794c515f7a`
- output: `/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d243_20261006/7b_token_capture_22req`
- request contract: 22 requests, seed 42, `fixed_length_greedy_v1`, target cap 256,
  temperature 0, top-p 1, concurrency 2, canonical `role_lines_v1` prompts
- backend: the already qualified pinned vLLM environment and TP=1 on the
  existing four-RTX-3090 host

## Delivery and timing boundary

The remote node uses the already published, immutable gzip cache from D78/D80;
there is no request-time archive creation or fallback.  The 3B and 7B published
manifests are complete (500/500 objects each), with archive bytes
1,160,474,851 and 420,740,410 respectively.  Publication, administration,
hash scans, and cleanup remain outside the request path.  A request-triggered
remote read/response, local receive/write/extract/verify, cache miss, and the
resulting inference lifecycle remain inside the measured request path.  This
diagnostic records those fields but does not promote them to a performance
comparison.

Before launch, read back the remote link, manifest SHA values, free bytes,
inodes, service state, and listeners through the pre-authorized strict SSH
alias.  Start only the owned 7B artifact unit on the published cache, and stop
it after the client and all cleanup checks finish.  Do not change the remote
link, create a second pool, or run a packing stress test.

## Gates before GPU launch

1. Re-run the inference-host preflight with zero predicted growth.  It must
   report `issues=[]`, available memory at least 109,521,666,048 bytes, free
   inference-host filesystem at least 161,061,273,600 bytes, and no active
   experiment GPU processes.  The observed preflight at preparation time was
   121,837,080,576 available bytes and 172,863,201,280 free bytes; it is
   rechecked immediately before launch rather than reused blindly.
2. Verify that the fresh output and `/dev/shm/tc-d243-7b-token-capture-22`
   paths do not exist, then let the guarded runner create them.  Do not remove
   or modify any historical D210/D219/D227/D238 result.
3. On the artifact node, independently confirm the D233 measured peak rule
   still passes: the approved 502-attempt/64-MiB-log/16-GiB-reserve bound is
   97,407,250,432 bytes and the current free space is above that bound.  The
   inference-host disk and memory policies remain unchanged.
4. Run the existing GPU-idle and cleanup checks, with one experiment process
   only.  If any gate fails, retain the receipt and stop; do not lower a
   threshold or select another host state.

## Correctness acceptance

The run is accepted only as a diagnostic if all of the following are recorded:

- 22/22 requests finish successfully and satisfy the fixed-output contract;
- every successful request has a native non-negative integer
  `completion_token_ids` sequence and the corresponding SHA digest;
- prompt hash, adapter identity, request ID, target-token map, arrival order,
  and canonical renderer match the frozen D166/D167/D207/D208 identities;
- dispatch/admission and physical-resource evidence are complete, all leases
  are closed, and the canonical physical-resource sidecar is bound to the
  exact result file;
- the captured IDs are compared, without text re-tokenization, with the
  independent original-checkpoint PEFT reference for nonzero-adapter, zero/
  base, and same-adapter-repeat controls;
- exact sequence equality is the primary criterion.  A mismatch is retained
  with the first divergent position and logit/reference evidence; it is never
  hidden by a tolerance or removed from the denominator.

Failure of a gate leaves D243 incomplete and blocks D226/D238 follow-up,
Full/Delayed readiness diagnostics, 3B, and every external baseline.  A passed
diagnostic still does not authorize a 4,000-request replay: the independent
reference and the physical sidecar must be audited first.

## Reproducibility record

The launch receipt must retain the exact config SHA, plan SHA, Python/vLLM
environment SHA, trace and adapter-subset SHA, content-manifest SHA, remote
published-manifest SHA, service invocation ID, result-file SHA, physical
resource sidecar, and GPU/NVML readbacks.  The status document must identify
whether the run is complete, failed, or held and must not describe a diagnostic
as a paper performance result.
