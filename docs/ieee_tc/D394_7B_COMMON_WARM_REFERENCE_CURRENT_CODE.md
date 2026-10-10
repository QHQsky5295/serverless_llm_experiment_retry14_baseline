# D394 — 7B common warm reference on current code

Status: COMPLETE, 2026-10-10. This is a calibration/qualification artifact,
not a Prime Full result and not a G1/G2 ranking.

## What was executed

The sealed D168 input index and D169 batch-8 candidate were reused. No weights,
trace, adapter pool, remote service, or request workload was regenerated. The
only material difference from D170 is the serving/preflight source: D394 uses
the current branch code and records its source hash; D170 used the earlier
preflight/replay source. Remote artifact delivery was explicitly disabled for
this GPU-ready native reference, as required by the reference protocol.

The run contained 16 warmup requests and 1,536 measured requests: two native
input-length groups, 256 requests per group per round, and three rounds. Every
measured request passed the native timing contract, had `actual_tokens` equal to
the target, and used the same monotonic clock domain.

## Current-code measurements

| Native input group | Round 0 TTFT / TPOT (ms) | Round 1 TTFT / TPOT (ms) | Round 2 TTFT / TPOT (ms) | Equal-weight mean (ms) |
|---|---:|---:|---:|---:|
| `<=616` | 600.233 / 29.857 | 598.182 / 29.801 | 598.793 / 29.958 | **599.070 / 29.872** |
| `>616` | 926.578 / 42.693 | 925.408 / 42.633 | 928.022 / 42.545 | **926.669 / 42.623** |

Applying the frozen 5x TTFT and 2x TPOT multipliers gives the current-code
candidate thresholds:

| Group | TTFT threshold (ms) | TPOT threshold (ms) |
|---|---:|---:|
| `<=616` | 2995.349 | 59.744 |
| `>616` | 4633.345 | 85.247 |

These values are not yet the global common reference: the V1 protocol still
requires the cross-model qualification and the separately frozen Resident
resource reference. They replace neither D170 nor any historical result.

## Contract and resource audit

- 1,536/1,536 measured requests passed `ieee_tc_native_v1`.
- 1,536/1,536 native output counts matched the sealed targets.
- 1552/1552 requests (including warmup) were released by the native reference
  path; the result is marked `calibration_candidate_complete=true`.
- 194 batch records were emitted at batch size 8.
- The run used one physical GPU, FP16, TP=1, `max_num_seqs=8`, `max_loras=8`,
  rank cap 16, GPU memory fraction 0.92, and prefix caching disabled.
- The independent watchdog recorded 1,417 resource samples (1,420 total events), a service memory
  peak of 7,569,285,120 bytes, minimum host available memory of
  107,259,219,968 bytes, no memory-high/OOM/swap events, and no abort reason.
- Service and watchdog exit codes were zero. The owned lease was released;
  final GPU census showed no service compute process and clear service CUDA
  context. The raw field `proves_physical_lease_release=false` is unconditional
  in the census producer: an empty census alone is not a lease-release receipt.
  Release is established separately by the owned allocation journal, worker
  exit and clear-context census. The false field is retained; it is not caused
  by the stable desktop graphics process.

## Reuse decision

D394 is the current-code 7B warm-reference candidate. D170 remains immutable
historical evidence. Both groups' TTFT and TPOT three-round means differ by
less than 0.2%, but this
closeness is descriptive, not a license to merge source identities. D394 is
eligible for the next 7B Resident-reference and formal-Prime preparation step;
it does not close numerical adapter identity, full-pool semantic qualification,
or G1/G2 by itself.

## Immutable evidence

Raw files are outside Git under:

`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d394_20261010/`

| Artifact | SHA-256 |
|---|---|
| `7b_measure1.json` | `86321c3c39e70e86c9e2ab81a2e71173ebf6024e04a4e76b05b5c7f6738c9f2e` |
| `7b_measure1_launch.json` | `59e320e387ca5e5610eb7652ac39dfddb0aac6e637b3dee4626bab74cffeedf6` |
| `measure1_sources.sha256` | `5b6153420c35529e81fc70c0f8223f6a0e7cce7c0f3d241de6d759a19fccffd7` |
| physical allocation journal | `cb9c26225531e85b4e31c7cdb165839f3fa2f4592fa1af8f4acc8b2b7183e575` |

After curation, the 707,508,991-byte raw JSON was losslessly compressed to
`7b_measure1.json.gz` (17,526,550 bytes). The archive SHA is
`99a4083f2945f8f2e2c8fe48e7a9382f24d9f2cf177bef2a29fb489459bb047f`.
Decompression reproduced the raw SHA above before the uncompressed duplicate
was removed. The raw SHA therefore identifies decompressed evidence content.
The independent streaming/native audit passed all 1,552 request checks,
194 batch checks, release-journal checks and resource checks; its curated JSON
is `paper_results/ieee_tc/p2_backend/20261010_d394_7b_warm_independent_audit_v2.json`
(SHA-256
`3cb7c9ebdc4340b03311b8942022181f0f4febb848691b006408c1aad02f9eb5`). The two
focused native-token/identity test modules passed all 23 tests in the vLLM 0.30
environment; an earlier system-Python invocation lacked vLLM and is recorded
as an environment-only test failure, not an experiment failure. The protected
artifact manifest still carries the approved pre-existing baseline edit; its
entry count is 147 and the changed file's recorded SHA matches the approved
value, so it does not alter the Prime evidence or final_v2/figs/paper tree.

The plan SHA was
`0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7`; the
warm-index SHA was
`563abfbdeb74c1bc03e42a26c34d48d634c342ac4fa43bd00c5ff8dd82269793`.
