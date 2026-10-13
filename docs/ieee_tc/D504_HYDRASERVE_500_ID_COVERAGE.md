# D504: all 500 existing logical adapters executed through native HydraServe

2026-10-13. Functional qualification only, not W0 performance or a baseline
optimum. The original scheduler, backend, placement, loading, idle retirement
and D503 settings are unchanged. Existing unrelated k3s was not accessed.

## Observation

| Evidence | Actual value |
|---|---:|
| Planned / arrived / terminal / native-contract responses | 500 / 500 / 500 / 500 |
| Failed responses | 0 |
| Distinct logical adapter IDs executed | 500 |
| Remote transfers, each matched to local request receipt | 500 |
| Wire bytes | 420,740,410 |
| Remote request packing / temporary archive creation | 0 / 0 |
| Observed container-reservation union, GPU-s | 4604.206954383 |
| GPU-s/offered request (not per-correct) | 9.208413909 |
| Mean / P95 TTFT, ms | 289289.305915 / 505736.984892 |
| Mean / P95 TPOT, ms/token | 60.719879 / 72.787144 |
| Mean / P95 E2E, ms | 298309.556198 / 514048.441662 |

All native lengths equal their targets. Canonical inputs, logical adapter/native
indices, native process/time-domain identity, timing decomposition and artifact
content SHA passed the independent postrun audit. All four actual GPU contexts
were observed and all native leases closed. `n_correct` and per-correct resource
remain null: identity/length checks are not an independent numerical reference.
500 logical IDs still mean the existing 498 zero / 2 nonzero adapters, four
weight SHA classes, not 500 independently trained models.

## What this experiment tests

The shared `adapter_coverage_v1` view maps sorted static IDs to the first500
existing trace prompts/arrivals/targets. Source trace stays immutable; the view
hash is `286f7547a8129fd6e75885379428b84c9a5c23a7eb02bf1d7546f49bcf2cee5d`.
It intentionally changes the adapter map, so it is neither an ordinary W0 prefix
nor a paired output comparison with D502/D503. No weights or replacement traces
were generated. Rank8:350; rank16:150, within unchanged native ranklimit16.
All tokenizer metadata has the source SHA already exercised in D503.

Observed mean submit-to-instance wait is 231210.710430 ms, assignment-to-first
native-output 58071.049064 ms and artifact resolution 4453.328894 ms. The last
is nested within the preceding path and must not be added again. These identify
where delay was observed but do not prove its causal mechanism. No baseline
performance configuration search, larger LoRA slot setting or latency repair was
introduced. All long waits are retained. Do not compare this diagnostic with
4000-request W0 points or infer main-workload SLO from its partial population.

Physical windows in GPU-s: pre-arrival0, arrival2239.345318356,
drain2227.887869176, cleanup136.973766851; sum4604.206954383. The D503 physical
boundary and pp1 bookkeeping caveat apply unchanged. One physical host/four pp1
instances does not qualify multi-node pipeline/consolidation or KV migration.

## Prerequisite sequencing deviations (preserved)

The standalone prelaunch health probe returned connection-refused immediately
after service start; the agent launched before incorporating that asynchronous
failure. That prelaunch check did **not** pass. Independent guarded-launch memory,
disk, ownership checks and the publisher's authenticated frozen remote manifest
handshake subsequently passed before deployment notice. No remote restart,
configuration change or health retry was performed during measured execution.

After actual local release, automatic remote collection stopped before SSH because
it expected the missing prelaunch health file. The first offline audit was also
started prematurely and stopped before creating outputs. These are orchestration
errors, not failed native generation. Original scripts/errors and tmux exit92
remain. A new post-release health receipt, explicitly not a prelaunch receipt,
identified the remote clock; the original start-Invocation checks, exact-owned
stop and SHA-verified downloads were then reused successfully. Every one of the
500 transfer IDs independently matched request-time local receipts. No rerun or
retroactive replacement of failed evidence was performed.

Future launches must explicitly await and verify all prerequisite results before
starting inference or analysis. The functional evidence above is retained with
these deviations, not silently promoted to formal performance qualification.

## Safety, provenance and next action

Actual service Invocation `827508296d1a42a28fa73b8c3147628e`, cgroup inode24089907
was removed. Auxiliary launch Invocation `404bb6498f9f4170b1b99b59dd0f44f2`,
inode24089865 was recursively empty and closed. Peak61,854,396,416B; all
high/max/OOM/OOM-kill/swap events zero under the common72/80GiB envelope.
GPU workers/provider/storage/pause were actually removed. Remote3B/7B/monitor
were exact-Invocation stopped AFTERlocalrelease; ports18080/18081 absent.
Diskfree187,416,596,480B (~174.5GiB); no broad cleanup or unique-asset deletion.
Prelaunch actual link1000Mb/s/full; the old100Mb observation's cause remains unknown.

Raw: `/home/qhq/serverless_llm_experiment/results/ieee_tc/hydraserve_qualification/d504_20261013`.
Curated source/request/physical/remote/error/closure evidence:
`paper_results/ieee_tc/hydraserve/d504/`. D503's88 CPU tests are reused for unchanged
source, with fresh backup smoke separately recorded. No credentials or weights
are published. Protected147 files and existing user edits are unchanged.

Next: original1000 compatibility and one complete4000 HydraServe point, with
the missing numerical qualification explicitly tracked; pipeline/KV migration/
consolidation remain unqualified rather than forced into a one-host result.
Then full available baseline comparison and PrimeLoRA optimization. No repeated
D504 key, no new adapter pool, no changing ranking thresholds. Full7B/3B,
W1/W2/G2, ablation, sensitivity and final paper-evidence scope remains active.
