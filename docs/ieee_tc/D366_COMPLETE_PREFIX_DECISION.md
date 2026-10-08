# D366: complete native prefix after the immutable-file repair

2026-10-09. Execution `0da5917af192a8e9055ac9c0651462c427e00355`.
Source SHA `0788a2b972887baafffc87cec7a501f8e078298eb294b7497bedcb386fd41905`,
6,097,205 bytes. All 100 offered requests completed without fallback; native
fixed-output, input/adapter binding, timing and ownership checks passed. This
does not establish numerical `N_correct` or formal G1/G2 qualification.

| Same-config prefix observation | D363 | D366 |
|---|---:|---:|
| Completed / failed | 100 / 0 | 100 / 0 |
| Mean TTFT, ms | 909.898 | 694.859 |
| P95 TTFT, ms | 2,441.887 | 1,538.732 |
| Mean TPOT, ms | 26.234 | 29.218 |
| P95 TPOT, ms | 31.492 | 40.760 |
| Physical GPU-s | 803.253 | 335.653 |

Mean/P95 TTFT decreased by 23.63%/36.99% and measured GPU occupation by 58.21%.
Mean/P95 TPOT increased by 11.38%/29.43%; that regression remains in the result.
Both use the same D362 initialization and all controller/model configuration;
only the D365 identity correction and output/cache paths differ. Natural closed
loop execution still varies: this run has two native owners versus four in
D363. A single short run is not a causal latency estimate or confidence interval
and is not extrapolated to 4,000 requests. D364's full-run failure stays failed.

Unchanged provisional D170 bounds give 99/100 joint timing passes, one TTFT miss
and no TPOT miss. These are timing-only bounds, not final common SLO acceptance.
The pending common warm, numerical correctness and Resident reference gates
cannot be replaced by this value.

The strict file-demand audit covers 33 file transactions (29 Remote, 4 NVMe),
58 native-GPU and 9 native-HOST acquisitions. Eight file transactions observe
intervening worker-epoch progress (maximum five); live capacity is rechecked,
not borrowed from the earlier snapshot. Control receipts cover all 100 requests
and two owners. This verifies exercised mechanisms, not avoided RPC counts or
the exact filesystem event behind D364.

Remote journals match 29 transfers / 50,650,865 wire bytes, no request-time
packing, actual 1000 Mbps/full. All 262 resource samples are non-aborted;
service peak 11,054,030,848 bytes, minimum host available 101,917,089,792 bytes,
and high/max/OOM/swap zero. Actual native GPU release and service-domain removal
passed; tmux ended and the three owned remote units were stopped once. After
release, 69 regression checks and complete native/binding/remote/timing/resource
audits passed. No figures were produced.

## Retained setup failures and reuse

The first preparation passed all 69 native checks, 50 OS guards and 200/24 profile
class bindings, then failed the final Git whitespace check on one additional
YAML EOF newline. The original archive/manifests remain unchanged. Removing
only that byte is verified against the first SHA; tests were reused, configuration
assembly rechecked, and `preparation_final.*` is the actual launch contract.

The first starter closed preparation and started the correct remote services,
then its health-client import failed in system Python (missing numpy), before
tmux or GPU requests. The resume verified the same remote InvocationIDs and
used the existing qualified model Python for health. It did not restart services,
change dependencies, regenerate artifacts or repeat preparation cleanup. Both
attempts are retained. All serving measurements belong to the one launched run.

## Decision and mainline

Proceed, without additional tuning, to a fresh complete 4,000-request W0 replay
of this same candidate/configuration. Compare retained D358/D353/D336 with the
frozen V1 and retain every failure/regression. The short prefix does not include
the late failure site or all phase-two contention targeted by this candidate.
There is no claim that 7B optimization is complete. Continue 7B → 3B → paused
baselines → M1/M2 → ablations/sensitivities, without new figures or asset pools.
