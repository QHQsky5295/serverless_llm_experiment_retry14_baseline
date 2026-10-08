# D366: native prefix after the D365 immutable-input repair

2026-10-09. Preparation for a 100-request native check, not a full-workload
qualification or a new tuning point.

D365 is backed up at `f7ccb13803cdcc3056c141aae6fe1cef371226bb`. Its unchanged-code
fixture reproduced the file-demand rejection on bounded allocation refresh;
77 targeted checks and 662 checks in each existing environment passed after the
repair. D364's complete failed run stays failed and its exact kernel event stays
unknown. No claim of GPU performance improvement follows from these checks.

Reuse D363's exact model/workload/control configuration and the D362 service and
preparation initialization. Only output, NVMe cache and HOST cache paths change.
The existing runner limits the same W0 trace to its first 100 requests and uses
the full IEEE scenario, not a synthetic replacement workload. No new weights,
trace, profile, capacity threshold, autoscaler setting or remote pool is created.

Before launch, verify D365's source/checksum/closure receipt, the actual child
model configuration, all 200 service and 24 preparation classes, source plan,
frozen V1, old protected files, and memory/disk/GPU safety gates. OS guards run
in system Python; native/analysis tests run in their existing qualified Python.
Seal and push this preparation before starting one tmux-backed guarded run.
Reuse the already published immutable remote archives and frozen services.

After all GPU contexts actually release: stop only the owned remote services,
retain matching transfer/monitor evidence, close exact-owned scopes once, then
audit all 100 offered records, native binding/token/timing, file-demand receipts,
physical GPU lifecycle and resource events. Failure is not erased by cleanup.
Any prefix statistics are single-run diagnostics; retain the D363 regression
and do not extrapolate resource cost to 4,000 requests or declare formal SLO.

If correctness/measurement checks pass, immediately proceed to the same-config
complete W0 replay and compare D358/D353/D336 under V1. No intermediate tuning
or new profile campaign is planned. Numerical correctness, final common warm
and Resident references, and formal G1/G2 remain open. Mainline remains 7B,
then 3B, then paused baselines and the retained comparison/ablation/sensitivity
matrix. No performance figures before optimization targets.
