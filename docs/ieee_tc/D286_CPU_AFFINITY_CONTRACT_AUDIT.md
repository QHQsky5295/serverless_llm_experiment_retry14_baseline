# D286 CPU-affinity contract audit

Date: 2026-10-06

## Finding

The Resident-v1 resource contract specifies service CPUs `4-23,28-47` and
auxiliary CPUs `2,3,26,27`.  A pre-run systemd check showed that the user
manager can display the slice-level `AllowedCPUs` request, but the user
service cgroup has only `memory,pids` delegated.  It therefore has no
`cpuset.cpus.effective` file, and the slice property alone is not evidence
that the worker processes are constrained.

The earlier D285 pilot recorded an empty `AllowedCPUs` field on each unit.  It
is retained as a valid pilot for its other protocol checks, but this resource
detail was insufficient for a formal reference run.

## Repair

The vLLM replica and replay transient units now receive systemd's
`CPUAffinity=` property at exec time.  The slice-level `AllowedCPUs` request
remains recorded for environments where cpuset delegation is available, but
formal evidence uses the kernel process mask read from `/proc/<pid>/status`
(`Cpus_allowed_list`) while each unit is active.  The receipt also records the
configured `CPUAffinity` and the enforcement method.

This is a resource-contract repair only.  It does not change the model,
trace, adapter subset, generation contract, request routing, or lifecycle
metric.

## Focused self-test

An isolated transient unit was launched with
`--property=CPUAffinity=4-23,28-47`.  `systemctl show` reported
`CPUAffinity=4-23 28-47`, and `taskset -pc` on its live PID reported
`4-23,28-47`.  The unit was stopped immediately after the check.  No GPU
process was present and no experiment service remained active.

The corresponding replay mask will be checked in the next protected Resident
pilot.  A formal run is admissible only when every active replica and the
replay unit reports the expected `CpusAllowedList`, with the existing memory,
swap, task, GPU-release, and generation-contract checks also passing.

## Provenance

- Baseline repository branch: `main`
- Previous pilot: D285 (100-request protocol pilot; not a formal budget)
- Changed files: `scripts/run_vllm_fair_experiment.sh`,
  `scripts/ieee_tc/resident_protocol_receipt.py`
- User dirty files were not staged.
