# D287 — Resident-vLLM 7B CPU-affinity pilot

Date: 2026-10-06

This is a 100-request protocol pilot after the D286 resource-contract audit;
it is not a formal Resident reference repeat and is not a G1/G2 ranking point.

## Qualification table

| Check | Observation | Result |
|---|---:|---|
| Planned trace / pilot requests | 4,000 / 100 | diagnostic prefix, as registered |
| Successful requests | 100 / 100 | pass |
| Fixed-output target matches | 100 / 100 | pass |
| Native completion-token source | 100 / 100 `vllm_token_ids` | pass |
| Prompt/target request-map SHA | `a83e56ea59d72e5f2c01bb0566a48b55450cbcaf77135933c2fa6cfb96151974` | pass |
| Replica startup (s) | 72.5846–73.2818; max 73.2818 | recorded |
| Mean TTFT / P95 TTFT (ms) | 760.11 / 875.83 | descriptive only |
| Mean TPOT / mean E2E (ms) | 25.14 / 5104.7 | descriptive only |
| Lifecycle GPU-seconds | 2,254.12 | descriptive only |
| Final NVML experiment processes | none | pass |
| Remote delivery | `prepublished_gzip_v1`, `artifact_timing_v2` | pass |

## Active resource and affinity evidence

At both `replicas_launched` and `replay_started`, every vLLM unit reported
`MemoryHigh=72 GiB`, `MemoryMax=80 GiB`, `MemorySwapMax=2 GiB`, and
`TasksMax=4096`.  The replay unit reported a separate 4-GiB high/max,
zero-swap, 128-task envelope.

The active snapshots now include the enforceable process mask:

- every vLLM main PID: configured `CPUAffinity=4-23 28-47`, kernel
  `Cpus_allowed_list=4-23,28-47`;
- replay main PID: configured `CPUAffinity=2-3 26-27`, kernel
  `Cpus_allowed_list=2-3,26-27`.

The slice-level `AllowedCPUs` field remains empty because this user service
does not have delegated cpuset.  D286 therefore changed the contract to use
systemd transient-unit `CPUAffinity` and to record the process mask, rather
than treating `AllowedCPUs` as proof.  No request path, model, trace, subset,
generation contract, or lifecycle accounting changed.

## Resource safety

The preflight passed with a 32-GiB predicted growth check and the existing
150-GiB disk floor.  The minimum displayed host `MemAvailable` after replay
was 101.87 GiB; no cgroup high/max/OOM event was observed in the retained
watch, and all local GPU processes and the remote artifact unit were stopped.

## Provenance

- baseline branch: `main`
- CPU contract repair: commit `1115b16`
- result root: `results/ieee_tc_resident_protocol/d287_7b_cpu_affinity_pilot_seed42/`
- receipt: `d287_7b_cpu_affinity_pilot_seed42_resident_protocol/d287_7b_cpu_affinity_pilot_seed42_lifecycle.json`
- fair runner SHA-256: `21cf3a2e280e5188331a9ef6a017cd35e8a11df6a72c4aa4a89cd9f0d40f30f6`
- receipt helper SHA-256: `b13353c2f80182639e6bd3e1495bf4caa310ec74308af18fe16c5cd782b23fcf`

The pilot qualifies the process-affinity and cleanup path.  The next
admissible action is the pre-registered three-repeat, 4,000-request 7B
Resident reference series; its mean lifecycle GPU-seconds, not this pilot,
will define the G2 budget reference.
