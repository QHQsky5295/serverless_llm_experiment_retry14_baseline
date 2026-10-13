# D507: remove an isolation-adapter lifetime error; preserve native HydraServe

D506 is a sealed, incomplete `protocol_or_launcher_error`, not a HydraServe
performance result. Its six containers exited after approximately 1,800 seconds,
all with `OOMKilled=false`: our adapter wrongly passed `--timeout=1800` to Podman.
That option limits the whole container lifetime, not startup or an individual
request. The installed Podman 3.4.4 help and the
[matching official documentation](https://docs.podman.io/en/v3.4.4/markdown/podman-run.1.html#timeout-seconds)
confirm the distinction. The failure and partial population remain in D506.

## Necessary correction, not baseline performance optimization

Remove `timeout_s` from `ContainerRecipe` and remove the container `--timeout`
argument. Do not replace it with a larger arbitrary expiry. HydraServe's native
controller still owns worker retirement, including the unchanged 30-second idle
policy. The existing full-campaign bound, independent resource watchdog and
exact-owned cleanup handle abnormal termination. Podman infrastructure commands
remain bounded to 60 seconds. Request guards remain 1,800 seconds; native
transport/init deadlines, runtime model tuple `(13,9000,200,1000000,1)`, native
batch 8, cache policy, placement, startup overlap and evaluation SLO do not change.

Regression tests exercise creation commands for storage, provider and GPU-worker
roles: no whole-service TTL, unchanged resource limits and pre-model entry gate.
The removed keyword is rejected, so an old request guard cannot silently regain
service-lifetime semantics. Existing cancellation/recursive-cleanup tests remain.

## Evidence preservation and next actual experiment

Raw directory: `results/ieee_tc/hydraserve_qualification/d507_20261013` in the
original experiment repository. Before removal, preserve native stdout/stderr,
container State, IDs and ownership from all six stopped D506 containers. Remove
only those IDs after repeated ownership/stopped checks; do not delete weights,
adapter pools, traces, historical results or unrelated cluster resources.

D507 reuses the same full 4,000-request W0 view, 500-ID static registry, fixed
native generation contract, deployment notice, real-remote read-only objects,
resource envelope and common physical-GPU ledger. Only campaign identity and
the erroneous isolation-layer lifetime limit change. D503/D504/D505 passed
100/500-ID/1,000 qualification; do not repeat those GPU runs. This is one new
attempt after a demonstrated launcher error, not another chance to select a
favorable performance result. Native 9,000/200 ms control values are not the
common D394 evaluation thresholds. Raw `n_correct=null` and numerical limits
remain unchanged; only a complete audited run can produce a V1 observation.

Preserve all failures. Do not claim pipeline/consolidation or multiple compute
nodes if they do not occur on this one-host four-GPU testbed. The unrelated k3s
cluster is not accessed. After the actual full result, compare all available
baselines, then resume PrimeLoRA optimization; the remaining approved matrix
and both review rounds remain in scope. No figures at this pre-target stage.
