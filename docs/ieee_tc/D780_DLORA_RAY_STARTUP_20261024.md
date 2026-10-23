# D780: dLoRA private Ray startup adaptation

The actual 3B DP2/TP1 qualification reached private Ray initialization and native
model loading after preparing all 500 unchanged remote adapters. It did not
reach a native ready receipt or complete inference. This is an incomplete
diagnostic, not a valid latency, GPU-time, correctness, or baseline-ranking point.

## Observation and necessary change

At manual stop, the service snapshot contained 36 `ray::IDLE` processes and four
native actors (two engine actors and two RayWorker actors). Their summed idle
RSS was 16,882,376,704 bytes; summed anonymous RSS was 9,572,519,936 bytes. These
are process RSS sums, not unique physical memory. The whole-service peak was
78,497,169,408 bytes, with 12,336,539 memory.high events and zero max/oom/oom_kill
events. The diagnostic was deliberately stopped for observed startup thrashing,
not a watchdog-threshold abort or a demonstrated OOM. No claim is made that all
pressure came from idle workers or that removing them suffices for model loading.

The retained Ray 2.51.2 Python startup passes the logical CPU count (40) as
`num_prestart_python_workers`. Official defaults have first-driver prestart
enabled. The private dLoRA entry now explicitly sets:

```python
_system_config = {
    "enable_worker_prestart": False,
    "prestart_worker_first_driver": False,
}
```

Logical CPU resources remain 40, physical affinity and 72/80 GiB limits are
unchanged, and the production object store remains 8 GiB for the whole service.
Actors are created on demand by the same native manager. DP2/TP1, GPU 0/1,
placement, periodic migration, LoRA operations, pool, generation and HTTP
measurement remain unchanged. No baseline performance parameter search.

Pinned primary sources:

- [Ray 2.51.2 startup code](https://github.com/ray-project/ray/blob/ray-2.51.2/python/ray/_private/services.py)
- [Ray 2.51.2 config defaults](https://github.com/ray-project/ray/blob/ray-2.51.2/src/ray/common/ray_config_def.h)
- [Ray 2.51.2 worker registration](https://github.com/ray-project/ray/blob/ray-2.51.2/src/ray/raylet/worker_pool.cc)

## Changed-path validation and limits

The existing unit check passed. A private CPU-only Ray 2.51.2 probe in the actual
native Python 3.9 environment kept CPU resources at 40, hid GPUs, and used an
explicit 80 MiB test object store. It observed zero prestarted workers and exactly
four on-demand actor processes, each in the bounded auxiliary CPU domain. The
probe completed in 9.466 s; that duration is not a model-startup benchmark.
Its resource-domain peak was 993,918,976 bytes, with no high/max/OOM/swap events.
The probe's empty domain was closed. No packages were changed in D780.

The main-repository launcher separately fixes cleanup census churn: a PID whose
owner/membership changed is recorded and excluded from individual signals;
qualified, pinned owned-cgroup cleanup proceeds. Qualification census remains
strict. The initial regression exposed a duplicate root cgroup.procs enumeration;
that duplicate traversal was removed, and all three targeted checks passed.
Both failed and passed test logs are retained.

RAW: `/home/qhq/serverless_llm_experiment/results/ieee_tc/dlora_qualification/d780_20261024`.
Reuse D778's unchanged native source view and D779's retained environment repair.
Next: actual native loading/GQA/LoRA and common HTTP qualification with a new
execution identity, then the existing limited 3B W1/W2 matrix. CPU success does
not establish native model correctness. Shared k3s and unrelated Ray are untouched.
