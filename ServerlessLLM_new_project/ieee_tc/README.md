# IEEE TC Serverless audit and minimal router repair

Display name: **Serverless**. Internal identity remains `serverlessllm_new`.

## Source identity

- Official commit: `9f50241baa5386e06a9321c51f19a9ef5f964c2b`.
- Official router SHA256: `b594f8c288df615f98ba2cb35b151e8ef8c95e25c4e06c953b5d7468e21fe7ca`.
- Pre-TC adapted router SHA256: `0182bd4862c1c0e1c4bf5d00507704f05d3d5092c267b4670ff31a4c79b18843`.
- TC repaired router SHA256: `2bcbba0d8a3fdf46354f75c1d8ca5bcd79e505f059fcb296bf30dbff176e8f6e`.
- `pre_tc_router_compat.patch` preserves the existing router adaptation relative
  to the official commit. It is not attributed to this execution.
- `ready_before_wait.patch` is the incremental TC change. Existing staged vendor
  adaptations are preserved; this patch is separately backed up in the harness.

The [official historical source](https://raw.githubusercontent.com/ServerlessLLM/ServerlessLLM/9f50241baa5386e06a9321c51f19a9ef5f964c2b/sllm/routers/roundrobin_router.py)
waits one second before checking ready replicas for every dequeued request.
The current public main path checked on 2026-09-25 has the same ordering.

## Repair scope and evidence

Check ready replicas first; retain the original one-second wait when empty and
the original loop interval when the selected replica is full. Round-robin index
progression, capacity checks, autoscaling, loading and migration are unchanged.
Queued cancellation is skipped; cancellation during awaited reservation rolls
back the acquired count so it cannot kill the routing loop or leak that slot.
No Prime routing, planner or admission policy is introduced.

Six no-GPU tests execute the actual method AST (not a separately rewritten
router): official ready path, repaired ready path, empty, full, queued cancel,
cancel during reservation. All passed. A virtual clock verifies mandatory waits;
its timings are **not inference latency measurements**.

| Case | Official | Repaired |
|---|---|---|
| Four requests, two ready replicas | a,b,a,b; four mandatory waits | a,b,a,b; no mandatory wait |
| Empty then ready | Not the optimization target | Wait and recheck |
| First replica full | Original progression retained | Wait then select next replica |
| Queued/during-reservation cancellation | Not used as valid performance input | Router remains live; acquired count rolled back |

Full inference cancellation, scale-down races, runtime import/source identity,
loader fidelity, actual worker resource containment and two 1,000-request model
pairs remain qualification gates. No model replay has run with this repair yet.
The environment's installed source has not been overwritten.

## Historical reuse

Reuse clean 7B and 3B logs, not the contaminated superseded 7B run. Extend the
existing summarizer using `--timing-audit-only --replay PATH --model-profile
llama2_7b|llama32_3b --output NEW_PATH`. Output is exclusive, never overwrites.

Curated evidence and figures live in the main repository under
`paper_results/ieee_tc/serverless_audit/` and `figs/ieee_tc/serverless_audit/`.
Each has 4,000 recorded successful requests, but differs from the new generation
and resource contract: **R2 diagnosis only**, not a reused TC main result.

Average router wait is 236.666 s (7B) and 237.200 s (3B), while average
service TTFT is 0.409/0.499 s. Median backend-start gap is 1.00227/1.00220 s.
These observations plus the source/test evidence support polling as a material
bottleneck hypothesis. They do not quantify the repaired end-to-end gain,
prove GPU saturation, or measure the paper's native checkpoint-startup path.
The old output lacks ready-at-enqueue telemetry; do not invent it.

## Reproduction

```bash
python3 -m unittest discover -s tests -p test_ieee_tc_serverless_router.py -v
```

For an isolated official checkout, apply the pre-TC router compatibility patch
then the incremental repair. The rest of the native environment adaptation must
also be accounted for before runtime qualification. Do not overwrite historical
installed source or treat this router-only patch as the complete environment.
