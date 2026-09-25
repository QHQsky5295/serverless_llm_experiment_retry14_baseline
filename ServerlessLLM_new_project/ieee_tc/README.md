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

## Loader fidelity finding (2026-09-25)

Both clean historical deployment JSONs specify `backend_config.load_format=auto`
and local `pretrained_model_name_or_path`, with LoRA enabled. The 7B serve log
also records the actual engine config using `auto` (lines 71/76 in the clean
run's serve log). The adapted backend only selects `serverless_llm` when no
explicit load format is supplied. Thus these runs did not use that specialized
checkpoint-loader path; their ~56–59 s engine startup is not a faithful
measurement of the paper's optimized checkpoint startup.

This is separate from the ~237 s average request dispatch wait. Before TC model
qualification, inspect and, if feasible, enable the native checkpoint mechanism
without replacing its policy. Otherwise report the reproduction boundary, not
an unqualified full-system claim. A ready-first router repair alone does not
resolve loader fidelity.

The clean deployment configs retain min=1/max=4/keep_alive=10, target=2 (7B)
and target=8 (3B). They are historical facts, not new frozen optima.

## Existing native-loader assets recovered (read-only audit, 2026-09-25)

Another local project's historical qualification provides a native 3B loader
compatibility witness, not a PrimeLoRA TC measurement:

`/home/qhq/relayserve_serverless_llm/paper_results/relayserve_v4_3/full_common/native_smoke_natural_store_devices_20260923/serverlessllm/`

Receipt SHA256: `fbdd1838df16213323653fbab74cc16802dd2f8e3bc6cad54ba58a7622613eb6`.
It records four HTTP-success requests, four native engines ready/reclaimed,
store registration/host loading, zero direct-path fallback and zero RelayServe
policy signals. Native commit is `0fd00cadaa8d495d53984f04777a3aec8137b363b77587d1221feac97da7c94e`,
not the older TC router-audit commit. Neither four responses nor GPU engine
startup proves complete 500-adapter support or the new generation contract.

| Existing asset | Evidence and reuse boundary |
|---|---|
| `models/vllm/v43-sllm-native-smoke-llama32-3b/rank_0/tensor.data_0` | Existing 6,425,499,648-byte native checkpoint; verify model identity before use; do not reconvert or duplicate blindly |
| `installs/serverless-llm-store-0.8.0-vllm0102-py312-v1/` | Python 3.12 build receipt PASS, SHA `d403ee96cffd7216ef0aa01c688da9a78838f177ca33a4d9fb91982d77f4ff13`; both wheel bytes rehashed and match receipt |
| Native store wheel | SHA `41a557075e62aa798d3ac922f5f9609451317344eac94c7815b27620fd99fffc` |
| grpcio 1.76.0 wheel | SHA `980a846182ce88c4f2f7e2c22c56aefd515daeb36149d1c897f83cf57999e0b6` |
| vLLM 0.10.2 loader compatibility overlay | Official serialization/load hooks ported to changed code locations; original patch SHA `0d5f9e4e1ea8901538b7a75bf3ac5caa706519bc9e2bbcb91d99aa8ce022f357` |

The loader environment was restored after that smoke. All six current source
files match the recorded preimages; the newly added loader file is absent, as
the restore receipt specifies. Thus native-loader support is **not currently
enabled merely because the environment exists**. Use version-checked reversible
adaptation only after reading the original installer and validating ownership.

The earlier `native_smoke_natural_queue_corrected_20260923` attempt failed:
the store exposed only one GPU, and other-GPU copies were invalid despite success
statuses. Do not reuse that attempt as qualified. Check actual store-visible
GPU UUIDs against workers. Existing cleanup uses global Ray-stop commands and is
not safe for TC unchanged. No other project's files or environments were modified
by this audit; no model weights, trace, pool or checkpoint were copied.

## Reproduction

```bash
python3 -m unittest discover -s tests -p test_ieee_tc_serverless_router.py -v
```

For an isolated official checkout, apply the pre-TC router compatibility patch
then the incremental repair. The rest of the native environment adaptation must
also be accounted for before runtime qualification. Do not overwrite historical
installed source or treat this router-only patch as the complete environment.
