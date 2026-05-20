# dLoRA Gate Evidence

This directory mirrors compact, non-overwriting dLoRA evidence from
`/home/qhq/serverless_llm_baselines/DLoRA_project/`. It does not contain the
19GB runtime cache from the formal replay.

Current state:

- Build/import and real-adapter gates pass for the local compatibility layer.
- The Llama-3.2 3B dispatch-only full true-remote replay completed
  `4000/4000` with `0` failures and `e2e_v3` metrics.
- That full run used upstream `migration_type=1` (`dlora_dispatch_only`), so it
  is appendix/ablation evidence rather than the official dLoRA main-table row.
- Upstream `migration_type=3` (`dlora_period_mig`) gates completed `128/128`
  on the 3B/500-adapter true-remote workload with no token fallback. The
  2-GPU `max_num_seqs=4` gate is currently the best local envelope, improving
  over `max_num_seqs=1` and `max_num_seqs=2`.
- The first 4-GPU topology startup gate did not reach replay: all `500/500`
  remote adapters materialized, then Ray killed a worker at
  `124.16GB / 125.38GB` host memory with the default `swap_space_gb=8`. This is
  recorded as a startup memory-envelope gate, not as a CUDA OOM or measured
  replay result.
- A `swap_space_gb=2` rerun also failed before replay. It reduced the dLoRA CPU
  KV cache envelope, but Ray's default object store was about `38.6GB` and the
  four worker startup still entered host-memory/swap pressure. The next fair
  4-GPU gate bounds Ray object-store memory from the wrapper.
- The bounded-Ray object-store rerun confirmed the wrapper fix was active
  (`object_store_memory=8589934592` and `RAY_ADDRESS=auto`), but DP4/TP1 still
  failed before replay. Four dLoRA/vLLM engines duplicated startup state and Ray
  killed a worker under host-memory pressure. This is still not a CUDA OOM, not
  a remote artifact failure, and not a measured replay result.
- The next fair dLoRA topology gate keeps the 4-GPU budget while reducing
  startup duplication via `num_groups=2, tensor_parallel_size=2`.
- The next fair dLoRA step is to run short runtime/topology gates, then a full
  `migration_type=3` replay without rewriting dLoRA scheduling or migration.

Files:

- `gate_2026-05-19.json`: source/build gate.
- `real_adapt_2026-05-20.json`: real PEFT adapter and filtered scale-gate
  summary.
- `formal_preflight_2026-05-20.json`: historical formal preflight snapshot
  taken while unrelated GPU memory was occupied.
- `formal_dispatch_only_3b_2026-05-20.json`: compact record of the completed
  3B/4000-request/500-adapter dispatch-only run.
- `formal_period_mig_gate128_3b_2026-05-21.json`: compact record of the
  official period-migration 3B/128-request/500-adapter gate.
- `formal_period_mig_gate128_s2_3b_2026-05-21.json`: compact record of the
  official period-migration 3B/128-request/500-adapter `max_num_seqs=2` gate
  plus the metadata-audit repair.
- `formal_period_mig_gate128_s4_3b_2026-05-21.json`: compact record of the
  official period-migration 3B/128-request/500-adapter `max_num_seqs=4` gate.
- `formal_period_mig_gate128_g4_s4_hostoom_3b_2026-05-21.json`: compact record
  of the first 4-GPU startup memory-envelope gate.
- `formal_period_mig_gate128_g4_s4_swap2_hostoom_3b_2026-05-21.json`: compact
  record of the 4-GPU `swap_space_gb=2` startup memory-envelope gate.
- `formal_period_mig_gate128_g4_s4_swap2_obj8_hostoom_3b_2026-05-21.json`:
  compact record of the 4-GPU bounded-Ray object-store startup
  memory-envelope gate.
