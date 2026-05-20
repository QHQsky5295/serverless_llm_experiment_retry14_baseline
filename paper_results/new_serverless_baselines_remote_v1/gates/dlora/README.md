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
- The wrapper-only DP2/TP2 four-GPU topology gate then reached HTTP readiness
  and completed `128/128` requests with no token fallback, no CUDA OOM, and no
  host OOM. It is valid but worse than the best 2-GPU `max_num_seqs=4` gate:
  `TTFT_e2e` avg `18630.69 ms` versus `14517.67 ms`, p95 `33096.62 ms` versus
  `26510.79 ms`, and CE `1.9696` versus `5.3354`.
- The selected 3B full `migration_type=3` replay used the best measured
  wrapper-only envelope, DP2/TP1 with `max_num_seqs=4`, and completed
  `4000/4000` requests with no token fallback, no CUDA OOM, no host OOM, and no
  dLoRA core rewrite. Final metrics: `TTFT_e2e` avg `11162.43 ms`, p95
  `27280.54 ms`, p99 `36419.18 ms`, `115.666 tok/s`, CE `45.5240`.
- The first Llama-2 7B DP2/TP2 four-GPU gate materialized all `500/500` true
  remote adapters and used a bounded 8GiB Ray object store, but it failed
  before HTTP readiness. Ray reported host-memory pressure, GCS aborted, and
  the object-store debug dump was still `0 / 8.58993GB`. This is a startup
  topology memory-envelope failure, not CUDA OOM, not remote artifact failure,
  and not a measured replay result.
- The follow-up Llama-2 7B one-group TP4 gate moved past adapter-pool loading
  and placement, but failed before HTTP readiness with `# GPU blocks: 0,
  # CPU blocks: 1024` and `No available memory for the cache blocks`. This is
  a GPU KV-cache envelope gate, not host OOM, CUDA OOM, remote artifact
  failure, or measured replay output.
- Raising that G1/TP4 gate to `gpu_memory_utilization=0.95` still produced
  `# GPU blocks: 0, # CPU blocks: 1024`, so the next wrapper-only 7B lever is
  lowering dLoRA GPU adapter capacity while keeping 500 remote adapters in the
  CPU pool.
- Reducing `gpu_capacity` to `2` under the same G1/TP4,
  `gpu_memory_utilization=0.95`, 500-adapter envelope still produced
  `# GPU blocks: 0, # CPU blocks: 1024`. Remote materialization, real adapter
  loading, and placement completed, so the remaining wrapper-only lever is the
  minimal `gpu_capacity=1` envelope with `gpu_memory_utilization=0.99`.
- The full dLoRA paper row remains pending until the matching Llama-2 7B
  true-remote full replay closes under an auditable wrapper-only envelope.

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
- `formal_period_mig_gate128_g2tp2_g4_s4_3b_2026-05-21.json`: compact record
  of the successful but non-selected four-GPU DP2/TP2 topology gate.
- `formal_period_mig_full4000_3b_2026-05-21.json`: compact record of the
  selected 3B official full replay candidate.
- `formal_period_mig_gate128_7b_g2tp2_swap2_obj8_hostoom_2026-05-21.json`:
  compact record of the 7B DP2/TP2 startup memory-envelope gate.
- `formal_period_mig_gate128_7b_g1tp4_gpu_blocks0_2026-05-21.json`: compact
  record of the 7B G1/TP4 zero-GPU-cache-block gate.
- `formal_period_mig_gate128_7b_g1tp4_u95_gpu_blocks0_2026-05-21.json`:
  compact record of the 7B G1/TP4 high-GPU-utilization zero-GPU-cache-block
  gate.
- `formal_period_mig_gate128_7b_g1tp4_u95_cap2_gpu_blocks0_2026-05-21.json`:
  compact record of the 7B G1/TP4 `gpu_capacity=2` zero-GPU-cache-block gate.
