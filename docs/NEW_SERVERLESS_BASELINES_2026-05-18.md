# New Serverless Baselines: 2026-05-18

This document records the ordered reproduction campaign requested after the
true-remote PrimeLoRA/FaaSLoRA mirror closed. It preserves each new baseline in
separate paths so old code and old paper data remain recoverable.

## ServerlessLLM-New

Status: closed formal candidate.

- Upstream: `https://github.com/ServerlessLLM/ServerlessLLM.git`
- Upstream commit: `9f50241baa5386e06a9321c51f19a9ef5f964c2b`
- Local source: `vendor_new_baselines/ServerlessLLM_new_main_20260518`
- Project entry: `ServerlessLLM_new_project/`
- Runtime env: `sllm_vllm0102_newserverless_20260518`
- Formal queue ids:
  - `20260518_serverlessllm_new_remote_v1` for the clean Llama-3.2 3B run
  - `20260518_serverlessllm_new_remote_v1_clean7b` for the clean Llama-2 7B rerun
- Result section:
  `results/paper_experiments/15_new_serverless_baselines_remote_v1/`
- Log section:
  `results/logs/new_serverless_baselines_remote_v1/serverlessllm_new/formal/`
- Remote artifact endpoints:
  - Llama-2 7B: `http://192.168.4.174:18081`
  - Llama-3.2 3B: `http://192.168.4.174:18080`

Validation:

- Llama-2 7B replay gate: `ok=4000`, `total=4000`, no `trace_expected`
  token fallback.
- Llama-3.2 3B replay gate: `ok=4000`, `total=4000`, no `trace_expected`
  token fallback.
- Clean 7B serve-log strict error scan was empty after excluding expected
  router request-log lines.

Headline metrics:

| Model | TTFT avg ms | TTFT p95 ms | service TTFT avg ms | dispatch avg ms | TPOT ms | E2E avg ms | RPS | Tok/s | SLO | Cost/req USD | CE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-2 7B | 237136.08 | 471638.44 | 408.59 | 236727.49 | 25.05 | 239704.18 | 0.96 | 98.397 | 0.0525 | 0.0026775 | 1.5581 |
| Llama-3.2 3B | 237811.23 | 472400.11 | 498.50 | 237312.73 | 14.89 | 239458.22 | 0.96 | 107.291 | 0.0553 | 0.0022500 | 1.8560 |

Interpretation:

- The result is fair to enter a candidate comparison because it uses the same
  true-remote shared trace, shared adapter subset, 4-GPU budget, LoRA requests,
  and `e2e_v3` summary fields as the closed remote mirror.
- It should be added as `ServerlessLLM-new`, not as a replacement for the old
  `ServerlessLLM` row, because the old result remains a separate historical
  baseline.
- End-to-end latency is dominated by dispatch/admission wait. Service-path TTFT
  stays sub-second, so paper wording must distinguish serverless admission
  behavior from backend token runtime.
- The first Llama-2 7B queue output is superseded by the clean `clean7b` rerun
  because a Medusa dependency operation briefly overlapped the earlier run.

## Next Ordered Gates

The requested ordered campaign is now closed:

1. Medusa official reproduction and LoRA/true-remote feasibility gate.
   Closed on 2026-05-19 as a local build/import success but not a formal
   runtime candidate on the current machine. The unmodified official build
   failed at `csrc/cuda_graph.cu` due a CUDA Graph API signature mismatch.
   A local adaptation patched that call, parameterized the SPDK/DPDK/GDRCopy
   paths, built local SPDK-Medusa and GDRCopy userspace libraries, and
   successfully imported `vllm._C`. Runtime still requires current-machine
   system state that is absent here: configured hugepages by default,
   SPDK-accessible NVMe/Optane devices, and GDRCopy kernel device setup.
2. FaaScale/LambdaScale official reproduction and LoRA/true-remote feasibility
   gate. Closed on 2026-05-19 as a local import/IPC/RDMA-binding success but
   not a formal runtime candidate on the current machine. The official package
   import path and protobuf runtime were repaired in an isolated env, the IPC
   extension builds/imports, and the RDMA-P2P `rdmc_shn` binding builds/imports
   after local dependency and CUDA 13 adaptation. Runtime initialization still
   finds zero IB devices because `/dev/infiniband` is absent and no usable
   InfiniBand class device is exposed. The source also lacks a ready
   Llama-3.2 3B config and LoRA/PEFT workload path.

Do not start a long Medusa or FaaScale formal run until their build/runtime
gate proves they can consume the Llama-2 7B and Llama-3.2 3B LoRA workload
without changing the closed true-remote workload variables.

## dLoRA Gate

Status: real-adapter scale-gate evidence, one 3B dispatch-only full replay,
three official period-migration short gates, four-GPU topology gates, and the
selected 3B period-migration full replay closed through 2026-05-21; not yet
adopted for formal table/figures as the official dLoRA row because the matching
7B full replay is still pending.

- Upstream: `https://github.com/LLMServe/dLoRA-artifact`
- Upstream commit: `75f1c439446fe194b1df8a24982ef9067841fab5`
- Local entry: `DLoRA_project/`
- Local source: `vendor_new_baselines/dLoRA_artifact_main_20260519`
- Evidence summary: `DLoRA_project/evidence/gate_2026-05-19.json`
- Real-adapter evidence: `DLoRA_project/evidence/real_adapt_2026-05-20.json`
- Formal preflight: `DLoRA_project/evidence/formal_preflight_2026-05-20.json`
- Full dispatch-only 3B replay:
  `DLoRA_project/evidence/formal_dispatch_only_3b_2026-05-20.json`
- Official period-migration 3B short gate:
  `DLoRA_project/evidence/formal_period_mig_gate128_3b_2026-05-21.json`
- Official period-migration 3B `max_num_seqs=2` short gate:
  `DLoRA_project/evidence/formal_period_mig_gate128_s2_3b_2026-05-21.json`
- Official period-migration 3B `max_num_seqs=4` short gate:
  `DLoRA_project/evidence/formal_period_mig_gate128_s4_3b_2026-05-21.json`
- Official period-migration 3B 4-GPU DP2/TP2 topology gate:
  `DLoRA_project/evidence/formal_period_mig_gate128_g2tp2_g4_s4_3b_2026-05-21.json`
- Official period-migration 3B full replay:
  `DLoRA_project/evidence/formal_period_mig_full4000_3b_2026-05-21.json`
- Compatibility patch: `DLoRA_project/patches/modern_ray_import_compat.patch`
- Real-adapter patch:
  `DLoRA_project/patches/real_peft_llama32_e2e_compat_20260520.patch`
- Formal runtime compatibility patch:
  `DLoRA_project/patches/formal_500_adapter_runtime_compat_20260520.patch`
- Build env: `dlora_medusa_clone_20260519`
- Import result: `vllm.__version__ == 0.1.4`

Local adaptation:

- first official-style environment path hit repeated network `IncompleteRead`
  failures while downloading CUDA/PyTorch packages, so an isolated clone of the
  existing cu121 Medusa env was created instead of modifying that env;
- `pip install -e` required `--no-deps --no-build-isolation` to avoid pulling
  a second torch build from the network;
- CUDA extension build required local CUDA 12.1 header/library precedence over
  mixed CUDA 13.x headers in the cloned env;
- Ray 2.51.2 imports were made tolerant of absent `ray.air` optional deps.
- 2026-05-20 real-adapter adaptation added a narrow PEFT
  `adapter_model.safetensors` loader for Llama q/k/v/o LoRA weights, Llama-3.2
  grouped-query shapes, adapter-subset-to-`model_id` mapping, explicit
  `--no-use-dummy-weights`, and replay parsing for dLoRA JSON+NUL responses.
  dLoRA scheduling, migration, and adapter orchestration were not replaced.

Formal blocker:

- The earlier "no real PEFT loader / no Llama-3.2 path" blocker is resolved for
  the local adaptation, but only as a compatibility layer.
- Real-weight Llama-3.2 3B filtered gates pass through 128 adapters / 512
  requests; real-weight Llama-2 7B passes a 2-adapter / 16-request gate.
- Llama-3.2 3B has one full 4000-request / 500-adapter true-remote replay:
  `ok=4000/4000`, no `trace_expected` fallback, `e2e_v3`. It used
  `migration_type=1` / `dlora_dispatch_only`, 2 groups on GPUs 0 and 1,
  `gpu_memory_utilization=0.92`, `max_num_seqs=1`, and
  `max_num_batched_tokens=1024`.
- That dispatch-only result is intentionally not the official dLoRA row. Its
  TTFT avg is `1283513.75 ms`, TTFT p95 is `4142829.11 ms`, throughput is
  `63.843 tok/s`, and CE is `0.2465`. The poor result is explained by static
  placement skew: the tail was drained by `engine_id 0` while the other GPU was
  idle. It is not an OOM or failed remote artifact path.
- The first official upstream `migration_type=3` / `dlora_period_mig` gate also
  passes on the same 3B true-remote workload variables: 500 adapters, first 128
  scheduled requests, `ok=128/128`, `fail=0`, no token fallback, and no
  in-replay OOM. Its 2-GPU `max_num_seqs=1` envelope is viable but slow
  (`TTFT_e2e` avg 29544.29 ms, p95 116423.03 ms), so it is evidence for
  viability and tuning, not yet a formal table row.
- A second official `migration_type=3` gate with only wrapper/runtime envelope
  change to `max_num_seqs=2` also passes (`ok=128/128`, no token fallback, no
  in-replay OOM). It improves the tail (`TTFT_e2e` avg 24674.56 ms, p95
  59104.07 ms, p99 67024.53 ms, 81.885 tok/s), but service-side engine wait
  still dominates execution. This is not yet the best-effort formal row.
- A third official `migration_type=3` 2-GPU gate with `max_num_seqs=4` also
  passes (`ok=128/128`, no token fallback, no in-replay OOM). It is currently
  the best 2-GPU envelope (`TTFT_e2e` avg 14517.67 ms, p95 26510.79 ms, p99
  28836.86 ms, 92.127 tok/s), reducing engine wait without touching dLoRA
  scheduling or migration.
- The first official `migration_type=3` 4-GPU topology gate did not reach HTTP
  readiness. Remote materialization completed for all `500/500` adapters, then
  Ray killed a worker because host memory reached `124.16GB / 125.38GB` at the
  `0.99` node-memory threshold. This is a wrapper/runtime memory-envelope gate
  at default `swap_space_gb=8`, not a CUDA OOM and not a measured replay result.
- The `swap_space_gb=2` 4-GPU rerun also did not reach HTTP readiness. It
  reduced the dLoRA CPU KV cache envelope but still hit severe host-memory/swap
  pressure because Ray's default object store reserved about `38.6GB` and four
  workers loaded model/adapter state concurrently. The next wrapper-only gate
  pre-starts Ray with bounded object-store memory and connects via
  `RAY_ADDRESS=auto`.
- The bounded-Ray object-store rerun also did not reach HTTP readiness. It
  confirmed the wrapper-level Ray fix was active (`object_store_memory` bounded
  to `8GiB` and dLoRA connected through `RAY_ADDRESS=auto`), but Ray still
  killed a worker before replay. The root cause is now the DP4/TP1 dLoRA
  startup envelope itself: four independent vLLM/Ray engines duplicate model
  and adapter-runtime state on a 125GB host. This is not CUDA OOM, not remote
  materialization failure, and not a measured replay result.
- The wrapper-only `num_groups=2, tensor_parallel_size=2` four-GPU topology
  gate reached HTTP readiness and completed `ok=128/128`, `fail=0`, no token
  fallback, no CUDA OOM, and no host OOM. It is a valid topology gate, but it
  is not the best formal replay envelope: `TTFT_e2e` avg worsened to
  `18630.69 ms` versus `14517.67 ms` for the 2-GPU DP2/TP1
  `max_num_seqs=4` gate, p95 worsened to `33096.62 ms` versus `26510.79 ms`,
  and CE dropped to `1.9696` versus `5.3354`. The root cause is not a local
  patch bug: TP2 avoids DP4/TP1 startup duplication, but it spends two GPUs per
  logical dLoRA group and service-side engine wait still dominates.
- The selected 3B official full replay then used the best measured wrapper-only
  envelope, DP2/TP1 with `max_num_seqs=4`, on the complete 4000-request
  Llama-3.2 3B true-remote trace. It completed `ok=4000/4000`, `fail=0`, no
  `trace_expected` fallback, no CUDA OOM, no host OOM, and no core dLoRA code
  changes. Final metrics: `TTFT_e2e` avg `11162.43 ms`, p50 `9308.08 ms`, p95
  `27280.54 ms`, p99 `36419.18 ms`, throughput `115.666 tok/s`, SLO attainment
  `0.27075`, infra cost/request `$0.00196784`, CE `45.5240`, and goodput
  tokens/GPU-s `4.0734`. The Raylet/AsyncEngineDeadError messages at the end
  occur after replay and summary are written while the wrapper stops Ray/vLLM;
  record them as shutdown noise, not replay failure.
- The dLoRA wrapper now writes `max_num_seqs`, `max_num_batched_tokens`,
  `gpu_memory_utilization`, `gpu_capacity`, and `swap_space_gb` into
  deploy/manifest metadata and launch logs so future envelope sweeps are
  auditable. It also fixes `MANIFEST.replayed_requests`, and records optional
  `ray_object_store_memory_bytes` / `ray_num_cpus` when the bounded-Ray path is
  enabled.
- Do not enter dLoRA into formal tables until a no-dummy, no-`trace_expected`,
  full 3B run with upstream `migration_type=3` and a full 7B run pass without
  rewriting dLoRA scheduling or migration. The full 3B requirement is now met;
  the remaining blocker is the Llama-2 7B true-remote full replay under an
  auditable wrapper-only dLoRA envelope.

## Loquetier Gate

Status: real-adapter scale-gate evidence closed through 256 adapters on
Llama-3.2 3B and 128 adapters on Llama-2 7B. The 3B/500-adapter formal
preflight OOMs on a single RTX 3090, so Loquetier is not adopted for formal
table/figures.

- Upstream: `https://github.com/NJUDeepEngine/Loquetier`
- Upstream commit: `aae33baeeb19777129c1ccbff99a898d4a0e2c63`
- Local entry: `Loquetier_project/`
- Local source: `vendor_new_baselines/Loquetier_main_20260520`
- Evidence summary: `Loquetier_project/evidence/real_adapt_2026-05-20.json`
- Formal preflight:
  `Loquetier_project/evidence/formal_preflight_2026-05-20.json`
- Compatibility patch:
  `Loquetier_project/patches/loquetier_local_compat_20260520.patch`
- Build env: `loquetier_20260520`

Local adaptation:

- isolated Loquetier in a cloned CUDA 12.1 / torch 2.1.2 environment instead
  of modifying existing closed experiment environments;
- initialized upstream CUTLASS and built SMLM kernels for RTX 3090 `sm_86`;
- rebuilt Loquetier kernels with `LOQUETIER_GROUP_SIZES=1,2,3,4,8` so
  Llama-3.2 3B's GQA group size 3 has a real dispatch path;
- added Python 3.9 annotation compatibility and PEFT 0.17 import/active-adapter
  compatibility;
- fixed mixed-rank PEFT adapter handling by deriving rank from actual
  `lora_A`/`lora_B` tensor shapes and padding along the correct rank dimension.

Closed gates:

- Llama-3.2 3B, 2 adapters / 16 filtered requests: `ok=16/16`.
- Llama-2 7B, 2 adapters / 16 filtered requests: `ok=16/16`.
- Llama-3.2 3B, 16 adapters / 64 filtered requests: `ok=64/64`.
- Llama-2 7B, 16 adapters / 64 filtered requests: `ok=64/64`.
- Llama-3.2 3B, 64 adapters / 256 filtered requests: `ok=256/256`.
- Llama-2 7B, 64 adapters / 128 filtered requests: `ok=128/128`.
- Llama-3.2 3B, 128 adapters / 512 filtered requests: `ok=512/512`.
- Llama-2 7B, 128 adapters / 256 filtered requests: `ok=256/256`.
- Llama-3.2 3B, 256 adapters / 1024 filtered requests: `ok=1024/1024`.
- Llama-3.2 3B, 500 adapters / 16-request preflight: OOM during
  `MixedLoraModel` adapter weight materialization before replay JSON output.

Formal blocker:

- Loquetier can now consume real PEFT adapters from the closed true-remote
  artifact set and replay both backbone traces through its mixed-LoRA path.
- It remains an offline single-GPU multi-LoRA runner rather than a serverless
  control plane. On this 24GB RTX 3090 hardware, the 500-adapter closed
  workload cannot be loaded without changing Loquetier's core adapter
  materialization or placement design. Do not enter Loquetier into formal
  tables unless an upstream-compatible path passes the no-fallback full replay.

## AIBrix Gate

Status: build/runtime-LoRA component gate closed on 2026-05-20; not adopted for
formal table/figures on this machine.

- Upstream: `https://github.com/vllm-project/aibrix`
- Upstream tag: `v0.6.0`
- Upstream commit: `52405d78c63c68df5be9e8bdc5e21ccd6c4abde2`
- Local entry: `AIBrix_project/`
- Local source: `vendor_new_baselines/AIBrix_v0.6.0_20260520`
- Build env: `aibrix_20260520`
- Python runtime env: `aibrix_py_20260520`
- Evidence summary: `AIBrix_project/evidence/gate_2026-05-20.json`

Local adaptation:

- cloned upstream release into an ignored vendor directory so no prior baseline
  code or data is overwritten;
- created isolated Go 1.22.6 and Python 3.11 environments;
- built upstream `controller-manager`;
- built upstream `gateway-plugins` in no-ZMQ mode;
- fixed the ZMQ build as a local dependency issue by using `/usr/bin/gcc` and
  user-space `zeromq`, `libsodium`, and `pkg-config` in the isolated env;
- installed/imported the upstream Python `aibrix` runtime package from source;
- launched a short local component gate where AIBrix runtime called a vLLM
  0.10.2 server's dynamic LoRA API with the real
  `llama32_3b_a500_v1_modelscope/code_lora` adapter.

Closed component gate:

- AIBrix runtime `/ready` returned HTTP 200.
- AIBrix runtime `/v1/lora_adapter/load` returned HTTP 200 for the real
  `code_lora` path.
- vLLM `/v1/models` showed `code_lora` with parent `base`.
- A completion request using `model=code_lora` returned HTTP 200 with
  `prompt_tokens=5`, `completion_tokens=4`.
- AIBrix runtime `/v1/lora_adapter/unload` returned HTTP 200 and vLLM removed
  the adapter.

Formal blocker:

- This component gate does not replace AIBrix's actual Kubernetes control
  plane. A formal AIBrix run requires CRDs, controller-manager, Envoy gateway,
  gateway-plugins, GPU vLLM pods, runtime sidecars, and `ModelAdapter`
  resources.
- Current machine state blocks that platform path: `/usr/local/bin/kubectl`
  and `/usr/local/bin/helm` are root-only, Docker socket access is denied,
  `kind`/`minikube` are absent, and passwordless sudo is unavailable.
- Therefore AIBrix should be treated as appendix/gate evidence only until a
  GPU Kubernetes runtime is available for the same true-remote 3B/7B LoRA
  workload. Do not enter AIBrix into formal tables based only on the runtime
  sidecar gate.

## HydraServe Gate

Status: control-plane import and LoRA-interface audit closed on 2026-05-20;
not adopted for formal table/figures on this machine.

- Upstream: `https://github.com/LLMServe/hydraserve`
- Upstream commit: `8ae605de354ccfa2e9095514cdb4a9e9c56aa56b`
- Local entry: `HydraServe_project/`
- Local source: `vendor_new_baselines/HydraServe_main_20260520`
- Control env: `hydraserve_py_20260520`
- Evidence summary: `HydraServe_project/evidence/gate_2026-05-20.json`

Closed checks:

- official source cloned at commit `8ae605de354ccfa2e9095514cdb4a9e9c56aa56b`;
- isolated Python control-plane dependencies installed and run with
  `PYTHONNOUSERSITE=1`;
- `main`, `scheduler`, `resource_manager`, `engine`, `ModelInfo`, and
  `ImageInfo` import successfully;
- official Docker image build path fails because the current user cannot
  access `/var/run/docker.sock`;
- official `src/main.py` and `src/start_storage_server.py` fail before service
  launch because no usable Kubernetes config is available;
- embedded vLLM 0.4.2 source parses static LoRA arguments
  `--enable-lora` and `--lora-modules`.

Formal blocker:

- HydraServe's official system is a Kubernetes deployment with storage-server
  pods, GPU vLLM pods, node labels, and GPU-share scheduling. This machine
  exposes neither a usable Kubernetes runtime nor Docker access to `qhq`.
- The embedded vLLM source has static LoRA support, but HydraServe's worker app
  does not wire LoRA arguments into the K8s pod runtime, and the
  `Request -> Instance -> ChatEngine` path sends a fixed base-model name rather
  than preserving per-request adapter identity. Supporting the closed
  PrimeLoRA LoRA workload would require scheduler/request semantic changes, not
  just local path or library adaptation.
- Therefore HydraServe is appendix/gate evidence only. Do not enter it into
  formal tables unless a real GPU Kubernetes deployment can replay the
  unchanged true-remote 3B/7B LoRA workload into `e2e_v3`.

## Sarathi-Serve Gate

Status: source/package metadata and LoRA-workload audit closed on 2026-05-20;
not adopted for formal table/figures.

- Upstream: `https://github.com/microsoft/sarathi-serve`
- OSDI artifact branch: `osdi-sarathi-serve`
- OSDI artifact commit: `ceaa0660ea2487976101a8167aad5c8046e85b27`
- Main commit checked: `96f9911790ecc00af12ee9fae47cb8fa9ba0d199`
- Local entry: `SarathiServe_project/`
- Local OSDI source: `vendor_new_baselines/SarathiServe_osdi_20260520`
- Local main source: `vendor_new_baselines/SarathiServe_main_20260520`
- Evidence summary: `SarathiServe_project/evidence/gate_2026-05-20.json`

Closed checks:

- OSDI artifact metadata resolves as `sarathi 0.1.7`;
- main branch metadata resolves as `sarathi 0.1.8`;
- OSDI artifact README targets CUDA 12.1 on A100/A40 with torch 2.3;
- main branch README targets CUDA 12.3 on H100/A100 and now lists torch 2.9.0;
- source scan finds zero LoRA/adapter/PEFT terms in the OSDI branch;
- main branch has one `lora` hit, `LoRAModulePath`, but no
  `enable_lora`, `LoRARequest`, `lora_modules`, PEFT loader, or adapter-aware
  scheduler.

Formal blocker:

- Sarathi-Serve is valuable scheduling related work, but the faithful OSDI
  artifact does not support the closed PrimeLoRA LoRA workload. Adding that
  support would require new model-executor LoRA layers, adapter loading,
  request adapter identity, and scheduler/API semantics. That is building a
  LoRA serving system for Sarathi-Serve rather than adapting it to local paths.
- Therefore Sarathi-Serve is appendix/gate evidence only. Do not enter it into
  formal tables unless an upstream-compatible LoRA path can replay the
  unchanged true-remote 3B/7B workload into `e2e_v3`.

## Medusa Gate

Status: closed local build/import gate, not adopted for formal table/figures on
this machine.

- Upstream commit: `6581d2e5ec8fa4ecdabcdb50560982a78ea3ca89`
- Local entry: `Medusa_project/`
- Build log root:
  `results/logs/new_serverless_baselines_remote_v1/medusa/gate/`
- Official gate: `cusparse.h` not found until conda CUDA target include paths
  were added; then `vllm._moe_C` compiled and `vllm._C` failed at
  `csrc/cuda_graph.cu(131)` because the local CUDA Graph header expects a newer
  `cuGraphGetEdges` signature.
- Local adaptation:
  - patch file: `patches/Medusa_localadapt_20260519.patch`
  - env: `medusa_localadapt_20260519`
  - SPDK-Medusa commit:
    `3dd897f8406f0824d6de08d0cb21df9e0f9ed76c`
  - GDRCopy commit: `737708f3b5955cb6ef6b47bba35600a31bce4222`
  - `vllm._C` and `vllm._moe_C` build and import successfully.
- Runtime blockers:
  - default SPDK initialization fails because `HugePages_Total=0`;
  - `MEDUSA_SPDK_NO_HUGE=1` smoke init/fini passes, but this is not a formal
    paper-equivalent configuration;
  - no NVMe/Optane device is visible via `lspci`, `/sys/class/nvme`, or
    `/dev/nvme*`;
  - `/dev/gdrdrv` is absent and passwordless sudo is unavailable.
- LoRA note: inherited vLLM LoRA request code is present, but no valid
  true-remote `e2e_v3` LoRA replay can be produced until the SPDK runtime
  requirements and Llama-2 7B / Llama-3.2 3B workload adapter are satisfied.

## FaaScale / LambdaScale Gate

Status: closed local import/build gate, not adopted for formal table/figures
on this machine.

- LambdaScale upstream commit: `9db210fcb6979f7c1f73f9819a77e0edb6c5e343`
- RDMA-P2P upstream commit: `ed83237439d2103141fbc7c9b97f348055b6cb53`
- Local entry: `FaaScale_project/`
- Build log root:
  `results/logs/new_serverless_baselines_remote_v1/faascale/gate/`
- Local adaptation env: `faascale_20260519`
- Local RDMA-P2P patch:
  `patches/RDMA_P2P_localadapt_20260519.patch`
- Official raw package import fails with `ModuleNotFoundError:
  test_bed_local`. An isolated symlink shim fixes the package layout.
- The generated protobuf files require `protobuf==3.20.x`; after minimal
  dependency installation, utility import succeeds and reports Llama-2 7B
  support but no Llama-3.2 3B config.
- The LambdaScale IPC extension builds/imports after setting the torch library
  runtime path.
- RDMA-P2P can build the targeted `rdmc_shn` Python binding after local
  Derecho metadata, pybind11, rdma-core headers, and CUDA 13 `cuCtxCreate`
  adaptation. Full helper build still has a nonessential `persistent_test`
  OpenSSL link/API failure.
- Runtime blocker: `wrapper_initialize()` finds zero IB devices and returns
  `False`; `/dev/infiniband` is absent, `/sys/class/infiniband` exposes no
  usable device, and passwordless sudo is unavailable for driver/device setup.
- LoRA note: no LoRA/PEFT/QLoRA path was found in targeted source search. A
  fair true-remote LoRA `e2e_v3` replay would require nontrivial model and
  workload adaptation in addition to RDMA runtime availability.
