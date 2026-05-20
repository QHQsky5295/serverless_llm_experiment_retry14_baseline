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

Status: real-adapter scale-gate evidence closed through 2026-05-20, not yet
adopted for formal table/figures.

- Upstream: `https://github.com/LLMServe/dLoRA-artifact`
- Upstream commit: `75f1c439446fe194b1df8a24982ef9067841fab5`
- Local entry: `DLoRA_project/`
- Local source: `vendor_new_baselines/dLoRA_artifact_main_20260519`
- Evidence summary: `DLoRA_project/evidence/gate_2026-05-19.json`
- Real-adapter evidence: `DLoRA_project/evidence/real_adapt_2026-05-20.json`
- Formal preflight: `DLoRA_project/evidence/formal_preflight_2026-05-20.json`
- Compatibility patch: `DLoRA_project/patches/modern_ray_import_compat.patch`
- Real-adapter patch:
  `DLoRA_project/patches/real_peft_llama32_e2e_compat_20260520.patch`
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
- Formal 4000-request / 500-adapter runs are not yet launched. The preflight
  says current external GPU memory occupancy makes multi-GPU dLoRA placement
  unsafe to start, and single-GPU 3B/500 is impossible because the LoRA GPU pool
  alone exceeds a 24GB RTX 3090. Do not enter dLoRA into formal tables until a
  no-dummy, no-`trace_expected`, full 3B/7B run passes without rewriting dLoRA.

## Loquetier Gate

Status: real-adapter scale-gate evidence closed through 16 adapters on both
closed backbones, not yet adopted for formal table/figures.

- Upstream: `https://github.com/NJUDeepEngine/Loquetier`
- Upstream commit: `aae33baeeb19777129c1ccbff99a898d4a0e2c63`
- Local entry: `Loquetier_project/`
- Local source: `vendor_new_baselines/Loquetier_main_20260520`
- Evidence summary: `Loquetier_project/evidence/real_adapt_2026-05-20.json`
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

Formal blocker:

- Loquetier can now consume real PEFT adapters from the closed true-remote
  artifact set and replay both backbone traces through its mixed-LoRA path.
- It remains an offline single-GPU multi-LoRA runner rather than a serverless
  control plane, and it has not completed the full 4000-request / 500-adapter
  workload. Do not enter Loquetier into formal tables until that no-fallback
  replay passes without replacing Loquetier's core SMLM/mixed-LoRA logic.

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
