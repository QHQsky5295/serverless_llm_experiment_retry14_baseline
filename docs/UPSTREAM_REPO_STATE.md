# 上游仓库状态记录

本文档用于记录当前 baseline 工程依赖的上游仓库版本，以及我们在本地对这些上游仓库做过的修改方式。

## 记录原则

1. 外层 baseline 工程仓库不直接提交整个上游源码目录。
2. 所有上游依赖均通过以下方式复现：
   - 记录上游仓库 URL
   - 记录上游 commit hash
   - 保存本地 patch 文件
3. 这样做的目标是：
   - 保证当前 baseline 工程可回退
   - 避免把大体量上游仓库和无关历史直接塞进当前工程仓库
   - 仍然保留我们本地复现所需的真实修改

## 当前上游状态

### ServerlessLLM

- 上游仓库：`https://github.com/ServerlessLLM/ServerlessLLM.git`
- 本地分支：`main`
- 本地路径：`/home/qhq/serverless_llm_baselines/repos/ServerlessLLM`
- 当前上游基线 commit：`9f50241baa5386e06a9321c51f19a9ef5f964c2b`
- 当前本地 patch 文件：
  - [ServerlessLLM_local_changes.patch](/home/qhq/serverless_llm_baselines/patches/ServerlessLLM_local_changes.patch)
- 当前本地工作树存在 fair-comparison 适配修改，主要覆盖 backend、controller、
  router、store manager 和 CLI。不要对该目录执行 reset/checkout；如需复核，
  先保存或审阅现有 diff。

### ServerlessLLM-new 2026-05-18

- 上游仓库：`https://github.com/ServerlessLLM/ServerlessLLM.git`
- 本地路径：`/home/qhq/serverless_llm_baselines/vendor_new_baselines/ServerlessLLM_new_main_20260518`
- 当前上游基线 commit：`9f50241baa5386e06a9321c51f19a9ef5f964c2b`
- 项目入口：`/home/qhq/serverless_llm_baselines/ServerlessLLM_new_project`
- 当前本地状态：保持为独立官方 clone，外层仓库通过 `.gitignore` 排除
  `vendor_new_baselines/`，避免把整份上游源码和嵌套 `.git` 提交进 harness。
- 当前复现范围：使用 `sllm_vllm0102_newserverless_20260518` 环境和外层
  `scripts/run_serverlessllm_new_remote_formal_queue.sh`，在 true-remote
  7B/3B shared trace + shared LoRA subset 上完成 `e2e_v3` 候选结果。
- 数据边界：不覆盖旧 `ServerlessLLM_project/`、旧 ServerlessLLM result
  directories、`figs/` 或 `paper_results/final_v2/`。

### Medusa 2026-05-18 gate

- 上游仓库：官方 Medusa clone
- 本地路径：`/home/qhq/serverless_llm_baselines/vendor_new_baselines/Medusa_main_20260518`
- 当前上游基线 commit：`6581d2e5ec8fa4ecdabcdb50560982a78ea3ca89`
- 项目入口：`/home/qhq/serverless_llm_baselines/Medusa_project`
- 当前状态：2026-05-19 official build / LoRA / true-remote feasibility gate
  已闭口为“当前机器不可正式复现”。在补充 conda CUDA target include/library
  paths 后，`vllm._moe_C` 可编译，但 `vllm._C` 在
  `csrc/cuda_graph.cu(131)` 因 CUDA Graph API 签名不匹配失败；同时官方源码
  仍硬编码 `/home/zsx/spdk`、SPDK/DPDK/GDRCopy link flags 和 `/home/zsx`
  模型路径。本机缺少这些官方硬件/系统依赖，因此不能生成真实 remote LoRA
  `e2e_v3` replay。

### FaaScale / LambdaScale 2026-05-18 gate

- 上游仓库：official LambdaScale/FaaScale clone
- 本地路径：`/home/qhq/serverless_llm_baselines/vendor_new_baselines/lambda-scale_main_20260518`
- 当前上游基线 commit：`9db210fcb6979f7c1f73f9819a77e0edb6c5e343`
- RDMA helper path：`/home/qhq/serverless_llm_baselines/vendor_new_baselines/rdma-p2p_main_20260518`
- RDMA helper commit：`ed83237439d2103141fbc7c9b97f348055b6cb53`
- 当前状态：排在 Medusa 之后继续 official build / LoRA / true-remote gate。
  已知风险包括 RDMA assumptions、hard-coded multi-node paths、Llama-2-only
  model slicing and missing LoRA/adapter workload path。

### S-LoRA

- 上游仓库：`https://github.com/S-LoRA/S-LoRA.git`
- 本地分支：`main`
- 本地路径：`/home/qhq/serverless_llm_baselines/repos/S-LoRA`
- 当前上游基线 commit：`c1ddf488781ea7f551cd0bb07bfd097124c93411`
- 当前本地状态：通过外层 wrapper 接入公平实验链；本地工作树当前有少量
  official server 适配修改（detokenization/router/model_rpc），不要回退。
- 当前复现范围：`slora_official_cu118` 独立环境、官方 CUDA 11.8 extension、
  native `/generate_stream` replay、S-LoRA token budget guard、`e2e_v3` summary
  已接入当前 fair round；不修改上游核心 serving 机制。

### Punica

- 上游仓库：`https://github.com/punica-ai/punica.git`
- 本地分支：`master`
- 本地路径：`/home/qhq/serverless_llm_baselines/repos/Punica`
- 当前上游基线 commit：`591b59899f0a20760821785d06b331c8a2e5cb86`
- 当前本地状态：通过外层 wrapper 接入公平实验链；不直接修改上游源码。
  当前仅 `third_party/flashinfer` 子模块状态显示为 modified，不作为主表 baseline
  的当前工作面。
- 当前复现范围：已完成 Llama-2 7B 受限 quick replay；不作为覆盖全部 backbone 的主表 baseline

### SGLang

- 上游仓库：`https://github.com/sgl-project/sglang.git`
- 本地分支：`main`
- 本地路径：`/home/qhq/serverless_llm_baselines/repos/SGLang`
- 当前上游基线 commit：`7d7fdc13093ccc151ddb43a5e5a2e0017872464e`
- 当前本地状态：通过外层 wrapper 接入公平实验链；不直接修改上游源码

### vLLM

- 上游/安装来源：本机 `LLM_vllm0102` 环境中的 `vllm==0.10.2`，并保留源码镜像 `/home/qhq/vllm`
- 源码镜像：`https://github.com/vllm-project/vllm.git`，本地分支 `main`，
  当前 commit `afd089f231d714e7fd06b51e3bc7df7fe004c7f9`
- 本地项目入口：`/home/qhq/serverless_llm_baselines/vLLM_project`
- 当前本地状态：通过外层 wrapper 启动 standalone OpenAI-compatible API server。
  `/home/qhq/vllm` 本地 requirements/pyproject 文件存在修改，正式 fair harness
  仍以 `LLM_vllm0102` 环境和外层 wrapper 为准；不要把这些源码镜像修改误当成
  paper baseline 算法变更。
- 当前复现范围：shared trace / shared LoRA subset / e2e_v3 / lifecycle monetary cost 已接入正式 harness

### SkyPilot / SkyServe

- 本地曾克隆过 SkyPilot 源码（`https://github.com/skypilot-org/skypilot.git`，
  本地分支 `master`，commit `ce5970ae46269cff18a22caf8102bf7dab097bd0`），
  但当前不属于 formal many-LoRA main-table
  harness。
- 若未来重新纳入，必须重新完成 shared trace、shared adapter subset、
  `e2e_v3`、cost model 和 per-system project entry 的准入检查。

## 当前 baseline 工程的 Git 说明

当前 baseline 工程仓库主要跟踪：

- `README.md`
- `docs/`
- `environments/`
- `scripts/`
- `ServerlessLLM_project/`
- `SGLang_project/`
- `patches/`

不跟踪：

- `results/`
- `logs/`
- `models/`
- `repos/*` 的完整源码目录

后续若新增其他 baseline，上述记录方式继续沿用。
