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
- 本地路径：`/home/qhq/serverless_llm_baselines/repos/ServerlessLLM`
- 当前上游基线 commit：`9f50241baa5386e06a9321c51f19a9ef5f964c2b`
- 当前本地 patch 文件：
  - [ServerlessLLM_local_changes.patch](/home/qhq/serverless_llm_baselines/patches/ServerlessLLM_local_changes.patch)

### S-LoRA

- 上游仓库：当前已本地克隆
- 本地路径：`/home/qhq/serverless_llm_baselines/repos/S-LoRA`
- 当前上游基线 commit：`c1ddf488781ea7f551cd0bb07bfd097124c93411`
- 当前本地状态：不修改上游源码；通过外层 wrapper 接入公平实验链
- 当前复现范围：shared trace / shared adapter subset dry-run harness 已通过；
  `slora_official_cu118` 独立环境、官方 CUDA 11.8 extension 与宿主 GPU 可见性
  preflight 已通过；下一步跑真实 500 请求 bring-up

### Punica

- 上游仓库：`https://github.com/punica-ai/punica.git`
- 本地路径：`/home/qhq/serverless_llm_baselines/repos/Punica`
- 当前上游基线 commit：`591b59899f0a20760821785d06b331c8a2e5cb86`
- 当前本地状态：通过外层 wrapper 接入公平实验链；不直接修改上游源码
- 当前复现范围：已完成 Llama-2 7B 受限 quick replay；不作为覆盖全部 backbone 的主表 baseline

### SGLang

- 上游仓库：`https://github.com/sgl-project/sglang.git`
- 本地路径：`/home/qhq/serverless_llm_baselines/repos/SGLang`
- 当前上游基线 commit：`7d7fdc13093ccc151ddb43a5e5a2e0017872464e`
- 当前本地状态：通过外层 wrapper 接入公平实验链；不直接修改上游源码

### vLLM

- 上游/安装来源：本机 `LLM_vllm0102` 环境中的 `vllm==0.10.2`，并保留源码镜像 `/home/qhq/vllm`
- 本地项目入口：`/home/qhq/serverless_llm_baselines/vLLM_project`
- 当前本地状态：通过外层 wrapper 启动 standalone OpenAI-compatible API server；不修改 vLLM 底层源码
- 当前复现范围：shared trace / shared LoRA subset / e2e_v3 / lifecycle monetary cost 已接入正式 harness

### SkyPilot

- 上游仓库：当前已本地克隆
- 本地路径：`/home/qhq/serverless_llm_baselines/repos/skypilot`
- 当前上游基线 commit：`ce5970ae46269cff18a22caf8102bf7dab097bd0`
- 当前本地状态：未检测到本地修改

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
