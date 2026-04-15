# ServerlessLLM Official Reproduction Environment

- Repo: `/home/qhq/serverless_llm_baselines/repos/ServerlessLLM`
- Suggested head env: `sllm_head_official`
- Suggested worker env: `sllm_worker_official`
- Current status: source cloned, isolated bootstrap scripts prepared

## Reproduction strategy

- 不在主项目环境 `LLM_vllm0102` 中直接安装
- 不强求官方 Docker Compose 路径
- 优先走官方 `single_machine` 文档里的非 Docker 路线
- 目标是在当前机器上最大程度复现其 Ray + store + serve 原型系统

## Local scripts

- 环境安装：
  - `/home/qhq/serverless_llm_baselines/scripts/setup_serverlessllm_envs.sh`
- 启动 head：
  - `/home/qhq/serverless_llm_baselines/scripts/run_serverlessllm_head.sh`
- 启动 worker：
  - `/home/qhq/serverless_llm_baselines/scripts/run_serverlessllm_worker.sh`
- 启动 store：
  - `/home/qhq/serverless_llm_baselines/scripts/run_serverlessllm_store.sh`
- 启动 serve：
  - `/home/qhq/serverless_llm_baselines/scripts/run_serverlessllm_serve.sh`
- 部署模型：
  - `/home/qhq/serverless_llm_baselines/scripts/deploy_serverlessllm_model.sh`
- 发请求：
  - `/home/qhq/serverless_llm_baselines/scripts/query_serverlessllm_chat.sh`

## Minimal local path

1. 建双环境：head / worker
2. 启动本地 Ray head
3. 启动本地 Ray worker
4. 启动 `sllm-store`
5. 启动 `sllm start`
6. 用 `sllm deploy` 部署一个小模型验证

## Current next step

- 执行 `setup_serverlessllm_envs.sh`
- 若依赖安装成功，再按上面的顺序启动单机最小复现链路
