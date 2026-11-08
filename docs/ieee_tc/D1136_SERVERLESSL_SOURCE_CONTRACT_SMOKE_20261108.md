# D1136：ServerlessLLM 源码合同 smoke 结果

2026-11-08。使用 D1135 的排他源码视图做 no-GPU 合同核验。没有启动 Ray、
ServerlessLLM 服务、远端工件服务或 GPU 回放；没有修改历史结果。

## 1. 门禁修正

在 baseline harness 分支
`codex/ieee-tc-serverless-contract-20261030` 提交了
`3e58a92`（`bind ServerlessLLM TC launcher to audited source view`）：

- launcher 门禁绑定到已提交的 `a6c5a4c` 后脚本
  `3e44cb3ed4daf65bd971db78e50ad727798c2de401f15f9453aab4f9c64a2d67`；
- backend 门禁绑定到 D1135 视图中的 fixed-output token-ID backend
  `04833c156f7f9770b3ade2110915bc0d7b54c8e861a83a5d8217df38219aa0f3`；
- repaired router 哈希保持 `2bcbba0d...`，没有改动路由策略。

门禁更新只修正“实际使用源码”与“检查常量”的不一致；没有放宽源文件检查，
也没有把旧 `/repos/ServerlessLLM` 视图混入新 baseline。

## 2. 运行环境中的 no-GPU 结果

使用正式 ServerlessLLM serving 环境
`/home/qhq/anaconda3/envs/sllm_vllm0102_newserverless_20260518`：

| 测试模块 | 结果 |
|---|---:|
| `tests.test_ieee_tc_serverless_launch` | 30 tests，全部通过 |
| `tests.test_ieee_tc_serverless_router` | 6 tests，全部通过 |
| `tests.test_ieee_tc_serverless_measurement` | 75 tests，全部通过 |

环境版本为 Python 3.12、PyTorch `2.8.0+cu128`、aiohttp `3.13.3`。此前系统
base Python 的 `pytest` 结果为 108 passed、2 failed、1 skipped；两个失败分别是
系统环境没有 `torch`，以及 aiohttp 3.11.10 下的 pooled-connection 诊断事件未
触发。相同 measurement unittest 在正式 serving 环境中 75/75 通过，因此这两项
不能归因于 ServerlessLLM 的 GPU 或调度机制。

## 3. 源码视图生成核验

以 D1135 视图作为 `native_source`，实际调用当前 harness 的
`prepare-measurement-view`（repaired router、`request_identity_v1` queue）并对
生成文件执行 `compileall`：

- measurement view 生成成功；
- `loader_policy_changed=false`；
- `routing_policy_changed=true` 仅表示已登记 repaired router；
- `performance_run_authorized=false` 保持不变；
- 无模型、adapter、trace、远端包或 GPU 进程产生。

这一步证明源码身份和 no-GPU 合同门禁已经可以在正式 serving 环境中闭合，但不
证明 ServerlessLLM 的 native loader、Ray worker、500-adapter LoRA 路径或共同
生命周期资源计量已经具备正式比较资格。

## 4. 下一项边界

下一步只做 source-view 绑定的低规模服务资格探针：先核对实际 worker、native
loader、fixed-length token-ID 回包、远端预发布工件读取和 teardown；探针通过后才
考虑一次完整的 7B W2 回放。若缺少共同到达时基或物理 GPU owner-union 账本，仍
保持 D1113 的 R2 诊断分类，不把结果写入 PrimeLoRA 主比较表。
