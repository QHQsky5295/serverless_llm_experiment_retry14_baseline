# D1134：ServerlessLLM 源码身份与资格门禁核查

2026-11-08。仅做源码、测试与历史运行记录的离线核查；没有启动 GPU、没有
修改 ServerlessLLM vendor 工作树，也没有修改既有结果。本文记录的是基线
资格问题，不是性能结果。

## 1. 当前测试事实

在 `/home/qhq/serverless_llm_baselines` 执行：

```text
python -m pytest -q \
  tests/test_ieee_tc_serverless_router.py \
  tests/test_ieee_tc_serverless_measurement.py \
  tests/test_ieee_tc_serverless_launch.py
```

结果为 `25 failed, 85 passed, 1 skipped`。失败主要不是模型或 GPU 失败，而是
`prepare_ieee_tc_serverless_stack.py` 在源码身份检查阶段提前拒绝了待测视图。
其中一个独立失败来自 ingress 回收测试：
`PreReadyIngressTests.test_pooled_connection_close_after_selection_is_a_visible_single_failure`
得到空列表，而测试期望 `peer_closed_after_pool_selection`。

## 2. 哈希对照

| 文件/对象 | 门禁或历史登记值 | 当前 vendor 工作树 | 当前 vendor index | 解释 |
|---|---|---|---|---|
| `sllm/backends/vllm_backend.py` | `994c80ae9c6106a6469f9d8d5889132e5c2032608c4e497141c00127f91cc777` | `04833c156f7f9770b3ade2110915bc0d7b54c8e861a83a5d8217df38219aa0f3` | `994c80ae9c6106a6469f9d8d5889132e5c2032608c4e497141c00127f91cc777` | 工作树比 index 多出固定输出的 `return_token_ids` 消费和原生 `token_ids` 回传；这正是共同 fixed-length 合同所需的观测适配，不能在未核对前丢弃。 |
| `sllm/routers/roundrobin_router.py` | `2bcbba0d8a3fdf46354f75c1d8ca5bcd79e505f059fcb296bf30dbff176e8f6e` | `2bcbba0d8a3fdf46354f75c1d8ca5bcd79e505f059fcb296bf30dbff176e8f6e` | `0182bd4862c1c0e1c4bf5d00507704f05d3d5092c267b4670ff31a4c79b18843` | 工作树是已登记的 repaired router；index 是历史 pre-TC router。二者不是同一源码视图。 |
| `scripts/start_serverlessllm_stack.sh` | `9ded784fd6c1990dc9c0994d3ec7744b799cfbff830554ff22b85b8f9ddbb95d` | `3e44cb3ed4daf65bd971db78e50ad727798c2de401f15f9453aab4f9c64a2d67` | 同工作树 | 现值来自已提交的 `a6c5a4c`（将选定 `SLLM_REPO_ROOT` 传播到 tmux 子进程）；门禁常量仍是旧值，属于 harness 与已提交脚本之间的身份漂移，不是 vendor 工作树的未提交改动。 |

另一个目录 `/home/qhq/serverless_llm_baselines/repos/ServerlessLLM` 还有独立的旧
适配视图（backend `76a96e1f...`、router `6cdbfcd0...`）。D1103 的 manifest
明确使用的是 `vendor_new_baselines/ServerlessLLM_new_main_20260518`，因此不能
把这个旧视图的哈希或安装状态冒充 D1103 的源码身份。

## 3. 工作树改动的性质

当前 vendor 工作树相对 index 只有两个文件有未暂存增量：

1. vLLM backend 在构造旧版 `SamplingParams` 前移除协议层
   `return_token_ids`，并把已有 `RequestOutput.token_ids` 以可选字段回传。
   这不会改变解码、路由、缓存或 admission 策略；它只提供共同生成合同所需的
   原生 token 计数。2026-11-06 的兼容性审计已经用 3B 16 请求 smoke 证明
   该适配可完成请求且无 token fallback，但它尚未被写入一个干净、可复现的
   vendor commit。
2. repaired router 保留 ready-first 语义，并在请求取消发生于异步容量检查期间
   时回滚已占用的槽位。该修复已由 router no-GPU AST 测试覆盖；它不是
   PrimeLoRA 的调度策略。

因此，不能简单把工作树重置到 index 来“让测试通过”：这样会删除固定输出合同所
需的原生 token 观测，也会混淆 repaired router 与 pre-TC router。相反，也不能只
修改期望哈希而不保存源版本、补丁边界和提交身份。

## 4. 当前资格裁决

- 本次失败被分类为 **源码视图/门禁漂移**，不是 ServerlessLLM 的性能失败。
- D1103 的历史回放仍保留为 R2 诊断；D1113 已撤回其当前主比较资格，原因是
  回放时基、TPOT 边界和资源账本合同不一致，而不是本次 25 个 no-GPU 测试失败。
- 在源码身份没有形成一个排他、可哈希、可重建的 source view 之前，不启动新的
  ServerlessLLM GPU 回放；否则即使请求成功，也无法说明它对应哪一个 backend、
  router 和 launcher 版本。

## 5. 下一步唯一有效动作

1. 在不触碰现有 dirty vendor 工作树的前提下，建立一个排他的 ServerlessLLM
   source view：明确 official commit、已审计的 staged TC adaptations、
   fixed-output token-ID compatibility patch、repaired router patch，以及
   `a6c5a4c` 之后的 launcher 版本；为每个文件和生成视图保存 SHA-256。
2. 让 `prepare_ieee_tc_serverless_stack.py` 在该 source view 上做一致的身份检查，
   并重新运行上述 no-GPU 测试。若 ingress 失败仍存在，单独判断它是测试期望错误
   还是连接回收语义缺口，不能把它与源码哈希漂移合并处理。
3. 只有 source view、launch manifest、fixed-length token audit 和 teardown
   全部通过，才重新评估 ServerlessLLM 是否具备新的 GPU 回放资格；否则继续以
   D555 等已封存 R2 结果做诊断对照，并在论文中披露复现边界。

本页不改变任何旧结果、旧图表或 vendor 工作树，也不宣称 ServerlessLLM 或
PrimeLoRA 在当前阶段具有新的跨系统优越性。
