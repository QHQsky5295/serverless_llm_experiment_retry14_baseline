# D1135：ServerlessLLM 排他源码视图建立

2026-11-08。根据 D1134 的裁决，在不修改 dirty vendor 工作树、不覆盖历史
结果的前提下，建立一个只读、可哈希的源码视图。此步骤仍未启动 GPU，也不授予
新的性能回放资格。

## 1. 源码视图

```text
/home/qhq/serverless_llm_baselines/results/paper_experiments/17_serverlessllm_source_views/d1135_vendor_worktree_v1
```

视图由 vendor 工作树的当前完整内容复制得到，排除了 `.git`、`__pycache__`
和 `.pyc`。同时保存了三个来源补丁：

| 证据 | SHA-256 |
|---|---|
| `SOURCE_VIEW_MANIFEST.json` | `354cf79d51995fe40dcab7bd73490753935d385f1e11eefe18bf79b3dc38819d` |
| `source_patch_from_HEAD.patch` | `7e40c996a6f9c0f3b8324d54d45b276752576ad43f3e58dbbb3d39b46e2e99eb` |
| `source_patch_index.patch` | 见 manifest |
| `source_patch_worktree.patch` | 见 manifest |

源仓库官方 HEAD 为 `9f50241baa5386e06a9321c51f19a9ef5f964c2b`（`fix: release
pipeline (#309)`）。视图保留了当前已登记的 staged TC adaptations，并保留
工作树中 fixed-output token-ID 适配与 repaired router 的未提交增量；没有把这些
增量静默折叠进官方 commit。

关键文件哈希为：

| 文件 | SHA-256 |
|---|---|
| `sllm/backends/vllm_backend.py` | `04833c156f7f9770b3ade2110915bc0d7b54c8e861a83a5d8217df38219aa0f3` |
| `sllm/routers/roundrobin_router.py` | `2bcbba0d8a3fdf46354f75c1d8ca5bcd79e505f059fcb296bf30dbff176e8f6e` |

## 2. 视图边界

- backend 的 `return_token_ids` 是协议层观察字段：它在旧 vLLM
  `SamplingParams` 构造前被消费，回包只暴露已有 `RequestOutput.token_ids`；
  不改变解码、调度、LoRA 加载或 admission。
- router 的 repaired 版本保留 ready-first 与 round-robin，只处理取消期间的
  槽位回滚和空 ready 集合等待；不引入 PrimeLoRA 策略。
- 视图使用真实远端工件合同的已有预发布包语义；本步骤没有打包工件、生成新
  权重、生成新 trace 或复制 adapter pool。
- `performance_authorized=false` 保存在 manifest 中。视图只能用于源码身份与
  no-GPU 合同核查，不能直接作为正式 baseline 结果。

## 3. 下一项核查

1. 在临时测试副本中将该视图作为 `native_source`，并把主 harness 的 launcher
   哈希绑定到已经提交的 `a6c5a4c` 版本；不得修改 dirty vendor 或历史结果。
2. 单独复核 ingress 回收测试的 aiohttp 事件语义；如果它仍失败，判断是测试
   环境/库版本不兼容还是 ingress 合同缺口，不把它归因于 ServerlessLLM 性能。
3. 只有 source view、launcher manifest、fixed-length token audit、无 fallback
   和 teardown 全部通过，才决定是否重跑 ServerlessLLM。若不能形成共同生命周期
   账本和到达时基，仍按 D1113 的 R2 诊断边界处理，不将旧结果升级为主比较。

本页不改变 D1103、D1106、D1108、D1113 的裁决，不覆盖任何旧图表或结果。
