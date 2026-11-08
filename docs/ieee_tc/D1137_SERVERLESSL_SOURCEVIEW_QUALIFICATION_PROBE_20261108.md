# D1137：ServerlessLLM 排他源码视图资格探针

2026-11-08。沿用 D1135 源码视图和 D1136 门禁修正，完成一次 7B、16 请求的
qualification-only GPU 探针。结果不进入主比较表、不替代 4,000 请求回放，也不
作为 PrimeLoRA 优越性证据。

## 1. 运行身份

| 项目 | 值 |
|---|---|
| 源码视图 | `/home/qhq/serverless_llm_baselines/results/paper_experiments/17_serverlessllm_source_views/d1135_vendor_worktree_v1` |
| 源码视图 manifest SHA-256 | `354cf79d51995fe40dcab7bd73490753935d385f1e11eefe18bf79b3dc38819d` |
| harness 分支提交 | `3e58a92`（仅更新源码身份门禁） |
| 模型 | Llama-2 7B，TP1，RTX 3090 |
| 输入 | D1103 W2 trace，SHA `7fd426bf7d98bdabe973b0a39c3d990d80b252ee093bf7e5076f31e882c82767` |
| adapter subset | 既有 500-adapter subset，SHA `aa94b21e1295aefde664e5b29c030a9e42e02af946369bfe9ae15336b89b3016` |
| 生成合同 | `fixed_length_greedy_v1`，vLLM native token IDs，目标上限 256 |
| 远端交付 | 项目自有 192.168.4.174:18081，`prepublished_gzip_v1`，请求路径无打包 |
| 实际回放 | 16 条（原始 trace 总长仍为 4,000；使用 `REPLAY_MAX_REQUESTS=16`） |

## 2. 合同结果

| 检查 | 结果 |
|---|---:|
| 请求完成 | 16/16 |
| HTTP/终态成功 | 16/16；0 failed |
| `output_contract_match` | 16/16 |
| completion token source | `vllm_token_ids`，16/16 |
| trace token fallback | 0 |
| 远端工件传输 | 8 次，20,200,499 bytes |
| 栈/GPU 清理 | 完成；结束检查无 GPU compute process |

远端服务在探针前由项目自有 user unit 启动，`/health` 和 `/manifest` 经 bearer
认证返回 200；探针结束后已停止，端口 18081 不再监听。

## 3. 观察值（仅资格诊断）

| 指标 | 16-request observation |
|---|---:|
| 平均端到端 TTFT | 43,553.3 ms |
| P95 端到端 TTFT | 54,929.8 ms |
| 平均 service TTFT | 332.7 ms |
| P95 service TTFT | 518.4 ms |
| 平均 dispatch/admission wait | 43,220.6 ms |
| 平均 TPOT | 25.6 ms |
| E2E P95 | 58,342.4 ms |
| TTFT SLO ≤5,000 ms | 0/16（qualification probe） |

端到端长尾几乎由首轮 scale-out/dispatch 造成；service TTFT 与 D1103 的历史
量级相近。这验证了 source view 能实际启动、加载 vLLM LoRA、读取远端预发布
工件并返回原生 token IDs，但不支持把该 16-request 观察当作 workload 结果：
样本太小、到达窗口被截断，且 lifecycle GPU 账本仍是 ServerlessLLM 自身的
部署账本，尚未映射为 PrimeLoRA 共同物理 owner-union 合同。

## 4. 产物与裁决

结果目录：

```text
/home/qhq/serverless_llm_baselines/results/paper_experiments/17_serverlessllm_source_views/d1138_probe_results/
```

关键 SHA：

- replay JSON：`3972eb641a51d72b304645b0432525a6b4144e94bcd1a0c04373f7809d2ec667`；
- summary JSON：`b32c6a14984a0cd34f0d32f5ef1845e4cebbbdf1b07640b20b6e9e000a5f8e8a`；
- deploy JSON：`d5fe1620241dfc6391af3df41363206ded6da1f2c21c9f58d118b829949995af`。

资格裁决：

1. ServerlessLLM 的源码视图、fixed-length/native-token 合同、远端 HTTP 交付和
   teardown 在低规模探针中闭合；可以进入下一阶段的 7B full replay 准备。
2. 该探针不解除 D1113 对旧 D1103 的 R2 分类，也不提供共同 G1/G2 资源—时延
   比较点。正式回放必须使用排他 source view、共同 4,000-request 输入、完整
   physical/resource ledger 和单独的 deployment/dispatch 分解。
3. 不因本次长尾而预热、固定最小副本或调整 ServerlessLLM 策略；这些会改变
   baseline 语义。下一项工作是登记 full W2 的唯一执行键和资源门禁，而不是重复
   qualification probe。
