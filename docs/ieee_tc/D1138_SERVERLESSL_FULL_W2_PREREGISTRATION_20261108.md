# D1138：ServerlessLLM 7B W2 full replay 预登记

2026-11-08。D1137 已证明排他源码视图可以完成低规模资格探针；本页只登记下一
个唯一的 full replay 执行键，尚未把它作为结果发布。执行仍遵守 single-run 规则：
不筛选 seed、不按结果改参数、不重复同一执行键。

## 1. 固定输入与源码

| 项目 | 固定值 |
|---|---|
| 执行键 | `d1138_20261108_serverlessllm_7b_w2_sourceview_full_v1` |
| 系统显示名 | `Serverless`（内部身份 `serverlessllm_new`） |
| native source view | `/home/qhq/serverless_llm_baselines/results/paper_experiments/17_serverlessllm_source_views/d1135_vendor_worktree_v1` |
| source-view manifest SHA | `354cf79d51995fe40dcab7bd73490753935d385f1e11eefe18bf79b3dc38819d` |
| backend / router SHA | `04833c156f7f9770b3ade2110915bc0d7b54c8e861a83a5d8217df38219aa0f3` / `2bcbba0d8a3fdf46354f75c1d8ca5bcd79e505f059fcb296bf30dbff176e8f6e` |
| model | Llama-2 7B `llama2_7b_main_v2_publicmix`，TP1，4-GPU scale-out envelope |
| trace | D1103 W2 trace，SHA `7fd426bf7d98bdabe973b0a39c3d990d80b252ee093bf7e5076f31e882c82767` |
| adapter subset | existing 500-adapter subset，SHA `aa94b21e1295aefde664e5b29c030a9e42e02af946369bfe9ae15336b89b3016` |
| requests | 4,000 offered requests，seed 42；不截断、不重排 |
| generation | `fixed_length_greedy_v1`，temperature 0，native vLLM token IDs |
| remote endpoint | project-owned `http://192.168.4.174:18081`，prepublished gzip artifacts |
| remote packaging | request-path packing 0；only prepublished read-only delivery |
| requested transfer metadata | 250 MiB/s label retained for traceability; HTTP path reports achieved physical throughput and does not inject a fictitious delay |

## 2. 结果目录与门禁

新结果只能写入：

```text
/home/qhq/serverless_llm_baselines/results/paper_experiments/17_serverlessllm_full_w2/d1138_20261108_serverlessllm_7b_w2_sourceview_full_v1/
```

启动前必须确认：

- 4 张 RTX 3090 空闲、内存与磁盘安全线通过；
- 远端 7B unit active、`/health` 与 `/manifest` 鉴权通过；
- source-view manifest、trace/subset SHA 和 baseline harness commit 可读；
- 完整结果路径不存在；
- ServerlessLLM stack 使用唯一 tmux/session 名并由 runner 自动 teardown。

## 3. 不改变的基线语义

- 使用 ServerlessLLM 官方 release pipeline、已登记 repaired router、原有
  autoscaler、round-robin、store/worker lifecycle 和 vLLM 0.10.2 环境；
- 不固定最小副本、不预热、不删除冷启动、不替换 router，不向 baseline 注入
  PrimeLoRA 的 readiness/admission 逻辑；
- 远端工件采用真实 HTTP 传输；服务端仅读取预发布 gzip，不在请求路径打包；
- 全量 run 的主报告只可使用 4,000/4,000、native token contract、共同 trace
  identity、dispatch/service/decode 分解和可审计资源记录；qualification probe
  与 D555/D1103 旧结果不混入。

## 4. 资格与发布条件

若 full replay 未达到 4,000/4,000、100% native output contract、无 fallback、
完整 request/adapter/target/prompt identity 或 teardown 证据，则保留失败并归为
R2/diagnostic，不将不完整数据放入 PrimeLoRA 主比较表。即使运行完整，也必须先
完成共同生命周期账本与到达时基审计，才能决定是否作为 G1/G2 参照点。
