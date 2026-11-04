# D1057：ElasticOnly 入口与统计支持已实现，性能回放尚未启动

2026-11-04。本次继续完整 PLAN 的 A1/A5，不新增外部 baseline、不改变
G1/G2、不重新运行 D1047 Full。上一次完成的是 D1058 指标取舍分析；
本次恢复机制消融主线。完整目标仍 active。

## 研究问题与本次进展

ElasticOnly 检验“相同弹性控制下，主动 adapter 管理整体是否有额外收益”。
这是复合消融，不能独自证明 confirmed-state propagation。它沿用 D1047
的 3B W1、1–3 实例、vLLM 0.30.0、输入和生成、D1031 profiles、D648 SLO、
D657 预算、物理资源持有计量及共同资源包络。

- 复用已有 owned `load_only`；不切换到另一套 legacy 资源管理路径。
- 明确关闭 activation handoff、主动协调/分层规划/软 admission、主动
  forwarding/HOST promotion；文件层采用普通 demand LRU。
- 保留物理容量检查、原子 reservation、引用与取消、原生按需缓存以及
  已到达请求驱动的来源准备。没有按 phase 清缓存或人为削弱按需加载。
- 新身份为 `elastic_only_v1` / `tc_ieee_elastic_only`，不能冒用 Full 或
  ElasticLocality。Full 的默认行为、九公式及已有 ElasticLocality 合同不变。

依据沿用入口审计，并再次读取固定版本的
[vLLM 0.30.0 按需 LRU 实现](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)。
关闭主动管理不代表取消正常缓存；该源码是设计边界参考，不是性能证据。
历史 D453 仅复用实现经验，旧 7B W0 数值不混入本轮 3B W1。

## 实现和验证范围

1. 现有 runner 扩展组合身份及实际 router/file-owner 检查；旧 Full 默认不变。
2. 现有 D1000 launcher 增加显式 `FAASLORA_D1000_SCENARIO`，默认仍为
   `tc_ieee_full`，不复制启动框架。配置三个工作路径与实际 launcher 一致。
3. 现有 D703 的 SLO、阶段、profile 和收尾分析函数增加 keyword-only
   `scenario`，默认 Full。新分析记录保存原始 scenario，并核对原始 metadata、
   detailed key/name、配置及分析身份。旧无 scenario 的 D714 记录仅允许
   按其历史 Full 身份读取，不回写原始结果。
4. 复用原 CPU fixtures：20 项通过，覆盖 Full/ElasticLocality 回归、
   ElasticOnly 组合、真实控制入口扩容且无主动准备、按需引用/释放。
   同时修正既有测试漏 `await`，不是生产算法补丁。
5. 统计入口的 25 项针对性测试通过，shell 语法通过；19 条现有依赖
   deprecation 警告保留在日志。两个测试资源域无 high/max/OOM/swap。
   没有把 CPU fixture 当作 GPU 性能或机制实际触发证据。

配置：`configs/ieee_tc/20261104_d1057_3b_elastic_only_w1.yaml`。
证据：`paper_results/ieee_tc/ablation/d1057_3b_elastic_only_w1/prelaunch/`。
原始测试记录：
`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1057_20261104/`。

## 用户本次集群答复的处理

现有 k3s 不是本项目集群。复核 `HYDRASERVE_ISOLATION_DECISION_20261012.md`
以及 D807/D808/D809：独立 rootless Podman 的必要适配和三种 3B 完整回放
已有证据。本次不申请该共享集群 kubeconfig，不接管、重启或改权限/网络，
不重复部署 HydraServe。保留官方专用后端以及适配范围；这些运行都是 pp1，
不据此声称验证了 pipeline/consolidation。既有异常与适配限制继续披露。

## 下一动作和运行身份

先提交并非 force 备份本检查点；随后在原入口新鲜核查磁盘/内存/GPU 和
远端 health，再启动唯一一次完整 3B W1 ElasticOnly。没有要求重新进行
500-ID 资格、无理由 smoke 或 Full 重跑。执行命令为：

```bash
cd /home/qhq/serverless_llm_experiment_retry14_baseline
tmux new-session -d -s tc-d1057-elastic-only-3b-w1 \
  env FAASLORA_D1000_RUN_ID=d1057_20261104 \
      FAASLORA_D1000_RUN_TAG=d1057_3b_elastic_only_w1 \
      FAASLORA_D1000_CONFIG=configs/ieee_tc/20261104_d1057_3b_elastic_only_w1.yaml \
      FAASLORA_D1000_SCENARIO=tc_ieee_elastic_only \
      FAASLORA_D1000_WORKLOAD=W1 \
      FAASLORA_D1000_HOST_DESCRIPTOR=fresh_d747_v1 \
  bash scripts/ieee_tc/run_d1000_3b_route_confirmed_arrival_prefetch_w1.sh
```

本页是待执行命令，不是已启动回执。运行期间只作轻量监测，不测试、构建、
压缩、推送或改远端。收尾复用 D1056 collector，四个新参数均显式传入
`scenario='tc_ieee_elastic_only'`；请求/机制分析使用相同原始 key，不能改名
JSON。继续记录 actual routing、handoff、tier/admission 触发与缺失项。

当前无新性能数值。source42 已知开发负载、单轮 V2、全量非零 LoRA 数值
正确性证明缺口及 `n_correct=null` 均保留。负结果、错误和局限不得隐藏。

ElasticLocality、LastKnown、NoHOST、7B A1/两个累积节点、受控 A2–A5、
S1–S13、成本/CE/Pareto、B1/B2/C1/C2/C3/推导与两轮审稿文档、最终 IEEE
图表及 FaaSLoRA V2 集成全部仍在范围内。检查点备份不等于最终 V2 集成。
两个既有用户 dirty 文件不编辑、不 stage；本页承担当前交接，保留旧状态
文件的原字节。
