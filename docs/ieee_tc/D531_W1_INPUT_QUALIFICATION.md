# D531：W1 共同输入与分析入口资格

2026-10-15。继续 D530 主线，不是新优化候选，也不是一次性能结果。
复用 `run-experiment` 的既有入口、原始工件与 vLLM 0.30 环境。

## 发现和必要修正

PrimeLoRA 的 `FrozenReplayPlan` 已支持计划中的 W1，但 HTTP 基线辅助入口
此前固定读取 W0。仅修改标签会导致不同到达条件被混作同一场景。
本次新增显式 `trace_w1_v1`：只允许完整、SHA 绑定的 4000 行源输入，直接
调用同一个 `FrozenReplayPlan.load(profile='W1')`，不另实现一套变换。
拒绝短前缀、额外速率缩放、冲突的 profile 标签和更大的源 trace 前缀。
原 W0 和 500-ID 功能覆盖入口保持原语义。

分析器新增显式 `full4000_w1`，核验源/视图 SHA、完整 offered 集合及生成
绑定。原 `full4000` 仍严格只接受 W0。统计公式、SLO、物理资源积分、失败
及独立数值正确性未知规则不变。W1 不可能用 W0 的预算参考替代。

实际入口的 publisher、service 及 physical ledger 均调用同一个 HTTP view
loader；PrimeLoRA 则从 launch context 重新加载并严格匹配同一个 view。
D533 在既有 runner 增加固定 W1 case；其 YAML 与 D476 仅输出/cache 路径
不同。D476 的原始 service/preparation profile 经实际 ScenarioRunner 路径
加载通过，元数据候选保持关闭。未修改九式、控制参数或后端机制。

## 已完成的 CPU 与真实输入核验

- 4000 行完整源，132 个实际 adapter ID，458224 个目标 token；所有源行、
  adapter、canonical prompt、原生 prompt token SHA、生成 body 与 W0 相同。
- 实际源 SHA：`efb903254fcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c`。
- W1 到达窗口 2186.9921293258667 秒；按独立公式逐行核验八个500请求阶段，
  phase 内时间减半，七个 phase 间隔精确30秒。不等待服务 drain、不清缓存、
  不根据 Full cooldown 改间隔、不额外添加末尾空闲。
- HTTP 与 PrimeLoRA 的现有两份 trace 路径不同，但源 SHA、4000 行内容、
  到达偏移及 view SHA 完全相同；保留路径身份，不复制新 trace。
- vLLM 原生配置 SHA 保持
  `361dc7645a8eef8ee304addfaa9e1611093dae1c6678b108970c1adc2e7fe3a5`，
  DP4/TP1、原 ordinal round-robin、LoRA cache 与原公开工作点不变。
- 103 项 CPU 测试通过，无失败/跳过；GPU 未初始化。这些检查不代替实际
  replay-ready 与服务时间线核验，不是 GPU 性能或完整数值正确性证明。
- 辅助内存峰值1019711488 B，high/max/OOM/swap为零；实际递归进程为空、
  同身份资源域关闭、路径消失。147个旧投稿保护文件和三份用户修改未变。

## 与官方测量资料的关系

本次核查 [vLLM 官方 benchmark 文档](https://docs.vllm.ai/en/latest/cli/bench/serve/)：
普通 request-rate 可合成到达，max-concurrency 可能改变实际请求率，self-timed
才保留 trace 时间。这里沿用项目既有 open-loop 定时回放，不换用随机到达、
不另加客户端并发闸门；官方文档仅支持测量边界，不证明本项目实现正确。

## 下一执行键与证据边界

1. D532：原 vLLM 冻结配置的一次完整 W1，预登记共同 Resident 参考候选。
   新 RAW 为 `results/ieee_tc/vllm_qualification/d532_20261015/`；仅 HTTP view
   和实验角色改为 W1。远端只读发布缓存、60秒通知、资源包络和生成不变。
2. 保存/审计真实释放与完整4000合同后，D533：D476 Full 的一次 W1。
   新 YAML `configs/ieee_tc/20261015_d533_7b_full_w1.yaml`，既有 runner case d533。
3. 后续按完整计划推进其余已发表系统、W2/G2及证据驱动优化。没有新性能
   结果前不声明预算达标、PrimeLoRA领先或7B完成。

原始资格：`results/ieee_tc/main_comparison/d531_20261015/`；发布记录：
`paper_results/ieee_tc/main_comparison/d531_w1_input/`。D532/D533 此时仅准备，
尚未启动；不重跑此前100/500/1000 GPU资格、W0或HydraServe基础适配。
共享 k3s 保持未触碰。单轮、未知数值证明、全部失败保留与完整剩余范围不变。
