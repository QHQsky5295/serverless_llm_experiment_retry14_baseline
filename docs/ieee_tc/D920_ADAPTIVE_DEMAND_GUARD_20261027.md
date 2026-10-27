# D920：7B IEEE 扩容的当前需求守卫候选

日期：2026-10-27。性质：单一候选的代码与配置资格检查，尚未产生新的
GPU 性能结果。

## 1. 动机与可证伪假设

D917 是当前代码、真实远端发布工件、固定长度生成合同下的 7B W2 完整运行：
4,000/4,000 请求完成，平均 TTFT 1,124.790 ms，P95 TTFT 2,672.268 ms，
物理生命周期占用 11,507.847 GPU-s。其控制事件中，最早三次扩容均发生在
`queue_depth=0`、`active_requests=0`，而 `p95_ttft_ms=13,669.073` 使
TTFT 归一化压力超过 1；随后才出现由 active/queue 压力触发的扩容。该证据
支持一个窄的、可证伪假设：历史端到端 TTFT 窗口在当前工作已完成时不应单独
启动新的物理副本，否则会把已结束请求的延迟压力转换成空闲 GPU 生命周期。

候选策略不是把 7B 的副本上限改成 3。配置仍使用四张可用 GPU 作为物理
`max_instances=4`，实际副本数由 IEEE 控制器的 live queue、active capacity、
TTFT 窗口和已在途 activation 决定。候选只在以下条件同时成立时抑制一次
扩容执行：

1. IEEE max-score 已提出扩容；
2. queue 和 active 两个归一化压力均未越过 1，唯一超限分量是 TTFT；
3. 当前 live work 与已有 ready/pending capacity 的 ready-time projection
   已覆盖；
4. handoff projection 没有等待到达新副本的请求。

任一 queue/active 压力越界均沿原 IEEE 扩容路径执行。没有读取未来 trace 行，
没有修改九个公式、SLO、预算、工件、负载或后端。

## 2. 实现边界

论文规范语义保持为 IEEE §3.1 的 max-score 扩容和 all-low/cooldown 缩容。
`IEEEReplicaControl` 仍先计算原始 score；`ScenarioRunner` 只在动作执行前
调用已有的当前需求/ready-time handoff 投影，将本次动作标记为
`no_action_current_demand_covered`。控制事件中额外保存
`scale_up_guard=current_demand_projection` 与 ready-time queue 数，便于审计
“提出了扩容但为何没有激活”。

## 3. 资格检查

- `tests/test_basic_smoke.py` 新增两个边界测试：TTFT-only 压力且当前容量已
  覆盖时抑制；queue 压力越界时不抑制。
- 目标测试通过：3/3。
- `python -m py_compile scripts/run_all_experiments.py` 通过。
- 候选配置 `configs/ieee_tc/20261027_d920_7b_adaptive_demand_guard_full_w2.yaml`
  与 D917 共享模型、trace、adapter、远端目录、生成合同、SLO 和控制阈值，
  只新增 guard 开关、独立 output/NVMe/HOST 路径；`max_instances=4` 与
  `hardware.gpu_count=4` 一致。
- 更大 IEEE 集成测试在本机已有环境限制下于启动阶段因 NVML 不可用而停止；
  未产生 GPU 结果，未将该环境错误归因给候选实现。

## 4. 下一步判定

按既定安全包络、GPU idle/cleanup、真实远端服务冻结和单轮执行键运行 D920
一次。结束后先做资源释放、合同/计量审计和控制事件统计，再决定是否保留：

- 若正确完成、共同 TTFT/TPOT 条件与物理 GPU-s 同时改善，候选进入 PrimeLoRA
  开发基线；
- 若只减少 GPU-s 但尾延迟或联合 SLO 明显恶化，保留为 trade-off 证据并回退
  到 D917 语义；
- 若候选没有触发 guard，说明 D917 的触发条件不再出现，不据此声称性能改进。

不重复运行 D917，不按结果修改 guard 规则，不用本次结果选择对手或改变共同
比较协议。
