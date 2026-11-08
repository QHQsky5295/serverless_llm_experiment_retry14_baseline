# D1130：7B W1 RoutingHandoff 累积消融

2026-11-08，性能运行前登记。D1129 已完成并备份为
`82d68db475376b8ba92ae6ce5acb51bfd5162209`；不得重跑该键、替换 Full
或切换到 3B。此项是批准 PLAN §8 A5 的第二个缺失中间节点。

## 科学问题及对照

在 RoutingOnly 的服务分桶路由、普通按需缓存和相同扩缩容条件下，仅加入
激活 handoff，是否减少扩容首次服务中的工件准备等待，改善资源—质量关系？
预期有条件：被准备工件必须被后续请求使用，且准备能够及时完成；无用准备、
带宽争用和资源持有可能抵消收益。仅有 plan 计数不能证明收益。

累积链保持 ElasticOnly → RoutingOnly → RoutingHandoff → CapacityOnly → Full。
直接对照 D1129 RoutingOnly，补充复用 D1119 ElasticOnly、D1128 CapacityOnly
和 D1115 Full。原始结果、所有负结果及输出差异保留。不重新挑选对照，不以
削弱对照、改变预算或重跑同键获得有利排序。

这项自然闭环比较不保证各运行产生相同的激活数或准备选集；同目标时序效应
仍由受控 A2/A4 回答。全链的边际收益依赖加入顺序，不冒充独立普适贡献。

## 实现与配置边界

源配置是 `20261108_d1129_7b_routing_only_w1.yaml`。新配置
`20261108_d1130_7b_routing_handoff_w1.yaml` SHA：
`517c2367ddd55b4d8221d37d36cb7ed08efb0475b836dd098510be7aa86e2572`。
仅变更组合身份、handoff 开关、允许的准备范围及三个独占结果/缓存路径；
完整配置归一化相等性已测试。

- 身份 `routing_handoff_v1` / `tc_ieee_routing_handoff`，service_bins 不变。
- `file_cache_policy=demand_lru`，新增 `file_preparation_scope=handoff_only`。
  该范围传到实际文件 owner，子进程身份和准备快照可读回；不是控制端标签。
- 仅 mode=handoff 且带 activation_id 的 owned planning/文件登记可进入。
  登记保留 mode/activation_id，不能复用 plan_id 改变来源；复用现有任务、
  plan 和取消生命周期，不新建平行资源账本。合法 native adoption 不误判取消。
- 保留现有 Eq.(4)/(5) 剩余空间密度选择。稳态 residency、主动软 admission、
  legacy forwarding 和自动 HOST promotion 关闭，九公式未改。
- handoff 的保护目标不能被请求淘汰；其他合法无引用 LRU 仍可服务真实请求。
  可选准备不能借按需 LRU 未规划驱逐；空间不足沿已有事件驱动等待。真实请求
  加入同一传输后按 live required interest 合法回收，取消请求不永久升级权限。
  demand_lru 对 replacement_epoch 的禁令保留，不恢复 residency replacement。
- Full 默认准备范围仍为 all，旧三个 demand-only 组合仍为 none。原生 runtime、
  selector、autoscaler、生成合同、预算、安全保护和采样方法不变。

公共条件：Llama-2-7B、vLLM0.30.0 FP16/TP1、14slots/seqs、1024context/batch、
0.92utilization、prefix=false、D362 profiles、原 D1100 predictive autoscaler、
min1/max4、5秒窗口、27.43356142744625ms service bin。复用原 source42 的
4000 请求、500 逻辑 adapter 和 W1 映射；60秒共同部署通知，native fixed-output
greedy 合同。无新 trace/工件；真实远端使用已有只读预发布 gzip，无请求路径打包。

## 历史、资料及验证

已复核 `ROUTING_HANDOFF_COMPOSITION_REVIEW_20261108.md`、D1049/D1057/D1119、
P1 D47 和 D1063 相关历史，以及实际 stack、owner、runner 和既有 activation fixture。
本次再次读取 [vLLM0.30.0 官方 LoRA LRU 源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)：
正常按需缓存不能被当成主动管理删除。[HydraServe 原论文](https://www.usenix.org/system/files/nsdi26-lou.pdf)
的弱依赖重叠提供检验动机，不证明本机 LoRA handoff 收益，也不构成 pipeline 移植。
`ablation-planner` 的指定评审模型不可用，采用本地复核，标记
`[pending external review]`，不冒称独立审阅、不增加实验门槛。

60 项 CPU 检查通过，23.046秒，main/wrapper 均0，实际域已移除。涵盖：配置单因素、
实际 launch owner、旧三个组合和 Full、前台 LRU 与可选准备、同 writer 合并及取消、
初始化先后/激活 adoption、真实 spawn 文件 owner。它们不是 GPU 数值正确性或性能结果。
两个前序未成功测试尝试完整说明见同目录 `prelaunch_tests.md`；不因环境中止扩大内存
上限或重做无关 GPU 资格。正式推理环境未替换。

## 唯一执行键

```bash
bash scripts/ieee_tc/run_d1100_7b_busy_service_rate_w2.sh W1 \
  /home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1130_20261108/7b_routing_handoff_w1 \
  d1130_7b_routing_handoff_w1 \
  /home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1130_20261108/7b_nvme_routing_handoff_w1 \
  /dev/shm/tc-d1130-7b-routing-handoff-w1 \
  configs/ieee_tc/20261108_d1130_7b_routing_handoff_w1.yaml tc_ieee_routing_handoff
```

tmux `tc-d1130-routing-handoff-7b-w1`。只执行一次：实际启动前重读完整 PLAN、核对
三路径未占用、GPU compute 为空、MemAvailable≥102GiB、磁盘≥150GiB、所属远端
服务及链路。沿既有服务72/80GiB、swap2GiB，辅助3/4GiB、swap0 和原 watchdog。
运行停止线16GiB/100GiB；不修改共享 k3s。运行时不测试/压缩/推送/改 serving。
144个旧投稿文件、三个用户 dirty 文件、历史输入与 raw 结果不变。

## 收尾与贡献判断

终态实际释放后保存同 clock 远端 journal，复用 D1113/D299/D455 显式 scenario 接口。
核查4000个输入及原生生成合同、所有实际 dispatch tier、真实生命周期资源积分、
handoff 的计划/实际开始/完成/被使用、自然 activation 和 first-service。
稳态 residency 必须为0；soft admission 关闭但物理检查保留。

沿冻结 V1、单轮 V2、D394 SLO、D594 W1预算；PLAN SHA
`b5170643aa5d332efc67407af10ab7022a2d68fd802e2389d253968bd92dcf8f`。
全部变体展示联合达成率、GPU-s/offered 和尾延迟，95%不是纳入门槛；n_correct 未知
保持 null，输出 digest 不同不能自行判正确/错误。G1条件资源、G2同预算质量和完整
资源—质量关系分开，不把更少资源/更差尾延迟写成全面最优。首个完整结果封存，单轮不报CI。

本轮只补这一累积节点；其后先统一审视当前7B A1/A5贡献边界，再推进受控A2/A4、
S2/S3、M2/S5及其余敏感性。3B同版本缺项、成本/CE/Pareto、审稿文档、IEEE图表和
最终V2交付全部保留。尚未启动本项性能回放，不关闭总目标。
