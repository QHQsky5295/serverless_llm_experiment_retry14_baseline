# D647：3B共同warm的batch8可行性验证

2026-10-17。一次原生GPU探测，使用D646已冻结的旧输入索引。此实验只验证
共同warm标定的容量与执行合同，不是Full回放，不产生SLO阈值或G1/G2结论。

## 实测结果

| 输入组 / 选择位置 | 不同adapter数 | 完整上下文所需KV blocks | 可用KV blocks |
|---|---:|---:|---:|
| 0 / batch 1 | 8 | 253 | 5,830 |
| 0 / batch 20 | 8 | 286 | 5,830 |
| 1 / batch 14 | 8 | 425 | 5,830 |
| 1 / batch 28 | 6 | 512 | 5,830 |

每block为16tokens；各批8请求。4/4批、32/32请求均通过原生token、prompt
SHA、adapter引用、计时与终态检查。批前adapter已确认GPU-ready，无前批
未退役工作；批后引用释放和scheduler drain通过，最终GPU/CPU cache清空。
因此当前配置可继续按既定batch8标定，无需降batch或放宽协议。
此探测覆盖选样中的最大KV与最大adapter数批次，不声称证明任意负载容量。

单GPU从实际分配至释放73.991899s，属于此次资格成本，不是GPU-s/request主
性能指标。服务内存峰值7,015,247,872B、最低主机可用105,769,992,192B；
high/max/OOM/OOM-kill/swap均0，无watchdog告警/中止。实际worker释放已核验。

## 方法与复用

沿用既有`native_warm_reference`，配置为D641当前3B、vLLM0.30.0、FP16/TP1、
prefix-cache关闭。原生输出固定长度、greedy，停止规则未修改。官方vLLM文档
说明`ignore_eos`用于继续生成而不因EOS提前停止；本实验同时核验真实输出
数量，不能仅凭配置声称工作量相同：[vLLM 0.30 SamplingParams](https://docs.vllm.ai/en/v0.30.0/api/vllm/sampling_params/#vllm.sampling_params.SamplingParams.ignore_eos)。
未执行PrimeLoRA路由/planner/admission，也无远端服务、传输或动态打包。

仅扩展已有warm analyzer的显式输入名、索引与probe/measure接口；原7B默认
参数不变。请求/批次覆盖从冻结索引推导，不再硬编码7B的两组样本数，仍用原
`completed_warm_length_samples`和native计时/引用校验。新增1/2/4组计数、
probe不冒充measurement与非法phase测试；17项CPU测试全部通过。
没有修改被测算法、原生collector或阈值计算式。

## 执行身份与保护

- 执行commit：`ab8be4c57b19569f82feba5a83a5c4656ce2ca97`。
- source1 SHA：`8a34dfcd1cb8b533018fe0cb7c0763010ae06c9308f7c123410f1661780658c9`。
- 原始证据：`results/ieee_tc/p2_backend_qualification/d647_20261017/`。
- 管理session15869、分析28013均exit0；start/analysis scope及原生launcher
  递归空、路径移除。分析峰值75,517,952B，内存事件/swap0。
- 147项当前保护清单通过，用户manifest不变。旧最初清单保留已披露的baseline
  replay脚本修改，并验证其既定SHA；不将该历史差异说成本次更改。
- 大原始JSON不入Git；小型资格回执与执行证据单独备份。没有新图、论文修改、
  7B重复实验、外部baseline优化或无关k3s操作。

下一步直接D648：同索引/配置完成16预热＋1,536测量，再按冻结V1独立计算和
封存3B共同TTFT/TPOT阈值。随后Resident参考、3B Full W0/W1/W2及完整对照；
不得以本probe替代共同标定或全部3B性能实验。完整消融/敏感性/审稿交付保留。
