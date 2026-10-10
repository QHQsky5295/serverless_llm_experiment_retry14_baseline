# D405：ServerlessLLM 全工件覆盖缺口与补测边界

2026-10-11。承接 D404；本记录为历史复用与 CPU 验证，不是新性能结果。
使用 analyze-results 先区分源数据、观测与推论；run-experiment 要求继续复用
现有 runner，未重建实验框架。github-sync 用于分别保存两个仓库的检查点。

| 证据 | 完成情况 | 实际 adapter 数 | 判定 |
|---|---:|---:|---|
| 历史 clean 7B | 4000 次报告成功 | 132 | 复用为历史解释，不是当前合同资格 |
| 历史 earlier 7B／3B | 各4000 次报告成功 | 各132 | 保留各自身份，不替代 clean 7B |
| D404 | 100 次原生合同通过 | 29 | 已完成，不重复 |
| 原 trace 前500／1000条 | 仅输入索引 | 46／60 | 均不覆盖500工件 |

原 trace 全4000条实际只出现132个逻辑 adapter。名义500是可访问集合，不能
写成500个都被引擎执行。D80的500-ID远端交付证明仍有效，但不补齐推理证据。
逐源SHA及集合见 `paper_results/ieee_tc/baseline_audit/d405_coverage_inventory.json`。

现有 baseline helper 增加显式功能检查视图：原 trace 前500条的 prompt、
到达、request ID、输出目标完全复用，按冻结静态清单排序逐项绑定500个工件。
每条保存原始 row SHA/adapter、派生 row SHA、pool SHA；独立 profile 标明
不是 W0、不进入性能比较。不写新 trace，不改权重和原始负载文件。
发布器、资源账本和超时使用相同视图，不能只改发送内容而遗漏资源身份。
87项 CPU 检查通过（7.478秒，4GiB/无swap），实际 GPU 覆盖仍待 D406。

D404四个副本的“分配至 worker 终态”重叠数各为2，整体8；平均时长
11.077382秒。它不是精确 router 在途计数（终态早于 router 计数扣减）。
结合固定官方源码 target 同时限制每副本请求容量，下一项公开配置验证有
合理依据，但不能直接认定GPU饱和，也不能由短回放外推完整吞吐。
[固定官方实现](https://raw.githubusercontent.com/ServerlessLLM/ServerlessLLM/9f50241baa5386e06a9321c51f19a9ef5f964c2b/sllm/routers/roundrobin_router.py)。

D406仅补全工件功能检查，采用与现有引擎四序列容量一致的公开target4；
其余min1/max4/keep10、真实远端、固定生成、60秒准备及资源保护不变。
这是待验证候选，不宣称最优；不同adapter映射不能用于target2/4因果比较。
随后回到原1000请求兼容验证、合法工作点冻结、4000 W0主比较。
全过程使用完整名称PrimeLoRA、ServerlessLLM。PrimeLoRA D371不改动；完整7B
比较、Resident参考、W1/W2/G2、3B、A1–A5、S1–S13和最终图表仍待完成。

详细接口、拒绝条件、CPU测试及下一轮验收见baseline仓库
`docs/ieee_tc/D405_SERVERLESSLLM_ADAPTER_COVERAGE.md`。
