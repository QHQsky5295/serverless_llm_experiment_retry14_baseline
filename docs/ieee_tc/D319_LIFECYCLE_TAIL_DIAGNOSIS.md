# D319：资源减少与尾延迟变化的完整日志复核

2026-10-07。7B 开发主线，复用 D315/D318 各 4000 请求；没有新 GPU
回放、远端服务操作、工件/负载生成、服务或配置修改、图形。源计划与冻结
V1 已完整重读；评价尺子、九个公式、D170 暂定参考保持不变。

## 问题和方法

D318 较 D315 减少 338.618264 GPU-s，但 P95 TTFT/TPOT 分别上升
3.299%/3.337%。先检查额外两次退役和重新启动是否集中覆盖退化请求，
以及恢复容量后是否仍然变差。不能由两次闭环轨迹直接推出因果归属。

只扩展现有 `analyze_d297_full.py --lifecycle-output`，复用 D304–D306
已测试的 decode、同 owner 窄 load/acquire、prefill 包络区间逻辑。
输出到新目录 `paper_results/ieee_tc/p2_backend/d319_lifecycle/analysis1`，
拒绝覆盖。所有原始与前置汇总 SHA 核验；大 outcome 用流式读取。

新增两种描述：

- 真实物理 GPU 相邻 allocation 区间之间的 release gap，按设备积分。
  这是真实未持卡的时间，不是“禁止退役”的反事实节约。
- 以两个运行相对业务开始时间的 startup/release-gap 并集定义共同到达
  集合，同一 request ID 在两边属于相同组；另报两边最后 engine-ready
  之后的到达集合。窗口来自事件边界，不按延迟高低选择，但仍属事后分组。

逐请求保留所有 8000 个区间记录、原生输出 SHA、240 个输出差异和所有条件
组。首次调度至首 token 不是纯 prefill kernel；load 包含 acquire fence，
不重复相加；任何重叠均不当成可扣除的阻塞时长。N_correct、正式联合 SLO、
warm/Resident/旧 Prime 新指标仍未闭口；单次开发对照不提供 CI。

## 历史及主要资料

已核查 D304–D307、D314/D315 和 D316–D318 的完整回放决定。D307 的
512-token 候选已因 TTFT/资源代价关闭，D315 已恢复 1024；本次不再扫描
token 预算、不重复 profile、不重开 D294/D295 的数值小诊断。

重新核对 [vLLM 0.30 LoRA worker 源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)：
LoRA load/activation 位于单线程核心路径，不能从外层异步推断它不干扰生成，
也不能直接搬进线程池。[同版本优化说明](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/docs/configuration/optimization.md)
与 [Sarathi-Serve 原论文入口](https://www.usenix.org/conference/osdi24/presentation/agrawal)
支持检查 prefill/decode 共享执行的方向，但不证明本次日志的因果效应。

使用 analyze-results 的原始表—观测—解释—下一验证结构，不生成图形。
不因局部条件均值立即改控制阈值。

## 完整观测结果

两次全部4000请求的原生时间线、owner、lease、输出数与前置结果一致。
全部240个输出SHA差异保留。物理区间与U积分一致。

| 指标 | D315 | D318 |
|---|---:|---:|
| GPU-s | 15865.617 | 15526.999 |
| 相邻持卡区间之间的释放时间，GPU-s | 0 | 324.251 |
| 全4000请求P95 TTFT，ms | 2934.366 | 3031.176 |
| 全4000请求P95 TPOT，ms | 62.152 | 64.226 |
| 两边最后engine-ready后的同2963请求P95 TTFT，ms | 3050.342 | 3149.713 |
| 上述2963请求P95 TPOT，ms | 56.803 | 62.352 |
| 上述2963请求临时时间联合超阈值数 | 124 | 135 |
| 请求关联native load次数 | 2168 | 2135 |
| native load总时间，秒（不是GPU-s） | 74.269 | 74.021 |
| 单次load平均/P95，ms | 34.257 / 63.870 | 34.670 / 64.814 |
| 与同owner load区间重叠的decode请求数 | 1400 | 1614 |
| 与同owner peer-prefill包络重叠的decode请求数 | 2217 | 2436 |
| decode期间其他native请求的时间加权占用均值 | 1.466 | 1.695 |
| 自身scheduled→first包络平均/P95，ms | 198.472 / 346.693 | 198.720 / 332.658 |

完整各组均值、分位数、失败分类在paired_groups.csv；全部重叠交叉组和
空组N/A在overlap_groups.json。没有根据是否支持假设删除组。

1. D318的释放区间为业务起点后143.148–363.340秒、965.166–1069.225秒。
   合计324.251GPU-s，是两运行U差338.618GPU-s的95.757%；其余14.367来自
   实测其他生命周期边界差异。这是积分分解，不是禁止缩容后的反事实收益。
2. 释放期间同298请求的临时联合超阈值数12→11；startup窗口同158请求
   为12→18。扩容时段有退化，但最后engine-ready后（共同边界1113.182秒）
   的2963请求仍有尾部退化。因此“仅消除两次再扩容便可解决全部退化”
   不受支持。不同组分位数不可相加分解全体P95。
3. D318有107条TPOT临时超阈值，92条不与其他副本startup重叠；其中
   98条与peer-prefill包络重叠。native load次数和累计事务时间反而略降。
   更合理的待检方向是并发落点和时序，而不是“远端多下载”或“多加载”。
4. 并发、缓存状态及请求路径共同改变，慢decode更容易被划为重叠组。
   上述关联不证明prefill或load单独导致了退化；没有反事实扣除，也没有CI。
   P95 TPOT升高而临时TPOT超阈值数110→107并不矛盾，分组阈值与单一
   全局分位数是不同统计量；正式N_correct/SLO仍未知。

## 决定与下一主线动作

保留D318的实测取舍及D316已测试实现；不冻结“最优Full”，不据此延长
cooldown、增加GPU常驻数、放宽SLO或重复512-budget候选。本项离线定位结束，
不继续围绕相同日志做无止境分组，不重跑D318以寻找更有利一次。

下一项只验证一个可证伪问题：**在相同当前可行集里，router消费的Q_i和
服务观测是否准确覆盖已分配、准备中、已进原生调度但未完成的工作；是否
存在生命周期变化后落点负载低估，使现有公式在错误输入上作决策。**
先检查已有routing_snapshot/native记录、相应源码历史和官方实现。若字段
一致、未观察到低估，则明确否定该实现缺口，不硬改公式或另加偏好系数；
若确认遗漏，才做一项可重复的确定性测试、修正和预登记最小验证。保留
共享容量和原子reservation，不把简单负载均衡替代论文路由规则。

已只读核对：retirement cleanup在控制协程中await，但外置arrival receiver
及request task独立运行，不能仅凭await声称所有新请求被阻塞。尚未修改
这条生命周期路径，也不将整个清理时长从TTFT/E2E或GPU-s中扣除。

本轮先43项CPU测试通过（含5项新测试），完整离线计算exit0。分析峰值
1041989632B，3/4GiB限制、swap0，所有high/max/OOM事件0；实际cgroup
空且populated0后按InvocationID关闭。新增派生数据约6.7MiB，原始证据
零覆盖。全部对照仍是单次开发运行，不是最终7B验收。3B、旧Prime新口径
必要补测、baseline、主比较、消融、敏感性矩阵继续保留待执行。

最终364项CPU回归通过（18.470秒）；22个来源和9个产物SHA全部复核。
检查峰值1095843840B、无high/max/OOM/swap，实际为空后关闭。147项历史
保护清单仅保留既有baseline replay脚本差异，raw_pass=false如实保存；
用户dirty manifest SHA未变、不纳入提交。完整回执为
`paper_results/ieee_tc/p2_backend/d319_lifecycle/validation_checkpoint.json`。
