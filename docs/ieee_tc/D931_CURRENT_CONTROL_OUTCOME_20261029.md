# D931：7B W2 当前控制完整结果与主线交接

日期：2026-10-29。当前阶段仍是 PrimeLoRA 的 7B 同合同优化；没有宣布
全模型 G1/G2 完成，没有开始新的消融或敏感性矩阵。

## 1. 单轮结果

三轮均使用已有 source42 的 W2 开发回放、4,000 请求、500 逻辑 adapter、
真实远端预发布 gzip、原生固定输出合同。不是独立留出，也没有重复运行取优。
以相同 D394 阈值及 Type-1 分位数重新统计：

| PrimeLoRA 版本 | TTFT通过 | TPOT通过 | 联合时延通过 | GPU-s | GPU-s/offered | P95 TTFT ms | 平均 TTFT ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| D917 | 3898 | 3974 | 3883 / 97.075% | 11507.846575 | 2.876962 | 2672.029257 | 1124.789903 |
| D930 压力分类守卫 | 3875 | 3976 | 3856 / 96.400% | 12382.350638 | 3.095588 | 3256.378457 | 1230.830830 |
| D931 守卫关闭 | 3896 | 3979 | 3881 / 97.025% | 11716.064100 | 2.929016 | 2997.200298 | 1215.954490 |

D931 相对 D930 的 GPU-s 减少 5.381%、P95 TTFT 减少 7.959%，但相对
D917 仍增加 1.809% 和 12.169%，联合通过少 2 条。不能把相对慢候选的改善
写成已达到旧系统水平，也不能以 0.05 个百分点差异声称统计优劣。

三轮 `n_correct=null` 保留。全部 4,000 原生 token/绑定/时序检查通过不等于
独立数值 LoRA 正确性已证明。上表是联合时延统计和 per-offered 资源诊断；
G1 的 per-correct 指标保持未知。也未在这里新冻结 Resident 预算或宣称 G2 合格。

## 2. 机制归因更正

D930 控制日志有 1,916 次 no-action、10 次 activation、9 次 retire，
**0 次 `no_action_current_demand_covered`**。所以初版 D930 结论中的
“守卫抑制导致早期扩容推迟”应撤回。D931 关闭守卫后有 11 次 activation、
8 次 retire；反馈轨迹不同，不是同一组被拒绝动作的恢复。

D917 的首请求 TTFT 为 13,669.073 ms，前 3 次扩容均在 queue/active 为 0
时由该样本的 TTFT 压力触发。D930/D931 首请求分别为 1,291.387/941.665 ms，
最早扩容由 active pressure 触发。不能忽略这个差异，也不能尚未检查编译日志
就断言原因是 JIT。守卫开启时会计算额外投影，即使最终不抑制动作；其代价尚未
隔离，不能用这个可能性替换原来未证实的因果解释。

## 3. 下一项有依据的诊断

复用现有 `partition_retained_request_stages`，三个运行的可加和阶段误差均为
0 ms。重要观测（均值 ms）：

| 阶段 | D917 | D930 | D931 |
|---|---:|---:|---:|
| 计划到达→请求释放 | 23.577 | 112.168 | 106.039 |
| 释放→开始获取服务容量 | 21.438 | 37.797 | 36.947 |
| runtime slot 等待 | 280.534 | 253.203 | 251.943 |
| 控制器准入→engine dispatch | 323.387 | 368.962 | 359.549 |
| 原生 engine 队列 | 4.551 | 4.780 | 4.929 |
| 原生 prefill 墙钟区间 | 202.765 | 196.372 | 200.225 |

这支持先核查请求释放/控制器路径与源码历史，不支持直接把退化解释为 GPU
排队。阶段是墙钟区间，不能当作 CPU 占用，也不能把删除某区间当作反事实速度。
唯一下一主线是：比对 D917/D930/D931 的实际执行配置、源码、编译初态和已留存
事件，定位释放延迟和控制处理差异；有具体新证据后才提出一个候选，并查阅相关
官方实现/论文。当前不追加另一轮守卫规则或无变化回放。

## 4. 合同、计量和来源

- 三轮 request、adapter、arrival、canonical prompt、native prompt、target
  合同哈希相同：`61505a528bc287906e5d8e93a023d3a1ce2d09e2176533ada4c5d6808e17f052`。
- 每轮原生数量匹配 4000/4000，完整 dispatch tier 4000/4000；10 类绑定检查
  各 4000/4000；原生 TTFT/TPOT 和阶段恒等式通过。没有按标签填补数值正确性。
- 物理区间并集与互斥窗口积分一致，allocation journal SHA/release 和 terminal
  journal SHA 通过；无未关闭物理 lease。旧 infra/CE 摘要不替代物理占用。
- 分析复用 `scripts/ieee_tc/analyze_d455_stages.py`，增加显式历史来源接口，
  没有新建性能框架。完整三源分析成功；3 个拒绝覆盖/重复来源/错误协议测试通过。
- 分析文件：`paper_results/ieee_tc/p2_backend/d931_control_audit/summary.json`，
  SHA `2d380bf08f5f5f8d073cb4ccbbb9d0e6035133a361152bd8c919eacd9ef48ce8`。
  内含三个 raw result、launch、physical summary、outcome 及协议 SHA。
- V1/单轮 V2 不变；远端备忘录修正了 per-offered/per-correct、native token/chunk
  混称和 no-delay 严格上界措辞。这是对齐现有协议，不是按结果换指标。

## 5. 资源与保护

D931 launch/service/watchdog 正常结束，GPU 计算进程已释放。4079 个外置采样
中无中止、外部 compute、逃逸 worker；memory high/max/OOM/kill 均为 0，
服务 swap 峰值 0，内存峰值 31,990,972,416 bytes（29.79 GiB），主机最小
MemAvailable 92,616,564,736 bytes（86.26 GiB）。末尾 NVMe 私有缓存已由
既有收尾逻辑清理，仅剩空目录；不是永久增加第二套工件。收尾后根盘可用
170,412,204,032 bytes（158.71 GiB），新重型任务前仍须检查预计增长量。

watchdog SHA：`159ee947569a02e93b34c0a2ae1f9756f55fee00843d82fa1c4e13b4078115c1`。
launch SHA：`a3f721817bf5d6f8d9c48b6412fd3491276d994ac8667f72ef1caca451219434`。
physical summary SHA：`96681c90488b43b44b8534330a336c854207fd0a9d46f249f158e239a18a7706`。
保护清单 `d435_20261011/protected_before.json` 核验 147 项零变化；用户 manifest
SHA 仍 `3b69bd0a6888aa56e98c3b9efb61234860d4b80f25729cfea6b79126d19127df`，不 stage。

## 6. 保留的完整范围

HydraServe 的隔离适配和 W0/W1/W2 结果已存在，不能重新写成尚未复现，也不
改动非本项目 k3s。已发表 baseline 身份、ServerlessLLM 优先审计和有限适配规则
保持不变。7B/3B 完整同配置 W0/W1/W2、G1/G2、剩余 baseline/dLoRA、独立数值
证据、A1–A5、S1–S13、两轮审稿对应文档和最终图表/V2 交付均未被本轮省略。
优化接近目标前不画图；不新增 13B/SGLang；不改论文或旧投稿文件。
