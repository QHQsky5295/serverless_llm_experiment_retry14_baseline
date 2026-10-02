# D176 — 用已持有 HOST 副本凭证接续原生加载事务

2026-10-03，P2/P3 单一开发候选；父版本 `95de3c348a59529cfe57ea5aa2501cdb9e534372`。
状态：45 项定向检查、947 项回归和精确父版本操作对照通过；不是新的性能结果
或 7B 验收，待普通 Full 实测。

## 问题与可证伪假设

D175 已完整封存：4000 次原生完成，平均 TTFT 3.914028 s、P95 8.695495 s，
其中进入引擎前平均 3.525121 s。D170 暂定阈值下 timing-only 联合上界为
61.175%，不满足 95%，也不是正式 SLO。不能把上一轮改善写成模型已达标。
D161 旧条件诊断中 native-HOST 的 source→dispatch 为 2.005812 s；这是
路由形成的条件集合，不能据此单独归因。D162 栈中反复出现 HOST inventory，
D174/D175 已减掉无消费者的描述表，不能重复提出该优化。

当前请求链为：全量路由观测→选 HOST→原子持有精确 CPU 副本→全量路由观测
（再次遍历全部注册 adapter）→原子 load/acquire。第二次观测没有选择新副本，
也不授予物理容量；成功持有的 CPU 对象本来就不可被驱逐/悄然替换。

假设：把已经确认且仍持有的 CPU 副本凭证直接传给现有原子事务，可消除普通
HOST 请求的一次全量查询及其同步 CPU/传输成本，不改变路由公式或容量检查。
若发生 GPU promotion，原子事务显式拒绝 HOST 条件并重新读取实时状态；若
容量不足，等待真实释放后重新观测。假设仅预测少一次无冲突查询；端到端是否
有净收益，必须用随后同配置普通 Full 4000 请求检查，不能预先宣布。

## 历史和同类实现

- D111/D144 的同次查询共享、D151/D174 的投影和 D172 的解析器均已在父版本；
  本次不再改消息编码、图表示、跨调用缓存、batch、profile 或超时。
- [vLLM 0.30.0 worker manager](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/lora/worker_manager.py)
  的 LRU `add_adapter` 在单线程 core 中检查已注册对象并激活。原生加载不要求
  控制器先拿全注册池的 tensor 图，再提交一次加载。本项目的第二次图查询是
  自己增加的分布式确认实现成本，并非 LoRA 推理不可避免的步骤。
- [vLLM 官方 CPU 性能分析](https://vllm.ai/blog/2024-09-05-perf-update)
  支持检查 CPU 对象/控制工作，但其性能数字不移用于本实验。
- [dLoRA 原论文](https://www.usenix.org/system/files/osdi24-wu-bingyang.pdf)
  是联合请求/adapter 管理参照；没有据论文层级描述推断它不存在相同开销。
  具体改动依据同版本官方源码及本项目已实现的序列化 owner、pin 和 copy ID。

## 不变量与执行边界

仅扩展 runner 的 acquisition 输入及 HOST preparation 调用；native owner/
worker、图校验器、九个公式、全体候选路由观测、planner/admission 不改。

1. 必须有同 reservation、同 engine 的已确认 HOST hold，匹配 owner、clock、
   adapter、路径、lease、CPU incarnation 与已冻结 admission；不接受任意旧快照。
2. 使用 hold 完成 epoch 和精确 source ID。原生事务仍在加载前验证当前对象、
   tier、pin、GPU/CPU 容量及 HOST 字节预算，加载后仍验证并完成 fence。
3. 只有无冲突 HOST 的中间全量图查询消失；初次路由及冲突后的图校验保留。
   不再要求对已持有对象重复取得完整表示图，不声称每请求仍有两次图检查。
4. 若期间已提升到 GPU，原生返回无副作用 conflict，按既有 fresh observation
   重选实际 native source；不把 dispatch-time HOST 追改为 GPU。
5. 精确 CPU copy 可以经历无关 epoch 变化。容量等待绑定新原子冲突的 epoch
   和 blocker owners；它不授予容量。释放后仍全量确认再执行，禁止轮询/强制驱逐。
6. RPC intent 仍先于 await 登记；丢失回复、取消和不明 native outcome 仍保留
   所有权，不把未知当成功或空闲。GPU-only 和 file/remote 原路径不变。

## 验证及下一步

| 检查 | 当前状态 |
|---|---|
| 无冲突 HOST 查询次数及凭证身份 | 精确父版本 2 次→1 次，原生事务保持 1 次 |
| 无关 epoch、GPU promotion、坏凭证/图、丢回复 | 45 项定向检查通过，新增 5 项；坏凭证含 10 个子例 |
| 容量等待、取消、真实释放与原有回归 | 947 项回归通过，99.951 s；和定向检查重叠，不相加 |
| 受限普通 Full 回放净收益 | 未执行 |

按 academic-plotting 与计划 §11，本次正确性/精确操作次数采用表格，不强画
性能图；不把模拟 owner 单元测试当 GPU 实验。复用 D174 有界 CPU runner、
现有真实 owner/request fixture，无新增权重、负载、远端缓存或框架。
待通过后备份，再只跑一个普通 7B W0 Full；保持 D175 cap4/D157 profiles。
7B 数值正确性、共同参考、旧 Prime 新指标和 G1/G2 仍开放，3B/外部基线暂停。

### 精确操作次数对照

复用既有真实 runner/native owner 的微型 CPU fixture，将 Git 中精确父版本
两个方法与候选分别绑定到新 fixture。无模拟推理性能主张，不测量 GPU latency。

| 选择 tier | 父版全量查询 | 候选全量查询 | load 事务（父 / 新） | 未释放 GPU/HOST leases |
|---|---:|---:|---:|---:|
| gpu | 1 | 1 | 1 / 1 | 0 / 0 |
| host | 2 | 1 | 1 / 1 | 0 / 0 |
| nvme | 3 | 3 | 1 / 1 | 0 / 0 |
| remote | 3 | 3 | 1 / 1 | 0 / 0 |

两版本各四例均完成，dispatch tier 不变，HOST hold/release 仍各一次；其他
路径的操作次数完全相同。promotion 冲突中仍重新查询一次完整图，坏图拒绝；
原测试“必须第二次取全图”改成初次图及冲突后图的负例，不隐藏实际执行变化。
未知加载回复仍留下未决所有权，而不是通过释放计数伪装成功。

三项有界 CPU 工作均在 3/4 GiB、swap=0、CPU 2,3,26,27 下完成，实际资源域
自动移除，保存的内存事件/swap 均为零。没有 GPU/远端服务启动，没有失败测试
attempt。原始目录 `results/ieee_tc/p2_backend_qualification/d176_20261003`；
curated 表只陈述资格和操作次数，完整性能结果仍待后续单次 Full 验证。
