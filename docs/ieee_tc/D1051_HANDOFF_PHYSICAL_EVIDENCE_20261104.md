# D1051：handoff 的实际准备窗口与同后端输出差异

2026-11-04。复用 D1047 Full、D1050 NoHandoff 和 D980/D981 已封存记录，
没有新增 GPU 回放、修改服务算法、工件、负载或共同指标。当前主线仍为 A1–A5
消融/机制证据；本次有界分析结束后不因 NoHandoff 较好而另开 Full 调参。

## 1. 最新用户授权与现有 HydraServe 状态

用户确认现有 Kubernetes 集群原本不属于本项目，允许选择适配方式。此前的
隔离方案与这项要求一致：复用独立 rootless Podman 路径，不使用/修改其他项目
k3s 的权限、服务或资源。D807/D808/D809 已记录 3B W0/W1/W2 实际运行；
D510 及后续记录保留 7B 证据。不能因为收到同一问题的补充回复而重做基础部署。
所测 scheduler 为 pp1；未覆盖的 pipeline/consolidation 不能写成已验证，
也不能把未触发的机制认定为原论文的性能缺陷。

执行依据：BASELINE_COMPLETION_ORDER_20261012、BASELINE_ADAPTATION_STOP_RULE_20261016，
以及 D807/D808/D809 原始分析。没有申请共享集群管理员权限、修改 kubeconfig、
重启共享 k3s 或部署第二套共享控制面。

## 2. handoff 到底完成了什么

沿用 D1050 的完整实际 dispatch 分组，不使用旧 first-service 标签。Full 包含
1 个 initial 和 9 个 natural_scaleout；均有首个实际 dispatch，无 controlled。
以下计数来自同一时钟、同一 native owner 的实际加载回执，而非计划完成标签。

| 证据 | Full 本轮观测 |
|---|---:|
| GPU 准备目标/完成的准备区间 | 66，来源全部为 NVMe |
| 真正调用 GPU load 的次数 | 57 |
| 真正调用 native HOST load 的次数 | 58 |
| engine-ready 前完成的 GPU / HOST load | 0 / 0 |
| 首个实际 dispatch 前完成的 GPU / HOST load | 13 / 14 |
| 首请求 adapter 有提前 GPU load 的自然激活 | 1 / 9 |
| handoff 关联队列 job | 66，全部为 native HOST，不是文件复制 |
| handoff 关联远端传输回执 | 0 |
| 全运行远端获取次数（Full / NoHandoff） | 132 / 132 |

HOST 与 GPU 是可能相互依赖/重叠的阶段，不能相加冒充节省时延。
队列 job 数不等于物理加载数：等待、合并及已存在副本复用必须分开。
本轮 native HOST/GPU 准备在新引擎就绪后执行；在此前已登记的计划不等于
启动窗口内已完成物理加载。所有目标来自 NVMe，因此本轮没有 handoff 远端
预取可与引擎启动重叠的证据。这是实际场景结果，不是手动关闭远端、清缓存
或注入等待后的结论。

| 自然激活的首请求 | dispatch 前 tier | GPU load 数 | 首 dispatch 前完成数 | 是否匹配首请求 |
|---|---|---:|---:|---|
| req_00045 | NVMe | 2 | 2 | 否 |
| req_00049 | Remote | 2 | 2 | 否 |
| req_00180 | GPU | 9 | 4 | 是 |
| req_00741 | NVMe | 5 | 0 | 否 |
| req_01174 | NVMe | 4 | 4 | 否 |
| req_01777 | NVMe | 9 | 0 | 否 |
| req_02268 | NVMe | 9 | 0 | 否 |
| req_02763 | NVMe | 12 | 1 | 否 |
| req_03170 | NVMe | 5 | 0 | 否 |

req_00180 的 medical_lora 准备完成于 3231765.810028511；请求读取同一
content/source_id，GPU-confirmed=true，native_load_invoked=false。请求快照的
确认时间已更新，故不据此宣称完整的“没有中间驱逐/重加载”连续历史。它支持
“同 adapter 提前准备且请求复用 GPU 已确认源”的关联，不是反事实节省量。
后续同 adapter 的 GPU-hit 次数也只作关联；不能把所有后续命中都归给这次准备。

对应主指标仍原样保留：

| 观测指标 | Full | NoHandoff |
|---|---:|---:|
| 原生合同完整请求 | 4000 | 4000 |
| 联合 SLO | 95.90% | 95.85% |
| 物理 GPU-s | 6182.715971 | 5990.135370 |
| P95 TTFT / ms | 1114.091001 | 1052.182063 |

Full GPU 时间多 3.214962%、P95 TTFT 多 5.883862%，只多 2 条联合达标。
这轮不能作为 handoff 总体性能提升或“隐藏远端传输”的正证据。有限首请求
目标匹配和缺少启动前实际搬运，为为何没有观察到整体优势提供了机制解释；
不是对负差值的完整因果分解，也不能外推 handoff 在所有场景无用。
自然闭环的激活、请求与目标不同，原计划的受控同计划 Full/Delayed 对照
仍未完成，不用本轮替代。

## 3. 206 个输出差异的有界审计

Full/NoHandoff 的 4000 个请求输入、target、原生输出数量相同；3794 个
输出 digest 相同，206 个不同。当前性能日志只保留 digest，无原生完整 token
序列/logits，不能离线定位这 206 条的首个分歧，也不能用文本重分词替代。

复用 D980 的原生序列和 D981 的独立 HF/PEFT 参考，按 request、adapter、
prompt hash、原生 prompt-token hash、内容 SHA、target/actual 与生成合同
逐字段核对，已有前 100 请求全部可匹配，不需重新运行参考。

| 前 100 条的 digest 对照 | Full | NoHandoff |
|---|---:|---:|
| 等于已有独立参考序列 | 93 | 96 |
| 等于 D980 原生序列 | 94 | 93 |

此范围内两变体不同的请求是 00018、00021、00072、00083、00099：

- 00018、00072：Full 等于旧原生序列，NoHandoff 等于独立参考序列。
  D981 已记录其首分歧位置为 4、42（零基），旧原生 token 均在参考 top-2；
  FP16 top-1/top-2 间距分别为 0.0078125、0.015625。这两对当前 digest
  精确对应已观察到的近并列分支，不需要再生成一次同内容参考。
- 00021、00083：NoHandoff 等于参考，Full 为当前未解析的不同 digest。
- 00099：Full 等于参考，NoHandoff 为当前未解析的不同 digest。
- 其余 201 个差异不在这份历史参考覆盖内，仍未解释。不把它们默认为正确，
  也不把 digest 不同本身等同于请求执行错误。

官方 vLLM 文档说明默认不保证结果可复现，在线执行的调度/批次可能影响输出；
本地 0.30.0 envs.py 的 VLLM_BATCH_INVARIANT 默认也为 False。这只支持一个
可能机制，不证明本轮所有差异都是数值误差，不能用文档代替个例证据。
来源：[vLLM reproducibility](https://docs.vllm.ai/en/latest/usage/reproducibility/)。
此次没有改变 batch-invariant 设置、sampling 或性能合同。

93/100、96/100 是限定参考序列的一致数，不是任务准确率或全池资格。3B
现有 500 工件的全零更新限制继续披露；不能靠输出匹配证明非零 LoRA 算术。
n_correct=null、formal_ranking_eligible=false 不变。不开展 4000 请求逐条
HF 复测，不为获得相同输出反复跑性能键。需要进一步数值诊断时，应利用
后续已批准诊断保留最小必要原生序列，而不追改旧结果或阻塞全部机制工作。

## 4. 产物与分析修订

权威分析目录：paper_results/ieee_tc/ablation/d1051_handoff_receipts_v2/。
包含 handoff.json、outputs.json、分析脚本快照及来源/协议/资源 manifest。
复用 D1050 comparison 与已有 activation_first_services；未重建实验框架。

第一次离线投影成功，但把 tier=host/nvme 的队列记录统称 file_jobs。
检查 owner 后确认这 66 条均为 native HOST。保留首版目录
d1051_handoff_receipts/ 作为分析历史，v2 明确 storage_domain 并修正名称。
这不是新性能运行、没有覆盖原始测量，也没有改动计数以迎合结论。

v2 辅助资源域 3/4 GiB、swap=0、CPU 2,3,26,27，CUDA 隐藏；峰值
63,225,856 B，high/max/OOM/kill 事件均 0。两个原性能运行已结束，无并发 GPU
测量。当前盘约 151 GiB 可用，下次重型启动前仍须检查既定 150 GiB 门槛；
本次没有删除/压缩数据或操作远端服务。

EXECUTION_STATUS.md 是预存 dirty 文件，本轮不覆盖；以本文件承接 D1050 的
当前分析状态，历史 D998 页首不代表当前实验阶段。

## 5. 下一项与尚未完成范围

下一项为既定 A1 的 CapacityOnly：先修正 D1049 已发现的 activation 入口
capacity_only 透传缺口，核查 initial/natural 都只关闭软 admission、保留
物理可行性与原 victim/replacement 语义。复用 Full、输入、profile、共同
阈值和资源参考，最小 CPU 测试通过后再登记唯一 W1 执行键。这是机制消融
所需适配，不是看到 NoHandoff 胜出后改变 Full。

剩余 A1 十一变体及两个累积节点、A2–A5（含受控 handoff 和 confirmed-state）、
S1–S13、两模型最终证据边界、CE/价格/Pareto、两轮审稿设计文档、IEEE 图表
及 FaaSLoRA V2 最终集成均未删减。保持完整目标 active，不宣称论文实验已完成。
