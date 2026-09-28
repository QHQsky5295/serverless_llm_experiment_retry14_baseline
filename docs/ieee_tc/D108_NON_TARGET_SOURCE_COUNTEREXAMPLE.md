# D108：非目标副本迁移使旧准备计划失效的因果反例

2026-09-28。生产实现仍为 D106 的 `b7161bdcafe05117dcc9434f12ede5a7003c121d`；
D107 证据已由 `30165e382965d320531347d3ba65c477d0362263` 备份。
本轮只进行 CPU 正确性诊断，未修改生产算法、启动 GPU 或新回放。

## 问题与对照

D107 原生错误检查了全部当前来源，而 D106 只处理准备目标自身的已确认副本
变化。以下反例检验两者之间的缺口，不用修改内部字典伪造状态变化：

1. 复用 `AutomaticGPUReplacement` / `MixedOwnedPreparation`，由实际分层文件
   管理器发布 b 的 HOST/NVMe 副本，内容 SHA 相同；由实际 native owner 管理
   CPU/GPU 缓存、引用及准备计划。模型/缓存对象使用既有 CPU 测试替身，不执行 CUDA。
2. 通过实际 selector 选择 c 作为唯一 GPU 准备目标并注册。b 不是待准备目标。
3. 变化前执行相同 native 入口，在通过来源检查后由测试专用 cost-provider
   哨兵停止。因此可以验证未改变状态时旧 guard 并不会报错，不执行实际准备。
4. 通过实际 owner 驱逐无引用的 b，再持有已确认文件引用，由正常 demand 路径
   从 HOST 加载相同内容的 b，释放引用。原冻结目标中的 b 路径仍为 NVMe。
5. 重新执行 c 的准备入口，确认来源域仍覆盖当前缓存、c 的路径没变、所有
   名称相同、GPU 确认完整，唯一差异为非目标 b 的物理路径。

| 检验项 | 观测 |
|---|---|
| 变化前原 guard | 通过，抵达测试 cost-provider 哨兵 |
| 变化后原 guard | 复现 `replacement epoch source identity or GPU confirmation changed` |
| 唯一冲突来源 | 非目标 b 的路径；c 不变 |
| 来源域覆盖 / GPU 确认 | 均完整 |
| 新旧文件内容 | 实际 file owner 确认 SHA 相同 |
| owner epoch | 20 → 29，来自真实 owner 操作 |
| 变化后是否抵达 cost provider / admission | 否 / 否 |
| 原准备调用是否额外加载、驱逐或更改状态 | 否 |
| 旧计划和引用是否清理 | 是 |

运行通过 22 项检查：1 项新增反例加 21 项已有 `AutomaticGPUReplacement`
检查。后一集合由 unittest 对导入测试类的自动发现一并运行，不冒称新增实验，
不重复执行来修改这一计数。测试耗时 2.051 秒不是推理性能。

整个分析服务运行约 9 秒，按既有规则请求 3/4 GiB 内存 high/max、swap=0、
CPU 2,3,26,27，退出码 0，并自动清理。未保存其结束前的 cgroup 事件快照，
故不把未记录的事件/峰值填为零。测试中的小文件来自既有 fixture，不是新 LoRA
实验权重、远端工件池或新工作负载。

## 第一性原则解释与设计边界

逻辑工件身份、native 对象 incarnation、文件路径、冻结计划有效期是不同概念。
当前 owner 已通过 `_validate_source_binding` 允许无引用、非当前准备目标的
同名工件在撤回旧对象后从另一合法路径加载。可是准备执行又要求冻结目标内的
所有旧路径永久不变。这两处合同不一致，不能用普通“坏工件”解释上述反例。

核查 [vLLM 0.30.0 官方 worker manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)：
LRU 以整数 adapter ID 管理，按请求加载/替换；core 操作串行不使整个多 RPC
计划成为原子事务。这只是语义参考，不是本机性能或相同内容的证明。

可接受的下一候选应统一“旧计划失效”与“真实身份/确认损坏”的处理：
以实际 owner 的完整后继状态为负面证据，在定价、驱逐和拷贝之前明确拒绝旧
objective，返回绑定 plan/hash/lease/epoch 的未执行回执；控制器验证回执并完成
既有收尾。只有后续正常规划周期可选择新目标，不自动重试旧计划。

尚未实施该候选。实施时必须保持名称、内容、rank、owner、clock、完整 GPU
确认和物理容量检查；未知 RPC 结果不能当作“未执行”，不能提前释放后备引用。
需要同时验证观察阶段和 native commit 间的状态变化，而不只加一次父进程检查。
不改 IEEE 九个公式、需求/收益、profile、资源包络、trace、deadline 或远端协议。

本反例证明存在该缺陷路径，但 D107 没有失败现场的具体 source pair，不能
追认为其确切原因，更不能解释 Full8 的 30 条超时或承诺性能提升。

## 返回主线

先保存反例及来源后实施一个候选，验证合法变化、恶意/损坏证据、RPC 回执、
收尾和新周期执行；不叠加 parser/调度优化。通过后再按原完整回放协议验证。
正式比较仍等待 3B/7B Full、warm SLO/Resident 和其余资格，baseline 继续暂停。

原始脚本/日志：`results/ieee_tc/p2_backend_qualification/d108_20260928/`。
汇总：`paper_results/ieee_tc/p2_backend/20260928_d108_non_target_source_probe.json`。
当前未新增性能图；本状态/对照表即本轮诊断交付。
