# D181：请求范围原生占用观察

状态：最终实现通过 974 项 CPU 回归检查；不是 7B 性能或 G1/G2 合格声明。
父版本 `512b1be2d73924c59e7e36b1e80cf4e56669bd15`；Plan、V1、模型配置、
九个公式、远端交付、生成合同不变。沿用 D180 的历史、官方源码和依赖证据。

## 实现边界

- worker 在一个串行观察中读取完整新鲜 native owner/epoch/cache/slot/source
  identities，并只遍历显式 requested adapter 的 HOST tensor；无 TTL、
  旧观察缓存、未来请求信息或按模型猜测的 footprint。
- request-induced storage graph 明确标范围；其 union 和 exclusive 字段
  改为 `requested_host_storage_bytes`、`within_observation_exclusive_bytes`。
  它们不能被旧完整资源 parser 或物理预算消费者误认。
- dedicated frontend 校验 scoped graph；发送完整身份与目标的实测类别，
  验证仍在 GPU 串行循环之外。完整 planner/admission/storage endpoint 不变。
- 控制器仅在副本成员和 requested scope 都相同时共享仍在进行的观察。
  最后一个 waiter 取消时收回该观察；不同 target 的波次独立。
- 逐副本保留独立 identity frontier。同 epoch 不同 scope 比较共同身份和
  重叠 footprint，并保留本 epoch 已见 footprint 的一致性证据，防止
  A→B→A 交错漏检。该证据不用于给后续请求提供旧类别；新请求仍需新观察。
- 同 epoch 的完整 HOST union 和 GPU pool 容量另保留一致性证据；完整观察→
  scoped 观察→完整观察不能遗失之前已确认的全局容量。新 epoch 清除该
  一致性证据；它不用于提供新请求类别，也不成为预算或驱逐容量来源。
- Eq. (2)/(3)、可行集/原子 reservation、被选 source revalidation 和 leases
  不变。加载和冲突后的完整来源观察暂不改变，不叠加其他优化。

## 检查结果与失败记录

复用既有 source、native-owner、routing、cancellation 测试和 D178
bounded CPU runner；新增 16 个测试方法，覆盖 scope coverage、共享 storage、
frontend/proxy、cross-scope sharing/cancellation、同 epoch 重叠冲突、
identity frontier、交错全局容量一致性。counts 有重叠，不能相加当独立重复。

| attempt | tests | errors | status |
|---|---:|---:|---|
| tests1 | 102 | 1 | fixture_error_preserved |
| tests2 | 102 | 0 | pass |
| regression1 | 973 | 0 | pass_before_capacity_frontier_guard |
| regression2 | 974 | 0 | pass_final_source |

tests1 的错误是 SelectedSourceAdmissionIntegration 模拟后端缺少新接口
`ieee_request_sources`；保留失败日志及精确 fixture。补齐接口后 tests2
通过，未改生产实现绕过错误。后续独立复核发现完整容量→局部→完整观察的
一致性证据需要保留，补充 guard 和反例后重新运行完整 regression2。
regression1 的生产/测试源文件也保留，不将它冒称最终源码资格。

- tests1/tests2：102 项，0.340/0.309 s；返回码 1/0。
- regression1/regression2：973/974 项，100.151/100.743 s；均返回 0。
- 实际 scope InvocationID 依次为
  `f93fba661bc64a09a527cd4fc3921050`、`8c42d5d548004d80af679e1acbf15d03`、
  `8ef7a6a123fd44f192ce21eab2c7abb4`、`c6e82fc68f704fb692a576f182bebd84`。
- 全部 MemoryHigh/Max=3/4 GiB、swap=0、CPU=2,3,26,27；终态内存事件和
  swap 为零，scope 自动消失；无 GPU 模型、远端操作、完整权重/trace 复制。
- CSV/JSON 及本表是资格交付；没有用户 TTFT、吞吐或 CI 数据，不强造性能图。

## 假设与依据

D179 最新完整回放没有证明 D178 的端到端净收益；D180 随后对实际保存观察
证明 64 个目标的 footprint 和 192 个 Eq. (2) 类别在 scoped 测量下相同。
该既有证据复用，不重新运行。D181 落实范围显式的生产端点及新鲜度合同。

官方 [vLLM 0.30 worker manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)
在请求相关 LoRA 集合上执行加载/激活，并注明该加载路径运行于单线程核心
循环。官方 [CPU 性能改进说明](https://vllm.ai/blog/2024-09-05-perf-update)
将 CPU 开销与前后端分离列为优化问题。本次推论是减少控制观察进入该循环
的无关遍历可能有益，不借用其硬件加速比，不声称它已经验证 Prime 的改进。

## 尚未完成的验收

CPU qualification 后保存验证表、checkpoint，再做普通 7B Full 完整回放。
必须同时比较 RPC/collection 数、等待、TTFT/TPOT 和生命周期资源；
D180 工作项减少不能直接作为延迟加速比。3B/外部基线继续等待。

特别注意：不同 adapter 不能共享缺少自身 footprint 的观察；因此本候选
可能增加 RPC 数。只有端到端完整回放才能判断减少遍历与增加调用的净效应。
数值正确性、共同参考冻结、旧 Prime 新指标对照、G1/G2 仍未闭环。
