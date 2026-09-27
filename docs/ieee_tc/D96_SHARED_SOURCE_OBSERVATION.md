# D96：并发状态读取合并与整组发布

2026-09-28。Prime IEEE Full 开发验证；不是主比较或 SLO 达标结果。

## 已观察的问题

D93 的完整 3B W0 attempt2 未完成，320 条成功请求中 201 条发生过
selected-source 重选，累计 1242 次、单请求最多 47 次。此计数不含
整组 snapshot 被丢弃的次数，不能把它当成全部控制开销。

阅读实际 runner、native owner 和历史提交后，CPU 检查区分了三件事：

| 检查（真实控制代码，原生测量为 CPU fixture） | D95 版本 | 本次候选 |
|---|---|---|
| 32 个并发请求读取两个不变副本 | 64 次 source RPC 调用 | 2 次，共享正在执行的读取 |
| 同一组 32 个请求的实时 admitted count | 各自读取 | 仍逐请求读取，观测为 0 至 31，不复用可行集 |
| 下一次请求在前组完成后到达 | 新读取 | 仍新读取，两副本累计 4 次；无 TTL 缓存 |
| 第二副本观测过期，整组拒绝 | 第一副本已提前发布 | 整组拒绝且第一副本不变 |
| 32 次 acquisition 用同一原生 epoch，实际对象与 slot 不变 | 1 次成功、31 次 stale | 未修改此 guard，仍为待解释因素 |

最后一行是有意排队到单线程 native owner 的状态反例，不是 32 个
GPU 推理请求性能实验；不据此宣布 D93 的全部排队已归因。
两个预期行为红测试实际均失败（0.018 秒），原始记录保留。

## 设计依据与实现边界

[singleflight 官方文档](https://pkg.go.dev/golang.org/x/sync/singleflight)
给出合并同一 key 的正在执行操作的通用方法；本次仅借鉴该并发结构，
没有引入 Go 依赖，也不把它包装为论文的新算法。

[vLLM v0.30.0 EngineCore](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core.py)
在 engine step 前处理输入队列；
[LoRA worker manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)
说明原生 LoRA 操作依赖单线程 core。反复完整观测会增加该路径的工作量，
但这只是待在完整回放检验的瓶颈假设，不是官方源码对本机延迟的证明。
上述资料于 2026-09-28 核查。

候选保持以下不变量：

- 只有相同 slot/engine 身份、尚未完成的只读 collection 可以共享。
  结果完成后不作时间缓存，不加轮询周期或人为 sleep。
- 原生状态整组校验，再在不让出控制权的区间统一发布；相同不可变观测
  可以重复提交，旧 epoch、旧 capture、同 epoch 异内容仍拒绝。
- 文件状态、计数、负载可行性、profile 估计和选路保留逐请求计算；
  selected-source 的原生再验证、admission、引用和容量检查不变。
- 一个等待者取消不影响其他等待者；最后一个退出时取消并等待本组读取。
  读取失败要收尾其他读取，传播错误，不自动重试。
- membership 或 engine 替换使旧 collection 拒绝；不继承另一 owner 的状态。
- 请求证据保存 collection ID／是否共享；累计记录 collection、实际 RPC
  调用入口、membership/stale 拒绝次数，并纳入失败 outcome。RPC 调用入口
  不等于实际 native 开始确认。

没有改变九个公式、服务时间分桶、配置、生成、输入、profile、超时或
远端交付；没有改 native epoch 的含义。原生引用冲突先保留，避免把
尚未测量的多个性能改动堆叠。

## 验证与下一步

目标检查 57 项通过（0.415 秒）；最终相关回归 895 项通过（49.104 秒），
既有离线 CPU 环境、3/4 GiB high/max、零 swap、辅助 CPU 集合。
包含共享但实时计数不共享、整组发布、单个／最后等待者取消、读取异常
收尾、不自动重试、engine 更换、旧状态拒绝。review 为 local-only；
按 experiment-bridge 先反例后验证，当前用状态表，不制造性能图。

原始日志：`results/ieee_tc/p2_backend_qualification/d96_20260928/`。
结构化回执：`paper_results/ieee_tc/p2_backend/20260928_d96_shared_source_observation.json`。

CPU 证据支持进入一次新的 canonical 3B Full 4000 W0 开发回放；保持
D88/D89 配置、相同输入和期限，以 D94/D95 正确性修正＋本候选为清楚
标识的新版本。完整回放才检验是否减少实际重复读取、恢复进展及副作用。
不能把跨多个修正的前后 TTFT 差全部归因于本次合并。
不再重复 prefix、初始化 profile 或整池下载；不在新回放前继续叠加
per-copy epoch 优化。完成／失败均先清理、校验、状态表，再决定下一步。
7B、基线、warm/Resident、M1/M2、消融和敏感性仍未完成。
