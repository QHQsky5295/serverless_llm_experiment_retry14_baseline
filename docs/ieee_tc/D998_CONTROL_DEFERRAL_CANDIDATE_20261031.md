# D998：面向原生步进的只读控制调用延后候选

日期：2026-10-31
阶段：3B PrimeLoRA 主线；离线候选登记，尚未修改生产代码，尚未运行 GPU。

## 1. 为什么登记这个候选

当前 3B W2 开发参考 D992 完成 4,000/4,000 个请求，但按冻结的联合
TTFT/TPOT 服务约束只有 3,678/4,000（91.95%）达成。失败分解为：

| 类别 | 数量 | 现有证据 |
|---|---:|---|
| TTFT-only | 168 | 平均 arrival→admission 743.809 ms、admission→acquired 1,488.599 ms |
| TPOT-only | 132 | 其中 128 个输出长度不超过 64 token；平均 TPOT 44.226 ms |
| TTFT 与 TPOT 同时失败 | 22 | 平均 arrival→admission 914.659 ms、TPOT 42.223 ms |

D863 的原生控制诊断进一步显示，短输出 TPOT 违约与控制路径重叠；四类控制
区间的均值为 demand/acquire 66.909 ms、source snapshot 56.088 ms、HOST
preparation 50.765 ms、pending GPU reference 39.825 ms。这些区间是控制调用
的观测包络，不是可以从请求时延中直接扣除的 GPU kernel 归因。

D995 的“提前处理已到达远端需求”候选把 dispatch-visible Remote 从 509 降到
0，但实际传输仍为 505 次、1,172,100,920 bytes，联合达成反而降到
3,529/4,000（88.225%）。因此，下一步不再重复提前文件准备或扩大 graph/token
预算方向。

## 2. 语义依据

官方 vLLM 0.30.0 `EngineCoreProc.run_busy_loop` 的顺序是：先调用
`_process_input_queue`，再调用 `_process_engine_step`。官方输入处理函数在有
工作时仍会排空当前可用的 `input_queue`，所以输入队列中的同步 utility 会在
下一次原生模型步之前执行。官方源文件 URL 为：

`https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core.py`

本机用于正式回放的源文件 SHA-256 为
`268bf40534a853812867ae8e80040f190983cb364b41182b55969aace20c93f5`；PrimeLoRA
当前交错实现 `faaslora/scheduling/core_interleave.py` 的 SHA-256 为
`020cc411a4ef3b67f2b0e6d39da9ae844189dae957f886def9a34a31510a77dd`。

当前 `atomic_control_yield_v1` 已经在每个 PrimeLoRA 控制事务返回后让出一次
原生循环，但事务本身仍在 core thread 上同步完成。因此该机制解决的是“同一
输入轮次堆积多个控制调用”，并不解决“单个控制事务在模型步之前占用较长时间”。
D691 已经排除了当前版本仍然在有工作轮次中批量堆积多个 utility 的解释。

## 3. 唯一候选变化

登记一个新的、单因素候选策略（名称待实现时冻结）：

> 当 `EngineCoreProc` 已有未完成模型工作时，将**可证明为只读、无资源拥有权
> 变化、无请求生命周期变化**的 PrimeLoRA utility 从输入队列移入 core-owned
> deferred queue；先执行一个原生 engine step，再按 FIFO 处理 deferred utility。
> utility 的原始 Future/回包身份、异常和调用参数保持不变。

候选只包含以下类型：

1. `ieee_scheduler_observation`：返回当前 scheduler/KV 观测，不写入调度器；
2. `collective_rpc(ieee_gpu_reference, operation in {`\
   `source_snapshot`, `routing_source_snapshot`, `source_identity_snapshot`,
   `request_source_snapshot`})`：仅读取来源身份或快照。

候选明确**不**延后：

- `ADD`、`ABORT`、`EXECUTOR_FAILED`、`WAKEUP`；
- `ieee_pending_admission` 的 register/bind/withdraw；
- `ieee_pending_gpu_reference`；
- `ieee_request_retirement`；
- `ieee_prepare_host`、`ieee_prepare_file_host`；
- `ieee_admit_and_demand_load`；
- `ieee_transfer_event`；
- 任意未知 utility 或未知 `collective_rpc` operation。

这些排除项分别涉及请求生命周期、pending ownership、GPU 引用、释放栅栏、
HOST/GPU 物化、admission、传输压力或未知副作用；在没有新的原子协议前不能
通过排队顺序改变其线性化点。

## 4. 不变量与可证伪预测

候选实现必须满足：

- deferred queue 只由唯一 native core owner 读写；不启动额外控制线程；
- FIFO 只在同一 deferred queue 内保持；ADD/ABORT 与 deferred utility 的
  相对顺序不得伪造为已完成；
- 不允许在 deferred utility 尚未执行时返回成功回包；
- 快照在实际执行时重新捕获，不能复用过期 view；
- 发生 shutdown、executor failure 或异常时，deferred utility 必须收到原生
  failure/未完成身份，不能伪造成功；
- 不改变 IEEE 九个公式、tier authority、admission、handoff 或 residency
  语义；只改变只读观测的执行时刻；
- 每个请求的真实 TTFT、TPOT、E2E、传输、GPU 生命周期和失败分母仍按冻结
  协议计量，不能扣除被延后的控制时间。

若假设成立，3B W2 中短输出 TPOT-only 违约应下降，且不会以新增 TTFT-only
失败、正确性失败、pending/retirement 泄漏或 GPU-s 增长为代价。若联合达成不
提高、TTFT-only 明显增加、或任何 ownership/回包不变量失败，则候选撤销，
不通过继续增加分类项来追求结果。

## 5. 最小资格测试（GPU 运行前）

新增测试必须覆盖：

1. 有 work 时只读 observation/source snapshot 被延后，原生 step 先发生；
2. 无 work 时只读调用保持原生立即处理；
3. ADD、ABORT、pending register/bind/withdraw、GPU reference 和 retirement
   不被延后且保持原顺序；
4. deferred utility 的参数、异常、Future callback thread 和回包 ID 不改变；
5. snapshot 在执行时重新读取，不复用入队时的状态；
6. shutdown、executor failure、队列空转和多个 deferred utility 的 FIFO；
7. 未知 utility/operation fail closed；
8. 安装仍拒绝未知 vLLM source AST、非 UniProc、TP/PP/DP 非 1 和重复 owner。

测试只证明语义，不证明性能。任何实现还必须通过现有完整测试集和一次
100-request 3B native correctness/identity smoke，才可登记新的 4,000-request
开发执行键。

## 6. GPU 验证与撤销规则

若静态资格通过，沿用 D992 的 3B W2 trace、adapter subset、生成合同、资源
包络和远端内容；只改变候选开关并使用新 execution key。单次完整 4,000 请求
运行后核对：4000/4000 terminal/native token/adapter binding、n_correct（若
仍不可得则不得宣称“全部正确”）、联合 SLO、TTFT/TPOT/E2E 分解、实际 GPU-s、
deferred/urgent counters、pending/retirement/transfer receipts 和 OOM/磁盘门槛。

D992 作为开发参考，不能因新候选失败而覆盖。新候选只有在合同有效、数值正确
且联合 SLO 与物理 GPU-s 按冻结 G1/G2 口径有可解释改善时，才进入后续完整
W0/W1/W2 评估；否则保留失败证据并回到主线的下一个有证据瓶颈。

## 7. 当前决定

本文件是候选登记和语义边界，不是性能结果，也不是接受该策略的声明。它把
D992/D863 的相关性转化为一个可撤销的单因素实验问题，避免再次重复已关闭的
提前准备、整段 HOST 异步化、图捕获或 token-budget 方向。
