# D330：规划实测信息的双消费者发布

2026-10-08。7B 单一开发候选，485 项 CPU/传输/生命周期回归通过；
尚无 GPU 性能验证或 G1/G2 达标结论。保留 D326 完整回放及 D328 诊断，
不重复诊断来选择有利结果，不绘图，不恢复 3B 或外部基线。

## 问题与可证伪假设

D328 的 1,403 次完整请求查询平均总历时 119.264 ms，其中原生开始前
62.946 ms、原生函数体 11.905 ms；它们是 RPC 区间，不是请求 TTFT，
并发区间不能相加。D329 已拒绝无 storage 生命周期证明的 epoch-only
物理表示缓存。D289 加载身份传播、D301 同 epoch 保留、D316 GPU-copy
保留都已存在，本轮不重新实现这些机制。

当前 initialized residency planner 每次已经读取完整、实时的 native
HOST allocation/alias 图与 GPU pool；路由仅在自己请求过同一 scope 后
才保存测量。此前规划测量未发布给路由，因而可能重复查询。

假设：让同一份已经发生的权威测量供两个消费者使用，可减少后续请求
重复进入原生查询队列的次数。不会取消 planner 的新鲜读取，不复用其
选择结果或预算许可，也不声称可以消除全部等待。

## 原始实现依据

[vLLM 0.30 core](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core.py)
的输入队列与 utility dispatch 在 core loop 中处理；async 调用不意味着
原生观察没有排队代价。[官方 LoRA worker manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)
区分已有对象复用、加载和 LRU 移除，并说明修改发生于单线程 core。
本轮联网核对了对应 tag 的函数体；这些只提供减少冗余观察的依据，
不直接证明 Prime 的协议或借用任何性能数字。

## 唯一改动与不变量

1. `NativeSourceSnapshot.request_view` 仅投影已验证完整测量：保留原始
   owner/epoch/clock/capture time，清除全局 HOST union 及其他目标的 footprint。
   identity-only 或已经 scoped 的记录不能通过此接口伪造完整测量。
2. planner 收到本次 native 返回后，把同一份 payload 发布给现有 confirmed
   route cache。没有新 RPC、轮询、TTL、睡眠、阈值或预测器。
3. 只接受当前 pool 中同一 slot/engine、同一物理 GPU 的观察；迟到 epoch/
   timestamp 不覆盖更新信息，同 epoch 的身份/footprint 矛盾仍报错。
4. 路由条目只含已测 scope。未知或 GPU 未确认条目不进入服务类别缓存。
   HOST 仍沿现有 epoch 规则失效，GPU 仍需 D316 exact-copy 条件。
5. 选中后的来源核验、原子引用、取消/释放和冲突重选不变。历史候选测量
   不冒称所有副本同时锁定，更不作为当前物理预算或 admission 许可。
6. full planning 图、实际预算、九式、D/T/O 更新、Q 排序、autoscaler、
   token 合同、配置和真实远程传输均不改。
7. 记录 publications、seeded scopes、hits；这些计数只说明机制触发，
   不能直接换算成减少的延迟或 GPU-s。新增 controller 解析/投影开销也由
   实际服务承担，必须在完整回放中检验净效果。

## 验证记录

新增验证覆盖完整→scoped 等价、GPU/HOST 复用、无额外 RPC、时间戳保持、
迟到事件、slot/engine/GPU 变化、同 epoch 矛盾、缺测量和物理图损坏。
真实 runner/planner/collector 组合保留原选择目标与预算，同时复用该次
测量；原生来源、取消、生命周期及请求回归一并执行。

首次 81 项中 3 个新增测试失败：两项错误地要求 identity_view 返回原
Python 对象，另一项把 fixture 原本 false 的 pinning 又设成 false。
改为比较已发布对象未被替换，并实际翻转 pinning。只修正测试，没有因
测试失败改变生产候选或放宽检查；首轮日志完整保留。

最终 485 项通过，测试执行 60.858 s。两组有重叠，不相加为独立实验。
CPU 环境 Torch 2.8.0+cu128，不是原生 Torch 2.13 GPU 性能资格。两个实际
资源域都是 3/4 GiB、swap0、CPU2,3,26,27；峰值分别 1887268864 和
1919430656 B，high/max/OOM/swap 全零，核对身份和空进程后均已关闭。

## 下一次实际执行

复用既有 guarded runner 的 `d331`：原 7B W0 的前 100 条索引、同 500 池、
同 D326 模型/160 service 与 24 preparation profiles，只有三个输出/冷缓存
路径不同。60 秒通知、真实远端已发布工件不变；无详细采样器、不造新 trace。
短测须有完整 token/binding/时间字段、实际释放、零来源冲突错误，并确认
planning publication/hit 真正触发。它不能决定性能胜负或外推完整 GPU-s。

短测合格后对同一候选做普通完整 4,000 W0，按 V1 比较已封存 D326 的
U、TTFT/TPOT、阶段等待、失败与输出差异。没有新证据时不反复做短测；
未改善则记录实际边界，返回主线瓶颈，不扩大此局部改动。

数值正确性、共同 warm/Resident、新旧 Prime 新指标验收仍 OPEN。
一次性远端缓存已由 D78/D80 完成，本轮不重建；计划和 V1 原 SHA 不变。

D331 启动前静态核验已通过：实际 child model 与 D298 已验证合同完全
一致，160/24 profiles 由生产 loader 读取通过；配置只有三个独占路径不同。
当时磁盘空闲 161765138432 B、主机可用内存 112507600896 B，保留原门槛。
核验域峰值 514015232 B、events/swap0，已空域关闭。保护清单仍只有已知
baseline replay 差异，用户 manifest 未动。此核验不代表已经启动 D331；
真正启动前仍须重新读取计划/状态、核查当前安全门槛和远端服务身份。
