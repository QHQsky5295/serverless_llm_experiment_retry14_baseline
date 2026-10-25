# D853：完整文件事务候选接入及有界验证

## 结论与主线

接续 D851/D852，完成 planning、activation/HOST allowance、主动 GPU/HOST
文件引用和 file-plan 生命周期的异步事务接入。未新增参数候选、局部 GPU
回放或 profiling；未改变 IEEE 九个公式、共同负载、远端交付和资源预算。
候选现在可进入一次完整 3B W2 评估，尚无本候选的性能结果。

当前参照仍是 D846：4000 条原生计数合格，联合 3715/4000=92.875%，
物理 U=11929.629332 GPU-s，预算=12083.063870 GPU-s。联合不足95%，
不能结案。D711 独立数值 LoRA 正确性仍未知，n_correct=null，不能将本轮
CPU 所有权检查或原生计数等同于数值验证、正式排名资格。

完整 PLAN SHA：d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83。
冻结 V1 与单轮 V2 已完整读取；目标仍是共同联合 SLO 下减少物理 GPU 时间，
以及预算内改善 W1/W2 尾延迟。CE 不参与主工作点选择。

## 实际接入

| 路径 | 保留的所有权与时序 |
|---|---|
| 共享 control-round 文件观察 | 同轮仍只读一次新鲜完整文件视图；await 后重查真实 slot/engine 身份和正在执行的计划，最后注册不 yield。新副本留待下一正常轮次 |
| 初始化/常态规划 | 完整 native graph 与 file view 仍分别采样；不是跨 owner 原子快照，实际执行继续重查。使用既有不可变 FrozenPreparationDescriptions，不缓存动态容量 |
| 激活 HOST 预留 | 配置、预留均在原异常清理范围内；成功结果先登记，再传播取消。清理先 join startup/preparation，复合观察及退还未采用额度仍为一个原 owner 事务 |
| 原生 HOST 额度 | reserve 成功后在取消前交接 member 与原 pidfd；未确认安装保留额度。只有真实进程退出才 retire；共享别名串行关闭一次 fd |
| 主动 GPU fallback | 同一原锁中观察并 pin 实际剩余最快文件，部分失败在该事务内退还；成功但取消时先接管再释放；原生结局未知继续保留 pin |
| 主动 native HOST | 当前观察与文件获取仍为一个事务；不持线程锁跨 native RPC。取消后的已成功获取也由原清理路径释放 |
| file plan | register/bind/unbind/finish 进入异步事务；注册成功但取消仍关闭真实 plan；注册失败不假造 close。沿用 D852 精确目标 start hold 与 writer join→close 边界 |
| 通知计时 | 已完成 IO 后的异步通知失败仍是失败，但不再重写真实 io_joined_at；另存 notification_failed_at，压力 finish 仍 join |

激活 trigger_at 现在位于 HOST 配置/预留之前，覆盖实际激活等待，不将其移出
activation-to-ready 计时；这是时序定义修正，不是宣称启动性能已改善。

保持同步的剩余入口已按生命周期核对：`_current_coord_metrics` 的三个普通
调用属于同步 legacy/backbone/transformers 路径；实际 IEEE `run` 只合并
coordinator views，不调用这些文件快照。IEEE 的该汇总位于
`_finish_ieee_main_scenario` 对 runtime shutdown/stack stop 的尝试之后。
成功退出后 workspace 删除和最终 inventory 同步执行；退出失败则保留错误，
不声称已清理。`_materialize_remote_adapter` 等文件发布事务本来就在既有
IO worker 内，不把整个 HTTP 操作放入 owner 锁。只读 owner_id/roots/固定
策略属性不是磁盘事务。此为已检查实际 IEEE 路径，不是全程序无阻塞证明。

## 依据与未重开的方向

复用 D843 全长栈、D849 阶段分解、D850 真正 owner 锁依赖、D851 原语与
D852 已完成请求路径；D733 是暂缓完整改造，不是实测否决完整候选。
不重开 D628、D595、D827、D822、D844 或 D840 已关闭的局部方向。

本轮重新核查 [Python 3.12 asyncio 官方说明](https://docs.python.org/3.12/library/asyncio-dev.html#running-blocking-code)
和 [vLLM 官方 CPU/异步优化分析](https://vllm.ai/blog/2024-09-05-perf-update)。
同步阻塞可延误同一事件循环；executor 边界是实现手段，不证明 CPU 工作消失。
vLLM 的进程隔离和其测得的加速比不能移植为本项目线程化收益。

可证伪假设仍只有一个：完整业务期文件事务移出控制事件循环，能减少控制/
请求停顿，并改善当前3B W2联合SLO，同时保持既定GPU预算。一次全长结果
决定接受、撤销或进一步解释，不另做半接入版本的短回放。

## 验证、失败与资源

沿用 D852 受限入口，CPU2/3/26/27，MemoryHigh/Max=3/4GiB、swap0、
CUDA不可见。真实微型文件/pidfd fixture；native模型由原可控夹具提供。

| 执行 | 实际结果 | 内存峰值 B | 处理 |
|---|---|---:|---|
| attempt1 | 26失败、30通过，主动中断 | 835801088 | 重复 deepcopy 不可变 mappingproxy 导致规划失败；两项旧测试固定空转未等待 executor。保留完整失败日志与源码 |
| attempt2 | 133通过，主动中断 | 3434598400 | 后段 launch fixture 意外使用真实规划进程，引入依赖编译；high=32575、max/OOM=0。只终止本轮已核验进程，不扩大包络 |
| attempt3 | 56通过、1失败 | 818356224 | 只跑未闭合启动/退役、受影响通知及新增取消检查。旧退役 mock 缺实际 file owner；改为真实文件租约 fixture |
| attempt4 | 1通过 | 721338368 | 仅重查该退役文件引用测试 |

最终覆盖187个不同检查，其中9个新增；不把重复执行累加为性能样本。
attempt2 的前130项完整通过并复用，其余按后续执行补齐；没有重跑全部168项。
attempt1/3/4 high/max/OOM/swap均0，attempt2 swap/max/OOM0，节流如实保留。
每轮19条既有依赖警告。所有源码/执行入口/SHA/失败记录保留。

新增覆盖：移除/替换副本、重叠 control round、取消实际 native HOST 预留、
取消真实 retire 与共享 pidfd、取消 activation 预留、取消 file plan 注册、
取消 native HOST 文件 pin、取消 GPU fallback pins、异步通知失败计时。
旧断言不放宽：改为真实 deferred 结束事件，不以固定次数 sleep(0) 假设
线程已经执行；launch 路径使用既有相同序列化 selector 夹具，实际进程传输
资格不在此重跑。数值/性能资格仍独立。

## 数据、备份与后续

原始记录：`results/ieee_tc/p2_backend_qualification/d853_20261025/`；
curated：`paper_results/ieee_tc/p2_backend/d853_async_file_remaining_integration/`。
147项旧结果与用户manifest须闭合复核，明确文件检查点备份，不force、不stage
用户manifest。共享k3s/远端服务未改；没有图表、工件/负载再生成或GPU任务。
磁盘约171.7GiB可用，D848已回收22.174GB，不重扫或删除未判明数据。

下一：完成本检查点备份后，直接复用 D846 完整 W2 入口，以当前完整候选
新版本运行一次。相同 trace/工件、SLO/Resident参考、硬资源包络及单轮规则
不变；先清理、核验、记录，再判断性能。完整同版本两模型 W0/W1/W2、
主基线/dLoRA、A1–A5、S1–S13、两轮审稿文档/图表和最终 V2 同步仍未完成。
