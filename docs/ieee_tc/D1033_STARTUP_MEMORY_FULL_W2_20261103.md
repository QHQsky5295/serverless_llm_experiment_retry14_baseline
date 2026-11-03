# D1033：启动显存画像复用的完整 3B W2 验证

2026-11-03，承接 D1031 冷画像与 D1032 命中验证，当前完整目标 active。
本文件是当前交接，不改用户 dirty EXECUTION_STATUS.md，不重开共享 k3s。

## 唯一假设与前置证据

D1029 的完整 W2 为 3774/4000 联合达标（94.35%）、8167.260452414863
GPU-s、P95 TTFT 1890.5496881343424 ms，离共同门槛仍差26条。
其编译已热的自然副本仍约28秒原生初始化，重复显存估计是可观测开销。
D1031/D1032 的唯一候选复用实际成功 KV 预算，不采用原生建议容量，
不改模型、32槽、并发8、token轴、GPU预算、九公式、扩缩容或 admission。
官方 gpu_worker/startup_plan 路径及历史失败方向见 D1031/D1030。

D1031 source2 已完成423请求、160服务类/96准备类及真实远端审计。
D1032 仅选原前两波16请求，实际 worker 回执 status=hit，指纹与原记录
相同、producer PID4043841；原生预算8092388312 B、block池8092385280 B、
32槽与LoRA池1825046528 B未变。真实服务图仍捕获一次，原生 init16.26s，
compile0.26s。不是全部冷/热差异的单因素因果证明，数值正确性仍未闭合。
不再追加短 Full 或重新收集423请求。

## 本次冻结配置与判据

配置为 configs/ieee_tc/20261103_d1033_3b_startup_memory_w2.yaml。
相对 D1029 仅14处结构变化：三个独占输出/缓存路径、新启动政策、两个
completed-length ID、两份新画像的路径/SHA/输入及资源上下文身份。
使用既有 strict loader 实际 dedicated 配置验证160/96类通过。
启动仍用 fastsafetensors_affinity_preserving_v1，和画像的loader不同会产生
自己的冷记录；不复制记录、预跑Full、抹去编译因子或降低free-memory条件。
自然后继副本是否命中必须用实际回执，不以较快初始化推断命中。

源输入仍seed42已知开发回放，4000请求，500逻辑adapter，W2 rotation100。
trace SHA 4ea5d026da3820301e753ad6b03ea776e25a5c3f01921933bd124598eb26018d；
W2 map SHA 28c97dc7c1ffcbad49801ec19bec2a9c4cf9c4915c88a7e5163e223ece315e23。
固定输出greedy、相同prompt/target/arrival、共同提前60秒部署全部保留。
原生remote读取、传输、竞争、物化计入；使用既有只读压缩交付对象，不打包。

PLAN SHA d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83；
V1 SHA 5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22；
V2 SHA 193bf1535af1aa50f83d98dc2e944ecbdc7bd27aa96baeca82f0654e02f9357d。
D648 SLO：prompt≤506/>506，TTFT1466.050801102/2539.648668191 ms，
TPOT29.921872367/36.215515193 ms。至少3800联合达标；G1先要求全部正确，
再比GPU-s；G2另要求U≤12083.063870，才比较P95与本轮SLO差。
不计算CI、不重复同键、不挑请求阶段、不改分母；n_correct未知不填4000。

## 执行、保护和终态

复用 scripts/ieee_tc/run_d1000_3b_route_confirmed_arrival_prefetch_w1.sh，
虽然旧文件名含W1，实际明确W2。RUN_ID=d1033_20261103，
RUN_TAG=d1033_3b_startup_memory_w2_full；CONFIG为上述新配置。
保持D747 descriptor、stdlib gzip、相同已冻结本地/远端资源环境。
显式清除prefix与详细profiling开关；唯一完整回放，不做暖启动重跑。

推理service72/80GiB、swap2、CPU4–23/28–47；aux3/4GiB、swap0、
CPU2/3/26/27，独立watchdog和native-worker归属核验继续生效。
测前MemAvailable102GiB、磁盘150GiB且覆盖预测增长；运行底线16/100GiB。
不在回放期间进行压缩、构建、远端重启、备份或其他重型任务。
本轮为一个新完整执行键，旧D1029及D1031失败/成功证据全部保留。

结束先验证真实GPU释放，再按D703分析4000请求、冻结SLO、GPU-s、
阶段/每个worker冷命中及KV容量、画像类和双端transfer证据。
D703只对原生日志明确 skipped memory profiling 时允许一次真实capture，
其他仍要求两次；该分支不单独宣称本政策命中，须绑定worker回执。
47项相关CPU测试已通过，原既有警告6个；未做多余基础资格矩阵。

D810外部baseline18点与已完成适配保留，不重跑来改变对手结果。
W2的vLLM联合100%、U16110.752、P95441.635ms；ServerlessLLM98.125%、
U16084.201、P95480.724ms。它们超75%预算但原始延迟更低，不隐藏。
达标后推进同一最终配置W0/W1和完整外部对比，不继续局部启动调参。
未达标保留反例，回到完整请求阶段证据；不强行将微测通过写成G1/G2。

全部剩余范围不删减：两模型同最终版本W0/W1/W2/G1/G2、独立数值正确性、
最终基线比较、A1–A5、S1–S13、CE/Pareto、B1/B2/C1/C2/C3和两轮审稿文档、
接近目标后图表、最终V2集成。备份分支不冒充历史分歧V2已集成。
