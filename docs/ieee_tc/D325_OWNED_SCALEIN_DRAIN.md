# D325：按论文语义完成撤销路由、排空与物理回收

2026-10-07—08，7B 优化主线。承接 D324 的单一假设，不修改 IEEE 九式、
三信号扩缩容律、阈值、38 秒 cooldown、服务/准备 profile 或冻结指标 V1。
这是实现与并发正确性检查，尚不是新的 GPU 性能结果。

## 依据和待证问题

D323 完整 W0 相比 D318 的平均/P95 TTFT、P95 TPOT 改善，但物理 U 增加
200.543989 GPU-s（1.29%）。两轮都实际缩容两次；D324 已分解物理区间，
不把预测误差改善当资源改善，也不继续重复拆分相同日志。

IEEE LaTeX 附件 SHA `35da2d89941452df2959b022aedddd4a984e45b8c15efb9951b5295deb198b90`
第215行规定先撤销路由可用性，再排空已接纳工作，最后释放。D48
`faf5c855b0cbb2e13d841ff711ac7aaead43a5ae` 只完成 idle-only 回收。
D323 的311.818820秒事件已满足控制条件，但没有空闲副本，因而未撤销路由。
到下一次决策的间隔不是可以直接声称节省的资源量。

[Ray 2.59.0 官方实现](https://raw.githubusercontent.com/ray-project/ray/ray-2.59.0/python/ray/serve/_private/replica.py)
的 `perform_graceful_shutdown` / `_drain_ongoing_requests` 将排空作为显式
生命周期。只借鉴依赖顺序，不移植其轮询间隔、timeout 或强制退出配置。
历史 D94 的“退出前保留成员”、D97 的单一 quarantine owner、D296 的
独立 runtime 并行收尾约束均保留。

可证伪假设：相同控制与负载下，真正执行已经决定的缩容，能否减少被放弃
的回收机会、降低完整 U，且不损害正确完成与服务质量？CPU 检查只能证明
状态与依赖关系；闭环资源/延迟结果必须由原4000请求回放决定。

## 单一回收所有权

复用既有 runner、InstancePool、请求 reservation、native shutdown 和
物理 allocator，不建立平行调度器，不修改 autoscaler.py 或 instance_pool.py。

1. 控制快照、决定与撤销路由之间不再插入 HOST 查询的 await。HOST 查询
   仍按原控制节奏执行，移至决定之后；没有新增轮询频率。
2. 从非 primary 的运行副本中选最少已接纳请求者，以原 last-selection/ID
   稳定打破平局；无需碰巧 active/load/forwarding 全零。撤销后新请求不能进入。
3. 正常 scale-in 与 native 所有权不确定共用一个按 engine 身份管理的回收任务。
   已接纳请求仍按原截止时间执行；completion/release 事件驱动排空，不新加 sleep。
4. draining 中仍在完成的请求不再冒充等待分配的队列；运行副本的 active
   与其 ready capacity 配对。终态失败但尚未释放的 reservation 不从 backlog 扣除。
5. 既有未完成候选状态查询也持有 native RPC 生命周期：先加入撤销前已发布的
   observation wave，再关闭后端。原等待者按既有 membership 校验拒绝过时快照；
   不靠吞掉 shutdown-induced RPC 错误重试。真正查询错误仍交给原请求并另记。
6. 准备计划、共享文件压力与 native 引用依照既有依赖收尾。物理 release 必须
   明确确认，随后才撤销 HOST budget、移除成员并记录物理缩容完成。
7. 请求中的 fatal 路径不等待一个正在等待自己的回收任务；保留原请求失败与
   设备禁用语义。quarantine 不确定性则升级同一任务，不并发重复 shutdown。
8. 全池收尾加入已存在的回收任务；调用者取消不能遗弃已经开始的回收。
   回收失败/取消时保留成员与占有证据，不在全池收尾中隐式重试未知退出。

新增 `ieee_scalein_events`，记录 withdrawal、drain、release 时刻及状态；
旧 quarantine 事件仍保留。计量不在 withdrawal 提前截止，也不从旧 U 中扣时间。

## 验证记录与边界

- 原实现的确定性反例：两个副本各已有一个请求，所有低压/cooldown 条件满足；
  期望撤销一个副本，但实际仍 `running`。原测试失败保留在会话执行证据中，
  不是性能实验；未另外落盘的原终端输出不冒称存在日志文件。
- 初次34项检查出现两个旧 `__new__` fixture 缺少运行时字段的错误：缺少
  unsettled map/lock，以及假 slot 缺少 engine。修正 fixture 的真实结构，
  不添加 production 缺省值规避检查。
- 39项专项通过，31.044秒；随后加入原子撤销顺序和 fatal 设备语义检查。
- 365项扩大回归通过，55.766秒；再核查出未完成候选查询的退出依赖并补测试。
- 最终结果和资源回执见后续 checkpoint；未通过前不启动 GPU。

新测试覆盖 busy scale-in、最少副本、无新 admission、正常请求完成、真实退出
前继续持卡、quarantine 升级、取消、未知物理释放、失败保留、同一关闭一次、
fatal 自依赖，以及既有 source read 的结束/失败。使用显式 CPU 模拟退出端点，
不称已验证真实 CUDA 性能。既有真实 native 资格与 D323 结果保持其原身份。

## 下一主线与保护

最终 CPU 回归、资源/保护检查和可回退 Git 备份后，准备单一 D326 原4000 W0
开发回放，与 D323 保持同模型、配置、profile、真实远端、输入与生成契约，
只换独占输出/缓存目录和当前回收实现。原100请求前缀不覆盖311秒的问题事件，
不再用相同短前缀声称验证该资源假设；本次完整回放仍受原 watchdog 和失败规则。
先观察真实撤销/排空/释放触发与完整完成，再比较 U、TTFT/TPOT，保留退化。
没有触发则不得归因；不把一次开发回放称正式 G1/G2 或统计领先。

仍是7B→3B→baselines→比较→消融/敏感性。数值正确性、共同 warm/Resident、
新指标下旧 Prime 对照和正式 G1/G2仍开放，不用此次修复绕过这些条件。
达标前不画图，不改论文源文或历史结果，不新增工件、trace、profile。

用户允许的只读交付缓存已由 D78/D80 完成：1581215261字节、每模型500个。
不再重复制作或复制解压池；真实传输仍计入，请求路径不打包。

## 最终 CPU 资格与下轮配置（2026-10-08）

最终426项通过，55.804秒；包含 source-read drain 新测试。下轮配置另有6项
检查通过，1.293秒，结构化证明 D326 与 D323 只差三个独占输出/缓存路径。
同一新版 vLLM、160个服务类别、24个准备类别、控制和生成参数均不变。
复用原 `run_d293_7b_prefetch_validation_100.sh` 的显式 `d326` 分支；该分支
清除 diagnostic prefix，执行原4000 W0，不因脚本历史文件名变成100请求。
新配置为 `configs/ieee_tc/20261008_d326_7b_full_w0.yaml`，SHA
`3c5ad20bf1d4ef38c74125dcdc9646436d5b18dc5afd1a3b32eb2c05d882f4f2`。
尚未启动，必须先重新检查磁盘/内存、remote 和实际生产 profile 绑定。

Serving SHA `734a51f1789861e1f2e023890057b911ae5b341da0d07bd92ceb73c4b1abc627`。
资源/测试/源文件回执在 `paper_results/ieee_tc/p2_backend/d325_owned_drain/validation_checkpoint.json`。
六个 CPU 测试资源域的实际 high/max 为3/4GiB、swap上限0；全部事件0，
确认没有进程、populated=0后按 InvocationID 关闭。最终426测试峰值
1967718400字节。未启动GPU或remote服务，没有GPU compute进程。
旧保护147项核查仍仅存在事先登记的baseline replay脚本变化，原pass为false；
没有伪称全147项原样。用户manifest、计划、V1和旧结果未发生新变化。

本次是实现候选完成，不是资源优化已经成功。下一项直接验证完整负载，
不要回到相同日志分析、输出长度标定或重复短前缀；G1/G2仍未闭口。
