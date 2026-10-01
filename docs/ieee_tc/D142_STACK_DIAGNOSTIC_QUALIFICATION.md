# D142：当前控制路径的周期调用栈诊断资格

2026-10-01。开发诊断，不是正式性能结果，也不是新的调度优化。

## 问题与选择

D139 从既有完整结果恢复的 generation RPC 回复拾取延迟均值为：7B
1095.831894 ms、3B 1264.889093 ms。这不能唯一归因于 JSON、GIL 或
source inventory。D140 的 100 Hz 全子进程 py-spy 诊断有 162 条滞后警告，
最大 46.84 s，异常结束后无最终 profile，不能用于推断热点。

本次只替换诊断观测方法：在通过启动门槛的非正式 prefix 回放中，使用
CPython 3.12 `faulthandler.dump_traceback_later(2.0, repeat=True, exit=False)`。
每次记录直接写入各进程独有文件，不等待实验末尾统一导出。描述符保留到
解释器结束。该接口使用 watchdog 线程；其 Python 栈信息有线程/帧上限，
没有局部变量，不能表示 native extension 内部工作。
[Python 官方说明](https://docs.python.org/3.12/library/faulthandler.html)。

不修改 ptrace/sysctl、信号处理、宿主权限或实验控制策略。默认不开启；
显式 formal=0、诊断前缀、已获准执行的 receipt 与实际服务 cgroup 必须一致。
现有启动 wrapper 只在控制器启动门槛之后设置环境开关；子解释器在导入
faaslora 时分别记录 PID、启动 ticks、线程身份及 cgroup。

## 已完成的资格证据

| 检查 | 结果 | 可支持的结论 |
|---|---:|---|
| 相关 CPU 测试 | 335 / 335，75.091 s | 受影响逻辑回归通过 |
| 默认关闭 | 通过 | 不导入额外 logger，不生成诊断文件 |
| formal / 错误开关 / 外部 scope / 非 prefix / 未放行 | 全部拒绝 | 不允许这些条件下误启用 |
| 两独立进程异常 `_exit(23)` | 均保留已完成 dump | 不依赖正常退出导出 |
| 同进程重复调用、已有文件 | 幂等；已有文件不覆盖 | 保持原证据 |
| CPU 测试实际限制 | high 3 GiB / max 4 GiB / swap 0 | 非推理负载隔离 |
| 测试 cleanup | 空 scope 关闭，六项 memory events 全零 | 无遗留测试进程 |

原始证据在 `results/ieee_tc/p2_backend_qualification/d142_20261001/`。
没有重跑 D141 的 1,071 项回归；它的已封存正确性证据复用。

## 单次短回放合同及解释边界

下一步沿用 D137 的 7B Full 配置与 D122 的原始 4,000 请求前 1,000 条索引视图，
真实远端已发布工件、D89 初始化、cap=2、seq=2、GPU slots=4、CPU slots=24，
deadline=1800 s 均不变。仅改新输出/缓存路径、prefix 数量与诊断观测方式。
仍计入真实传输和竞争；不重建一次性交付缓存，不新增 trace/权重/profile。

必须从实际 PID、worker 日志和栈文件核查 controller、frontend、planner、
GPU core 的覆盖；fork-only 或未导入本包的进程不能凭父进程开关宣称已覆盖。
观测包含 idle/wait 线程：栈出现频率不是 CPU 时间占比，不能把等待栈称为
计算热点，也不能把缺失角色视作零开销。每两秒不是严格同步时间线，不反推
单请求精确阻塞区间。诊断开销尚未定量，无“零扰动”或“开销更低”的实测结论。

终态后先清理与校验，再给进程角色/观察次数/主要栈的诊断表；失败和缺失保留。
即使正常完成，也不能用这个 1,000 请求诊断替代普通 Full、数值 adapter 资格、
共同 warm SLO 或正式比较。若证据仍不足，不猜测增加并发或 deadline。

状态：CPU 资格完成；本文封存时 GPU 诊断尚未启动。后续运行状态另记 ledger
与运行报告，不回写本资格表为性能结果。两模型性能、warm/Resident、主比较、
核心消融和敏感性仍未完成；baseline 继续暂停。
