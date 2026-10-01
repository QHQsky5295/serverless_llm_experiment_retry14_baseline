# D142：撤回周期 C-watchdog 调用栈诊断

2026-10-01。结论：诊断方法不适用于当前运行环境；撤回其生产代码入口，
保留所有失败证据，不继续用该方法回放。此前 CPU 资格通过不等于实际运行合格。

## 原始结果和状态

| 项目 | 观测值 / 状态 |
|---|---|
| 实际启动 | 7B，原 4000 请求前 1000 条，1 次，formal=0 |
| 控制进程 | 导入阶段 SIGSEGV，wall 6.17 s；service returncode=139 |
| 已到达 / 已提交 | 0 / 0；计划前缀分母仍为 1000，源工作负载为 4000 |
| TTFT、TPOT、完整 GPU-s | N/A；无完整服务数据，不作性能点 |
| 栈记录 | 仅控制进程，422 B，末次记录截断；其他角色尚未启动 |
| 资源采样 | 7 次；采样 memory peak 524550144 B，最低可用主机内存 111864242176 B |
| memory high/max/OOM、swap、保护告警 | 均为 0 |
| GPU | 各次已采样记录均无本服务 GPU context；终态 release/path cleanup 确认 |
| 远端 | 原生 1 Gbit/s/full，已发布交付不变；终态后精确身份关闭三服务 |
| 历史保护 | 147 项旧投稿资产不变；206 个启动来源引用已核查 |

采样 peak 与 `/usr/bin/time` 的进程最大 RSS 是不同观测，不混称全生命周期
连续内存峰值。未找到对应 health clock 的 transfer journal；不伪造一份空日志。
进程尚未进入业务回放；不存在用此轮推断缓存、传输或调度性能的依据。

## 最小导入对照：定位诊断自身的失败

禁用可见 GPU、不加载模型、不启动回放，只导入同一个 runner。两次均为
Anaconda CPython 3.12.12，原解释器与原库路径，不更换推理依赖。

| 诊断开关 | 结果 | 定位证据 |
|---|---|---|
| 关闭 | `IMPORT_COMPLETE`，解释器正常退出 | 完整导入日志 |
| 开启 | SIGSEGV，未完成导入 | `faulthandler_thread → _Py_DumpTracebackThreads → dump_frame → PyCode_Addr2Line(co=NULL)` |

GDB 自身的返回码为 0 不代表被调试进程成功；上述判定使用实际 inferior 的
终态和调用栈。第一次对照因宿主 GDB 错误加载 Conda 的旧 libstdc++，在两个
inferior 启动前失败；原日志保留。第二次仅为 GDB 清除 LD_LIBRARY_PATH，
对被调试解释器恢复原值，再执行上述一对；没有安装、升级或全局权限修改。

本机证据直接证明启用方法可在导入时令诊断线程崩溃。第一次回放没有 native
core dump，因此不声称已证明其每个底层细节与第二次相同。
CPython 上游也报告过无 GIL 保护的 watchdog 栈读取相关崩溃，但版本及复现
条件不同，不能自动认定是同一缺陷或已有修复适用于本机。
[上游报告 158200](https://github.com/python/cpython/issues/158200)、
[上游报告 131580](https://github.com/python/cpython/issues/131580)。

## 处理与主线边界

- `faaslora/__init__.py` 与 `faaslora/utils/logger.py` 已逐字恢复到 D141 的
  `d12075c6cf283204b7d7023cd4598872c664d981`，没有改变任何服务策略。
- 删除仅为该方法新增的活动测试文件；源码和通过的 CPU 测试证据完整保留在
  `f5f62c2c9fe0f2b1daa19b32134cd69300177065` 及其资格归档中，未抹去失败历史。
- 不升级 Python、不修改 ptrace、不重跑 D140 100 Hz、不延后启动 watchdog
  来绕过故障，也不增加 cap/deadline。
- 诊断资格文档是已封存历史，不回写其原始状态；本报告明确撤销 GPU 适用性。
- 本轮完成的是测量方法排错，不是 PrimeLoRA 性能改善；D137/D138 完整回放、
  D139 回复拾取延迟仍为有效既有证据，不能重复解析大原始 JSON。

下一步只针对仍缺少的控制路径证据选择安全观测：可评估持 GIL 的 Python
`sys._current_frames()` 线程采样，并记录实际采样时间、间隔及进程 CPU 时间；
它可能在 GIL 忙时延后，必须如实报告，不能解释为无偏 CPU 占比。先用既有
导入对照与最小 CPU 测试验证，之后才允许一次当前版本短回放。此替代方法
尚未实现或资格，不能直接启动。不得陷入诊断工具局部优化，取得足够证据后
立即回到一个可证伪的服务瓶颈及普通 Full 验证。

两模型性能、3B TPOT/输出差异、数值 adapter 资格、warm/Resident、基线、
M1/M2、A1–A5、S1–S13 均未闭口。一次性交付缓存不重建。
