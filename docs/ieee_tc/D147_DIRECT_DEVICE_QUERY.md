# D147：去除扩容设备状态查询中的同步命令行开销

2026-10-01；P2/P3 开发候选，不是正式系统比较。父版本
`7f42cbdd143ab917927237abcf136bc2dc18e608`。IEEE 九式、生成、配置、
真实远端交付和指标 V1 不变；只修改既有 runner 的设备提示查询。

## 历史证据与假设

D145 的成功请求平均 dispatch 等待为 675.934327 s，native 生成区间
约 4.712198 s；这些是条件统计，不证明某一个控制步骤解释全部等待。
D143 已有诊断栈可以进一步定位实际阻塞，无需重跑 profiling：

| 业务期控制器主线程采样 | 出现数 |
|---|---:|
| 全部业务期样本 | 570 |
| 栈包含 `_gpu_runtime_snapshot` | 43 |
| 其中明确调用同步 `subprocess.check_output` | 42 |
| 其中位于请求执行路径 | 16 |
| 其中位于请求 reservation 收尾路径 | 16 |
| 其中位于周期性显示路径 | 10 |

这里只计出现次数，不是 CPU 时间或 wall-time 比例。采样需要 GIL、有偏，
包含等待且可能缺失忙区间。43 是 inclusive 计数；D143 报告的 42 是
最近项目帧计数，两者口径不同。完整 caller 与源 SHA 保存在诊断数据中。

当前初始 TP=1 stack 的 monitor 只覆盖初始卡。扩容卡不属于该 accounting
范围，旧 helper 每次改走 `nvidia-smi`，同步等待进程启动、查询和退出。
可证伪假设是：去掉这个进程边界，保留同一设备的实际查询及状态语义，
能缩短控制器一次同步占用；能否减少 Full 排队则必须另外完整验证。

[NVIDIA 文档](https://docs.nvidia.com/deploy/nvidia-smi/index.html)
说明 CLI 的许多功能由 NVML 提供，并建议程序使用 NVML 接口；
[vLLM 的 CPU/GIL 分离分析](https://vllm.ai/blog/2024-09-05-perf-update)
支持检查同步主机工作对服务推进的影响，但其加速数字不外推至本机。

## 修改边界与第一次候选的拒绝

仅在 IEEE 路径原本将调用 CLI 时，直接读取指定物理索引的 NVML memory
和 GPU busy rate。不扩大 `monitor.devices`，不改预算，不缓存跨调用读数，
不改变刷新周期、准入、reservation、超时、重试和 native ownership。
既有 monitor 分支、非 IEEE 路径不动；已存在的 runner NVML 初始化/关闭
生命周期复用。无法得到可选提示时仍为缺失，不伪造成空闲、也不转回 CLI。

权威 IEEE 路由仍从 worker 的 GPU UUID 读取独立 utilization/source 观测，
不会把旧 monitor 的显存使用比例改称 GPU busy rate。这个候选也不修写
其他历史提示的语义。读取失败不改变真正的 native 预算/可行性检查。

第一次候选虽明显更快，但实机发现其默认 v1 memory.used 含驱动保留量，
与旧 CLI 的字段不同，故**不接受第一次候选为等价替换**。保留原代码、测试、
微测和失败原因。设备 3 的独立有界 API 核查为：v1 used=489,160,704 B，
v2 used=15,400,960 B，reserved=473,759,744 B。

第二次候选显式请求 `nvmlMemory_v2`，使用本机实际返回的 used 字段；
32 对 idle 查询均在 CLI 的 0.5 MiB 显示舍入范围内，busy rate 相同。
生产读取保留实际字节，不为对齐文本显示而硬编码扣除或舍入。
[NVML 内存接口](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html)
区分 v1/v2 与 reserved 字段；当前在线说明对 used 是否包含 reserved 的
文字与本机观察不完全一致，因此以已锁定驱动/绑定的实测兼容性为界，
不声称任意驱动均等价。初始 monitor 原有读法保持不变。

## 最小验证结果

原版本六项针对性测试出现六次失败，定位到仍调用 CLI/未获得所需观测。
第一次及修正后候选各六项均通过；测试覆盖刷新、物理索引、NVML 初始化、
memory v2、空卡、缺失/异常、原 monitor 范围和 legacy 路径。
首次相关回归 543 项通过；修正后同组 543 项再次通过（22.880 s，整命令
33.41 s、峰值 RSS 1,193,040 KiB），未重复其他无关测试。
测试与微测均在实际 3/4 GiB、swap=0、CPU 2/3/26/27 资源域执行，逐项
确认空资源域后关闭，未启动 CUDA context、模型或远端服务。

最终候选在正式 native Python 环境做每设备八对交错查询（不是八次服务重复）：

| 设备及路径 | 旧查询中位 ms | 新查询中位 ms | 旧均值 ms | 新均值 ms |
|---|---:|---:|---:|---:|
| 初始卡 0，原 monitor 分支不变 | 0.1160 | 0.0494 | 1.0203 | 0.1448 |
| 扩容卡 1 | 59.7824 | 0.0923 | 63.1803 | 0.3172 |
| 扩容卡 2 | 66.9568 | 0.1295 | 70.3188 | 0.6145 |
| 扩容卡 3 | 61.0174 | 0.1069 | 66.0972 | 0.5850 |

全部调用（含首次初始化）保留。卡 0 路径未改，它的差异不能归因于优化；
微测存在 warmup、调度与采样变异。不产生 replay CI、不把查询耗时改善
直接乘以请求数来推算整轮收益。NVML 本身仍是同步驱动调用，不称零阻塞。
按计划 §11 和 academic-plotting，采用精确诊断表，不制作整系统加速柱图。

第二次微测 SHA：
`15e915f8c1ea29ce4fd43cf3dab7b3924b2857c7cc2bbd14f39b139bfea7dc51`。
原始目录：`results/ieee_tc/p2_backend_qualification/d147_20261001/`。

## 决定与下一步

保留第二次候选进入普通 Full 验证；尚无整系统 TTFT、TPOT、GPU-s 或 SLO
增益结论。先完成来源/保护/秘密检查与可回退备份，再只清理可证明可重建
且无进程引用的缓存以恢复磁盘余量，随后沿用既有 7B Full 配置完整回放。
不增加 capacity/deadline，不重建权重、负载或远端交付缓存，不覆盖旧结果。
D145 四个 RPC 超时原因仍开放；D146 已补未来错误边界，不能追填旧证据。
3B 性能、TPOT/输出差异、数值 adapter、warm/Resident 仍未闭口；基线暂停，
M1/M2、A1–A5、S1–S13 未由本次微测完成。
