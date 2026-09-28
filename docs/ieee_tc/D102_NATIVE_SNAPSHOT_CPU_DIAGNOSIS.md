# D102：状态解析离线诊断与运行时采样资格

2026-09-28；开发诊断，不是正式性能结果。生产实现保持 Full8 的
`2f1bc4ba1d9fb3a1b17428390c4c3c63dbb09f1d`，本轮没有加入优化补丁。

## 问题与判定

Full8 完整提交 4,000 请求，3,970 条成功且原生生成合同匹配，30 条超时。
成功请求的平均 dispatch wait 为 721.192 秒，native service TTFT 为
0.485 秒。这两个条件统计不证明排队的具体原因；失败请求不能删掉。
已有父进程 CPU 观测提示应测控制路径，而不是直接改超时或增加重试。

D96 已共享同一时刻的 native source RPC，但每个等待者仍执行
`NativeSourceSnapshot.from_native`。本次假设是重复的纯解析/验证可能占用
事件循环。假设尚不足以接受实现改动：离线测量只确定一份真实旧快照的解析成本，
没有测到 Full8 每个路由波次的规模、频率和总 CPU 占比。

**结论：暂不实现解析共享，不立即盲跑下一次资格实验。先对保持原策略的完整
运行做 CPU 采样，分别看业务阶段和终态后的结果序列化。**

## 输入资格与测量

Full8 已留存的 readiness 快照没有完整 `native_footprints`，最初探针因此
显式拒绝输入，没有用伪造 footprint 补齐。随后复用 D88 已验证 profile 原件
`3b_admission_source_attempt4.json`；SHA 与 D89 初始化 manifest 一致。
291 个唯一记录中 288 个仅含 readiness，3 个含完整 footprint；按实际 owner
选最大 allocation 图，最终为一个 owner、8 个注册 adapter、1,792 个 HOST
allocation。不是 Full8 的 32-adapter 原生 CPU cache 状态。

计时区间只有纯解析，无模型、RPC 或工件 I/O。加载旧 JSON、选择输入和哈希
均在计时外。三个重复是同一输入的微测，不是独立 workload，也不提供主结果 CI。

| 同一快照解析次数 | 重复 1（ms） | 重复 2（ms） | 重复 3（ms） |
|---|---:|---:|---:|
| 1 | 2.460 | 4.465 | 2.536 |
| 32 | 83.054 | 95.047 | 77.491 |

所有输出相同，结构化结果 SHA 为
`6c81a4db66669bad04a9e7cf0600083d732a06976f16fcffd5c6ea727b39d371`。
另一次 cProfile 插桩的 32 次解析为 0.287 秒，其中 `_footprints` 累计
0.279 秒。插桩会增加开销，不能把该时间与上表混用，或把函数占比当实际控制
进程的占比。验证时 receipt 时间设为原 captured 时间，只检验解析，不模拟鲜度。

## 采样工具资格

复用现有工具，没有安装/升级环境。以实际 vLLM 环境的 Python 3.12.12
启动 3 秒纯 CPU 子进程，不导入模型或服务，在 3/4 GiB、swap=0、CPU
`2,3,26,27` 的独立资源域内检查限制。

| 工具 | 调用栈采样 | 子程序 | 启动器收尾 | 决定 |
|---|---|---|---|---|
| 已安装 py-spy 0.4.1 | 294 samples，0 sampling errors | 完成 | `No child process` 错误 | 不用于完整运行 |
| 已安装 py-spy 0.4.2 | 248 samples，0 sampling errors | 完成 | time 回执 exit=0 | 可用于后续独立诊断 |

第二次原始 stdout 在工具调用记录中，未重建成所谓原始日志；持久化的
speedscope 文件、时间/退出回执和资源域回执可独立核验。一次短验证不保证
长运行零开销；后续运行必须标 profiling，不作为未插桩的正式性能点。
只采父控制进程的持 GIL 栈，不据此推断 GPU kernel、native 子进程或整个
分布式系统 CPU 占比；保留请求阶段时间线以解释采样覆盖范围。

## 依据、保护与下一步

- [Python asyncio 官方说明](https://docs.python.org/3.13/library/asyncio-dev.html)
  说明同步 CPU 工作会阻塞同一事件循环；它支持测量方向，不证明本机根因。
- [vLLM profiling 文档](https://docs.vllm.ai/en/latest/contributing/profiling/)
  将 Python 路径剖析与 GPU profiling 区分；本轮问题先测前者。
- [py-spy 官方说明](https://github.com/benfred/py-spy)支持启动被采样子进程；
  没有改变 ptrace 权限或全局 sysctl。
  [0.4.2 发布说明](https://github.com/benfred/py-spy/releases/tag/v0.4.2)
  不足以认定旧版本 ECHILD 的具体根因，以本机退出回执为准。

原始路径 `results/ieee_tc/p2_backend_qualification/d102_20260928/`；
curated `paper_results/ieee_tc/p2_backend/20260928_d102_snapshot_cpu_diagnosis.json`。
保留失败输入探针和旧采样器错误，不覆盖 Full8。按完整计划继续 Prime Full
主线；7B、warm/Resident、基线、M1/M2、消融和敏感性仍未完成。
