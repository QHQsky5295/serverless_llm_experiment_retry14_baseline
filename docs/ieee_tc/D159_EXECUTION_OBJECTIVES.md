# D159：冻结准备目标的 CPU 执行位置验证

状态：候选实现与正确性/隔离回归通过；完整性能回放尚未执行，不是性能结论。

## 假设与历史依据

D158 的 7B Full 完成 4,000 条原生生成合同，平均 TTFT 6.410 秒，
其中调度前等待 4.347 秒；原生末 token 到控制器完成另有 0.679 秒。
这些阶段包含多种等待，不能直接归因于本次函数。
D153 保存的 D143 调用栈中，18/16 个 inclusive observations 分别进入
file/GPU execution objective；这是历史定位线索，不是当前 CPU 百分比。
D158 serving 源码仍在主事件循环内构造、深拷贝、规范序列化和验证这两个
纯计算对象。已存在的单进程 planner 只完成选择及输入验证。

可证伪假设：将相同纯目标构造合入原来的一次 planning transaction，
能减少请求事件循环被占用的时间，并在完整回放中减少非生成等待。
若序列化、worker 排队或更陈旧的观察抵消收益，则不宣称改善。

## 原始依据与适用边界

- [vLLM 官方 CPU/异步优化说明](https://vllm.ai/blog/2024-09-05-perf-update)：
  CPU 控制工作会妨碍及时服务；分离进程可以减少 GIL 争用。
  其 H100、版本及加速倍数不作为本项目证据。
- [Python 3.12 asyncio 文档](https://docs.python.org/3.12/library/asyncio-dev.html#running-blocking-code)：
  非让出的 CPU 工作阻塞同一事件循环；进程执行需要显式管理并发与数据边界。

## 本次唯一改动

沿用单个、受服务 cgroup 限制的 CPU worker，选择、冻结验证、GPU/file
目标构造在一次事务中完成。调用原来的 objective 函数及原始校验器，
不改变九个公式、h/d、候选、预算、目标、tie-break 或成本 profile。
随机 file plan ID 随目标创建；它只是标识，不提前注册或保留资源。
sealed plan 仍无可变共享别名；普通导出重新验证，不能携带“已验证”布尔值
绕过校验。执行使用同一 bundle，不再次传回整个计划排队计算。

pre-init GPU 目标依赖真实初始化快照，仍等待 engine-ready 后取得该快照，
再交同一个 worker 构造；不预测或伪造就绪。原 live owner/epoch、物理
reservation、替换、提交、取消和 cleanup 检查全部保留。新 await 之前已有
preparation task ownership，取消须等待计算返回并丢弃结果。
无新增 worker、同步失败回退、容量/超时提升、远端改动或新制品。

## 验证边界及后续

先比较原函数与 fused 输出（含 SHA），检查可变输入隔离、原校验拒绝、
取消收尾及现有执行回归；组件计时不冒称请求收益。
通过后仍须 D158 相同 cap4/profile/config 的普通 7B W0 4,000 请求回放。
共同数值正确性、warm/Resident 标定和 G1/G2 验收仍未完成。
磁盘低于 150 GiB 时不启动新的重型回放，不降低门槛或删除原始结果。

首次 targeted1：9 项中 8 项通过，1 项报错，证据保留。原因是 handoff
计划合同不包含 residency replacement 的 `size_edges_bytes` 字段；候选
错误地把它当共同字段。修正为显式携带已有冻结 profile 的边界，与原执行器
的输入一致；不填默认值或移除检查。将重新验证同一候选。

targeted2 全部 9 项通过。regression1 的 816 项中 809 项通过、2 failure、
5 error；7 个问题均在已有独立 file-only 执行合同中：它不涉及 native GPU，
不拥有 native profile。候选不应无条件要求该 profile。修正为按明确的
`source_view` 合同传入 native 类边界；file-only 用 N/A，不伪造边界，任何
GPU target 缺边界仍显式拒绝。没有放宽测试或延长测试等待，保留失败日志。

## 资格结果表

| 检查 | 数量 | 结果 | 解释 |
|---|---:|---|---|
| targeted1 | 9 | 8 通过、1 error | handoff 字段合同问题，保留 |
| targeted2 | 9 | 全通过 | 真进程、输出等价、不可变边界和取消 |
| regression1 | 816 | 809 通过、2 failure、5 error | 纯文件合同问题，保留 |
| regression2 | 816 | 全通过 | 修正独立合同后原测试不变 |

最后回归主体 139.897 秒，完整命令 152.11 秒，峰值 RSS 1,199,084 KiB。
所有检查采用 3/4 GiB、swap=0、CPU 2/3/26/27 的受限组；结束后均验证
组和 GPU 进程为空。自动移除后不可恢复的最终 cgroup 事件标为 unavailable，
不补填零。测试时间不是 serving latency，也不做 seed CI。

依据 `academic-plotting` 的数据对应原则及批准计划 §11，本节选择资格状态表，
不把单元测试数量画成系统性能优势图。历史 D158 性能值只用于提出待检验假设。

结论：允许进入同条件 Full 验证，不宣布采纳性能增益。需同时检查 IPC 字节、
worker 等待、计划陈旧/取消、远程获取、TTFT、TPOT 与物理 GPU-s；不能只选
改善的阶段。7B 未正式验收，不进入 3B 或外部基线比较。

## 回放前容量恢复

复用 D152 公开 wheel 下载缓存审计，不触碰已安装环境、vLLM 编译缓存、
模型、LoRA、trace 或原始结果。对 5 个 PyPI 官方 wheel 核对公开 URL、
大小和 SHA、HEAD 可获取性、独占普通文件、打开引用、软硬链接及项目引用；
按审定清单移除归档及对应 HTTP 头，共 10 文件，释放 2,334,081,024 字节。
清理后可用 162,690,375,680 字节（约 151.52 GiB）；正式启动仍重新检查。
复现环境和各项目跟踪状态前后不变。审计和应用的实际受限组均已退出，
全过程没有 GPU 推理运行或远端服务调整。
