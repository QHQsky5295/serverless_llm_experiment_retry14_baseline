# D507：完整 offered W0 的失败观测与复现边界

本轮已有 4,000 条计划请求全部到达、提交并进入终态，不再重启 D507。
它不是合格的完整性能点，也不能作为 HydraServe 原生性能的无条件结论。
原冻结配置、原始结果和先前的生命周期修正证据保持不变。

## 观测值

| 项目 | 实测 / 审计值 |
|---|---:|
| 计划 / 到达 / 提交 / 终态 | 4,000 / 4,000 / 4,000 / 4,000 |
| V1 原生合同完整响应 / 失败 | 3,876 / 124 |
| 失败原因 | 124 条共同 1,800 秒请求保护超时 |
| 共同联合 SLO | 6 / 4,000 = 0.15% |
| 响应集合平均 / P95 TTFT | 980.773 / 1,760.020 秒 |
| 响应集合平均 TPOT | 62.544 毫秒 |
| 响应集合平均 instance-assignment 等待 | 943.189 秒 |
| 响应集合 assignment-to-first-token | 37.579 秒 |
| 真实远端传输 | 132 次，131,480,060 线上字节 |
| 完整生命周期 GPU-s | 未知：3 个 release 记录缺失 |
| 已观测 GPU-s 下界 | 22,932.369 |
| 恢复时确认全部释放对应的保守上界 | 24,496.390 |

延迟仅以 3,876 条完整响应为条件集合；失败仍在 SLO 的 4,000 分母中。
输出原生 token 总数 435,123；没有把失败生成算作有效吞吐。
原始 `n_correct` 仍为未知；V1 的来源/adapter/原生数量/计时合同核验不冒称
独立数值等价证明。未验证的多 rank、迁移、consolidation 不宣称已经验证。

## 两类问题必须分开

1. 首个超时之前，已有 3,632 条完整响应，平均 TTFT 927.862 秒、平均分配
   等待 889.834 秒。这证明长排队并不是后续取消错误才造成的。客户端提交
   延迟远小于该量级，不能将排队减去、移出用户时延。
2. 124 次客户端超时后，服务端有 105 次 `generator already executing`，
   其余 19 次记录为取消。105 次错误全部在对应客户端超时之后，平均相隔
   2.064 毫秒。适配层在工作线程执行 `next()` 时，从事件循环同时执行
   `stream.close()`，违反生成器的单执行者约束。它可能影响后续响应和
   请求计数释放；不能把整个失败运行都归为 HydraServe 原生策略缺陷。

Python 明确规定正在执行的生成器不可再次调用其迭代器方法；取消等待线程
的协程也不等于终止线程本身。依据：[Python 3.10 生成器语义](https://docs.python.org/3.10/reference/expressions.html#generator-iterator-methods)、
[AnyIO 工作线程](https://anyio.readthedocs.io/en/stable/threads.html)、
[aiohttp handler cancellation](https://docs.aiohttp.org/en/stable/web_advanced.html#web-handler-cancellation)。
实际安装的 Starlette `iterate_in_threadpool`、本次执行的适配层及官方
scheduler 源码同时留存；在线文档不替代本地实际版本证据。

## 资源收尾与缺失边界

全部请求终态后，服务没有在既有 60 秒收尾保护内完成自然释放。外置启动器
执行强制收尾，留下 `released=false`、缺失末次 GPU census 和 3 个 release。
原始 launcher classification 保留 `protocol_or_launcher_error`；不覆盖为
请求全部成功，也不让这个后续错误抹去先发生的请求超时。

新的恢复回执核对原 Invocation、inode、全部子组实际为空以及 GPU 进程为空，
只关闭本轮空 scope。恢复时刻不是推断出的精确释放时刻。CPU/GPU、制品服务
和远端监控均已实际结束，历史的失败收尾记录不修改。服务峰值 54,397,161,472
字节，5,773 个样本中 high/max/OOM/OOMkill/swap 均为零。

Podman 部分容器仍显示陈旧 Running 状态；原 PID/启动身份和 cgroup 已消失。
另存 State 与五个仍有记录的容器原生日志，不据状态标签假称进程仍在推理。
一个容器已经由原生 idle 路径删除，有真实 release 记录，不能恢复其已删除
容器日志。此次没有删除其他容器或接触已有 k3s。

## 后续裁决

遵循用户 2026-10-13 的补充要求，异常差基线要先排除不合理适配。下一步先
核查取消所有权并做不依赖 GPU 的实际线程/连接取消测试；同时依据原生运行
日志、冻结配置和官方实现解释首个超时前的分配等待与后端等待。仅允许有证据
的必要适配修复，不扩大请求保护、不改变共同 SLO、不篡改官方调度策略来凑
达标结果。旧失败全部保留；修复影响后续服务时，新执行键重测受影响全回放。
完成忠实 HydraServe 结果和全部可用基线比较前，不启动新的 PrimeLoRA 候选。

输入、源文件、全部请求、失败路径的五次 first-touch 传输、原生 worker 身份、
资源边界与下/上界均已通过新的失败感知离线审计。原成功专用 `audit.py` 未执行。
第一次离线审计遗漏失败请求拥有的传输而停止，其源代码和错误保留；修订只补入
这些失败请求的真实 receipt，不删除它们的传输成本或改写成成功响应。

本节数据文件位于 `paper_results/ieee_tc/hydraserve/d507/failed_w0_summary.json`；
原始记录保留于原项目 `results/ieee_tc/hydraserve_qualification/d507_20261013/`。
