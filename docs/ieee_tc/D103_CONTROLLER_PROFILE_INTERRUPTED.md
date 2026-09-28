# D103：完整控制路径采样中断记录

2026-09-28。未完成的开发诊断，不进入性能排名或 CPU 占比分析。
运行基于已备份的 `e6c783110046842a9aaa0cd52b96e22de8c040de`；生产代码仍为
Full8 的 `2f1bc4ba1d9fb3a1b17428390c4c3c63dbb09f1d`，没有新优化。

## 实验目的与合同

复用原 3B、W0、4,000 请求、500-adapter 集合及真实远端已发布工件，保持
Full8 配置、60 秒准备、1,800 秒保护期限、资源限制和论文公式不变。
新配置仅替换三处本轮独占路径。通过既有受限启动入口，在控制父进程外加入
已验证的 py-spy 0.4.2（100 Hz、GIL、speedscope）；原生模型进程不采样。
包装入口在启动主程序前恢复真实 Python 路径，避免 workers 递归经过采样器。

该插桩本身可能改变调度时序，不能当正式性能点；也不能仅因它出现时运行
中断，就认定它制造了状态错误。必须检查实际失败条件。

## 观察结果

| 项目 | 原始测量/状态 |
|---|---:|
| 原计划请求数 | 4,000 |
| 回放已到达 / 已提交 | 59 / 58 |
| 服务保存的请求结果记录 | 27 |
| 成功且原生生成合同匹配 | 11 |
| 已保存失败但 error_type 未填写 | 1（req_00018） |
| 已保存取消记录 | 15 |
| physical summary 非中断终态 / 中断 | 12 / 15 |
| 完整 GPU-s | N/A：工作负载中断 |
| 实际观察到的 GPU-s | 346.7111347940081 |
| GPU 分配租约 / 确认释放 | 4 / 4 |
| 内存峰值 | 19,720,454,144 B |
| 最小主机可用内存 | 93,658,480,640 B |
| high / max / OOM / swap / watchdog 告警 | 均为 0 |
| 远端关联传输 | 12，均 published 且内容核验通过 |
| 客户端收到 / 服务端写出字节 | 27,884,609 / 27,884,609 |
| 请求内打包 | 0 |
| 完整采样文件 | 未生成，不能计算 CPU 占比 |

27 条记录不等于 27 条正确完成；15 条取消在 physical summary 中单列为
interrupted。余下已提交但没有服务结果的请求，不补造失败时间或 token；
未到达请求也不补作 timeout。4 个分配租约不代表 4 个 ready 实例。
计量保留原计划分母，不用短运行的低资源量宣称优越。

## 中断原因与证据边界

服务主异常为：

```text
ieee_prepare_host: replacement epoch lacks the current owned source/fallback set
```

具体抛出位置为 `IEEEBackendGPUReferences.proactive_host_prepare_and_acquire`
所在的 `faaslora/memory/residency_manager.py:2505`。检查条件包括：

1. 当前原生 CPU cache 是否被冻结 objective 的来源行覆盖；
2. 当前每个来源的名称/路径是否与相应行一致；
3. 当前占用 GPU slot 的 adapter 是否有确认记录。

错误消息未区分哪个子条件失败，现有失败 RPC 也没有返回完整 native 回执。
因此尚不能把“有新来源加入旧计划”写成已证实根因，更不能直接删除覆盖检查。

失败准备计划 `38aa40e1f6fb49a58b1ac539c82837a4` 在 owner epoch=1、全部 GPU
slot 为空时注册。后续保存的明确 deferral 已到 epoch=96/100；adapter
416753 的一次调用为 `native_outcome_unresolved`。这些记录证明执行发生于
变化中的副本状态，不证明失败调用没有产生任何副作用。

调用异常经准备任务收尾传递至 residency 任务，再由主控制循环报告；不是
已被确认的 GPU OOM、磁盘护栏或远端打包阻塞。之后回放连接中断，外层启动器
记录 `external replay failed` / `protocol_or_launcher_error` 并结束其拥有的
服务资源域。**保留该外层分类，但不能用它掩盖更早的 native 准备异常。**

采样器随服务收尾被终止，时间回执为空，speedscope 文件不存在。本轮没有
可用 CPU 归因，也不能沿用短 witness 的成功状态冒称长运行采样通过。

## 处理决定与下一步

- 不修改公式、SLO、超时或被测负载；不直接重跑 GPU 实验。
- 先用现有原生 owner / planner 的 CPU fixture，复现“计划注册后，合法
  请求改变来源集合”的情形，并分别检验上述三个条件。
- 区分正常计划过期与身份/确认损坏。只有在 native 权威边界能够证明
  尚未执行副作用时，才讨论明确的计划失效结果；不能把任意异常转成重试。
- 保留 D99/D100 已验证的所有权、关闭和释放规则，查历史后再决定改动。
- 采样失败另外记录；不同时堆叠诊断工具改造和未经验证的系统优化。

GPU census 与资源域释放已核验；两个 artifact 服务及本轮远端监控按实际
PID/invocation 身份关闭。两端链路均为 1 Gbps/full，没有网络配置改动。
147 项历史保护和 34 项源文件核查通过。一次性交付缓存不重建。

原始目录：`results/ieee_tc/p2_backend_qualification/d103_20260928/`。
结构化结果：`paper_results/ieee_tc/p2_backend/20260928_d103_controller_profile1_failure.json`。
后续仍先完成 Prime Full；7B、warm/Resident、基线、M1/M2、A/S 均未因此完成。
