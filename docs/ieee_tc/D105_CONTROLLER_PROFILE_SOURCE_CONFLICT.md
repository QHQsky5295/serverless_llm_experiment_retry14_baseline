# D105：完整诊断在目标来源身份检查处中断

2026-09-28。开发诊断失败，不进入正式性能排名；没有可用 CPU 占比结果。
运行源码为已提交并备份的 `00f0f8da3061f89cfbb857200da1ac9b20fd1560`。

## 问题与控制条件

在 D104 来源集合有效性修正后，再进行一次完整 3B Full/W0 诊断，试图取得
控制进程的实际 CPU 调用栈，解释 Full8 中的长队列。复用原 4,000 请求、
500-adapter 集合、D89 profile、真实远端只读交付缓存、60 秒准备和
1,800 秒保护期限。与 D103 相比配置仅替换输出、NVMe、HOST 三处独占路径。
没有增加新权重、负载、运行中调参或请求内压缩，也没有修改论文公式。

使用同一受限启动入口和外置回放；既有 py-spy 0.4.2 仅采样控制父进程
（100 Hz、GIL、threads、speedscope），原生 worker 不重复经过采样器。
插桩可能影响时序，因此即使完成也只是瓶颈诊断，不是无插桩主性能点。

## 结果状态表

| 项目 | 原始记录 |
|---|---:|
| 计划请求 | 4,000 |
| 实际到达 / 提交 | 59 / 58 |
| 保存的终态记录 | 24 |
| 成功且原生生成合同匹配 | 9 |
| 取消记录 | 15 |
| physical summary 非中断终态 / 中断 | 9 / 15 |
| 完整 GPU-s | N/A：运行中断 |
| 已观察 GPU-s | 356.32732690899866 |
| 分配租约 / 确认释放 | 4 / 4 |
| 资源采样点 | 190 |
| 服务内存峰值 | 19,528,437,760 B |
| 最小主机可用内存 | 94,120,902,656 B |
| high / max / OOM / swap / 安全告警 | 均为 0 |
| 关联远端获取 | 11，均 published 且内容验证通过 |
| 接收字节 / 服务端写出字节 | 25,563,136 / 25,563,136 |
| 请求内打包次数 | 0 |
| speedscope / time 文件 | 均为 0 B，不可用于归因 |

不为已提交却没有保存终态的请求补造结果，也不把尚未到达的请求写成超时。
4 个分配租约不等于 4 个已就绪副本。资源量只是中止前的观察值；不与完整
运行比较成本或计算改善比例，不以 9 条成功请求声明完整正确性。

## 直接原因与尚未证明的部分

主异常为 `ValueError: native HOST target source identity changed`，位于
`scripts/run_all_experiments.py:17137` 的 `_queue_ieee_native_host_preparation`。
执行前取到原生来源快照，然后将当前同整数 adapter ID 对应的
`(adapter_id, lora_path)` 与准备任务的 `(adapter_id, source_path)` 比较；
不一致就抛错。该检查发生在本任务取得文件引用及调用物化操作之前。

这与 D103 在原生替换 owner 内的来源集合覆盖检查不是同一报错位置。
不能仅因错误不同就宣称 D104 已通过完整负载验证，也不能把本次错误
归因为 CPU 采样器。合法跨层副本选择与错误名称/内容是不同情况，需分别验证。

失败计划 `94a17251eb0f4daba317658ae372257c` 保存了 4 次 deferred、1 次
completed 尝试，最终为 failed，close receipt 的 closed=true。
6 条 native HOST preparation 记录均未保存后续 file-held attempt。
它们的目标路径都位于本次 NVMe 目录，但**失败检查所见的来源快照和具体
不匹配项没有保存**；不能据此确定失败 adapter、究竟名称还是路径不同，
更不能断言两份工件内容相同。无 attempt 也不意味着每个任务都失败，GPU-ready
复用可以在创建 file-held attempt 前返回。完整原始记录保留供后续定位。

服务异常之后回放连接断开，外层启动器记录
`protocol_or_launcher_error: external replay failed`，service=-15、replay=1。
保留外层分类，同时保留更早的准备错误。采样器虽输出进程结束消息，但未
完成文件写出；空文件不算完成 profile，本次不能生成 CPU 热点排名。

## 处理与主线

1. 已确认 GPU/主进程退出，按实际 PID/invocation 关闭两端本实验服务，
   清理空资源域；远端原始日志按进程 clock 匹配后只复制一次。
2. 147 项历史保护、44 项运行来源 SHA 核查通过；本轮没有新生产代码改动。
   D104 的 1,241 项已通过检查对应源码未变，不反复重跑同一测试。
3. 下一步限一个 CPU 反例：用实际 file/native owner 和准备队列，检验同一
   adapter 在 NVMe/HOST 的合法副本与 native 注册来源不一致时的行为。
   先对照 D98 选中副本身份处理、D104 来源有效性和官方 vLLM 来源注册语义。
   未有因果证据前不删除检查、不 catch-and-retry、不改 deadline、不直接重跑。
4. 仍需完成普通无插桩 Full 正确性与正式共同 SLO 资格。7B、warm/Resident、
   基线比较、M1/M2、A1–A5、S1–S13 均未因本次诊断完成。

原始目录：`results/ieee_tc/p2_backend_qualification/d105_20260928/`。
结构化结果：`paper_results/ieee_tc/p2_backend/20260928_d105_controller_profile2_failure.json`，
SHA `3e2fcd6c1768dd86bac29af5d08d6ec918c91a8b9a2fb5439817982a5790f933`。
本表按结果分析技能选择失败状态表达；不为截断数据制作优势图。
