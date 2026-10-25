# D860：来源记录直接投影同次已核验的文件观察

## 预登记问题与新前提

D858 的 3B W2 联合达成3746/4000（93.65%），距95%差54请求；物理GPU时间
11311.401341s在12083.063870s预算内。独立数值正确性仍未知，不进入最终排名。
D859表明145/185条TTFT违约需要Remote，准入至获得来源是重要剩余包络；
这不证明全部包络都是文件检查，也不承诺本候选足以补齐54条。

当前 `_source_observation` 先执行一次两遍fresh-stat文件遍历，再为签名逐路径
第三次lstat。身份签名及footprint因此来自两次不同的读取。D857已让读引用使用
同一份经过第二遍检查的临时记录，但刻意没有修改source_snapshot、demand_witness
及发布观察。本候选将同一投影原则应用于它们实际共用的 `_source_observation`。
这是删除一次多余读取，不是重复D844/D856的ctypes迁移、D851–D854的线程迁移、
D850的通知改动或D628的发布移锁；不改跨事务核查、规划频率或控制参数。

可证伪假设：复用本次inventory输出的私有stat映射，每路径读取3→2，保持
来源/容量记录及变化拒绝语义；完整W2是否改善必须由后续唯一完整运行判断。
此映射不跨调用、异步间隔或owner锁保存，没有TTL或目录条目缓存。

## 官方实现参考与边界

vLLM 0.30.0 WorkerLoRAManager在受控单线程路径中先验证加载对象再激活/替换；
只借鉴一个事务内复用已验证对象，不声称它实现相同文件容量协议。
[固定版本源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)

Python说明DirEntry.stat可返回缓存元数据；本候选仍使用两遍fresh os.stat，
只删除核验之后第三次读取，不把DirEntry缓存当作当前状态。
[Python 3.12官方文档](https://docs.python.org/3.12/library/os.html#os.DirEntry.stat)

## 保留合同与验证范围

保留每路径的inode/mode/size/link/timestamp校验、预留extent包络、内容SHA、
发布前后独立观察、owner锁、epoch、读引用、LRU、native HOST/GPU检查。只在
完整两遍核验成功后投影，不把此方法称作对非协作外部写入的原子快照。
所有IEEE九公式、同一负载/工件、真实远端、SLO和预算不变。

先在未改生产父版运行两项新测试，确认重复读取及变化拒绝；然后只更改该
投影，复用D857同范围文件归属/发布/原子读测试。无需新增GPU短回放、广泛
后端资格或另建框架。测试只能证明工作量和合同，不证明端到端性能。
本候选尚无新性能结果。后续同版两模型/基线、A1–A5、S1–S13、
数值证明、图表/文档/最终V2交付范围均保留。

## 实际最小验证

父版两项新测试：计数测试的snapshot/witness两个子项均观测3次，未通过2次
断言；第二遍变更拒绝测试已通过。保留原始失败，不改断言或筛选结果。
只改上述投影后88/88项通过：两公开路径均逐文件fresh stat两次、输出相等、
无额外LRU触摸、现有引用不被错误退休；第二遍变更仍撤回来源并只推进一次
epoch。现有发布SHA、预算包络、链接/未知来源、原子首次加载/复用检查均通过。

父版辅助域峰值619311104B，候选676601856B，high/max/OOM/swap均0，未初始化
CUDA；147项历史保护与用户manifest未变。无新生产计时插桩、无GPU运行。
生产只改 `faaslora/memory/residency_manager.py` 的来源投影，测试新增于
`tests/test_ieee_tc_file_observation.py`；raw/d860的check/close/backup复用旧入口，
保存原始日志、执行源码和SHA。下一是同合同完整3B W2，不将本检查称为达标。
